# utils
import gc
import math
import sys
from pathlib import Path

import argparse
import logging
import os
from collections import OrderedDict
from copy import deepcopy
from time import time
from tqdm import tqdm


import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# DDP
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# VAE
from diffusers.models import AutoencoderKL

from create_model import create_model
from datasets_prep import get_dataset
from fmamba_modules import interpolate_pos_embed

# dataset scaffolding
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import save_image

# Flow matching
from flow import Sampler, create_flow

# for easy imports from eval toolbox
eval_import_path = (Path(__file__).parent.parent / "eval_toolbox").resolve().as_posix()
sys.path.append(eval_import_path)
# import dnnlib from evaltoolbox for convenience classes like EasyDict
import dnnlib
# for FID evaluations
from pytorch_fid import metric_main, metric_utils


# -------------------------------------------------------------------------------------------------------------------------
#  Training Helper Fucntions
# -------------------------------------------------------------------------------------------------------------------------

# exponential moving average for weight update
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    
    # can use list comprehension alternative instead as well dict = {n:p for n,p in ...}
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: consider applying only to params that require_grad to avoid small numerical changes in pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha = 1 - decay)

def requires_grad (model, flag = True):
    """
    Set requires_grad flag for all parameters in a model.
    """

    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()

def create_logger (log_dir):
    """
    Create a logger that writes to a log file and stdout.
    """

    if dist.get_rank() == 0: # master process
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{log_dir}/log.txt")],
        )
        logger = logging.getLogger(__name__)
    else: # dummy logger (does nothing) for non master processes
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

# TODO: might want to remove this if not used anywhere
def center_crop_arr (pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX) # average pixels in 2x2 BOX for downsampling
    
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y:crop_y + image_size, crop_x : crop_x + image_size])


# TODO : Ablation linear warmup that starts at 1 / warmup, change if doesnt work out
# TODO: Change lr to max_lr in args
def adjust_learning_rate (optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after linear warmup"""
    if epoch < args.warmup_epochs:
        lr = args.max_lr * (epoch+1) / args.warmup_epochs
    else:
        decay_ratio = (epoch - args.warmup_epochs) / (args.max_epochs - args.warmup_epochs)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        lr = args.min_lr + (args.max_lr - args.min_lr) * coeff
    
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    
    return lr

# ---------------------------------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------------------------------


def main (args):
    """
    Core training loop
    """

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # setup DDP:
    dist.init_process_group("nccl")
    # if "SLURM_PROCID" in os.environ:
    #   rank = int(os.environ["SLURM_PROCID"])
    #   gpu = rank % torch.cuda.device_count()
    #   world_size = int(os.environ["WORLD_SIZE"], 1)
    # else:
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    world_size = dist.get_world_size()
    assert args.global_batch_size % world_size == 0, "Batch size must be divisible by world size"
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}")

    # setup an experiment folder
    # TODO: maintain consistent argument flag attributes
    experiment_index = args.experiment_index
    experiment_dir = f"{args.results_dir}/{experiment_index}" # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints" # Directory for model checkpoints
    sample_dir = f"{experiment_dir}/samples" # Inference directory for specific experiment

    if rank == 0: # if master_process
        os.makedirs(experiment_dir, exist_ok=True) # Make results folder, holds all sub folders
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)
        logger = create_logger(experiment_dir) # creates log.txt in the experiment directory
        logger.info(f"Experiment directory created at {experiment_dir}")
    
    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    # instaniate model
    model = create_model(args)
    # Note that parameter initialization is done within the constructor of the model
    ema = deepcopy(model).to(device) # create an EMA of the model for use after training
    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=False)
    flow = create_flow (
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps,
        path_args = {
            "diffusion_form" : args.diffusion_form,
            "use_blurring" : args.use_blurring,
            "blur_sigma_max" : args.blur_sigma_max,
            "blur_upscale" : args.blur.upscale,
        },
        t_sample_mode = args.t_sample_mode,
    ) # default: vector field / velocity

    flow_sampler = Sampler(flow)
    # Runtime VAE option incase latent are not precomputed
    if not args.latents_precomputed:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"Backbone Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup Optimizer (default Adam Betas=(0.9, 0.999) and a constant learning rate 1e-4)
    # TODO: Ablate weight decay regularization
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.max_epochs, verbose=True)

    update_ema(ema, model.module, decay=0)    # Ensure EMA is inialized with synced weights

    if args.model_ckpt and os.path.exists(args.model_ckpt):
        checkpoint = torch.load(args.model_ckpt, map_location=torch.device(f"cuda:{device}"))
        # TODO: Keep track of epoch vs init_epcoh consistency
        epoch = int (os.path.split(args.model_ckpt)[-1].split(".")[0]) # ckpoints are saved checkpoint_dir/{epoch}.pt
        init_epoch = 0

        state_dict = model.module.state_dict()
        for i, k in enumerate(["x_embedder.proj.weight", "final_layer.linear.weight", "final_layer.linear.bias"]):
            # Note: Fixed for DiM-L/4 to DiM-L/2
            if k in checkpoint["model"] and checkpoint["model"][k].shape != state_dict[k].shape:
                if i == 0:
                    K1, K2 = state_dict[k].shape[2:]
                    checkpoint["model"][k] = checkpoint["model"][k][:, :, :K1, :K2] # state_dict[k]
                    checkpoint["ema"][k] = checkpoint["ema"][k][:, :, :K1, :K2] # state_dict[k]
                else:
                    fan_in = state_dict[k].size(0)
                    checkpoint["model"][k] = checkpoint["model"][k][:fan_in]
                    checkpoint["ema"][k] = checkpoint["ema"][k][:fan_in]
        
        # interpolate position embedding to adapt pos embeddings to different input resolutions/sizes
        interpolate_pos_embed(model.module, checkpoint["model"])
        interpolate_pos_embed(ema, checkpoint["ema"])

        msg = model.module.load_state_dict(checkpoint["model"], strict=True)
        print (msg)

        ema.load_state_dict(checkpoint["ema"])
        optim.load_state_dict(checkpoint["optim"])
        for g in optim.param_groups:
            g["lr"] = args.max_lr

        train_steps = 0

        logger.info("=> loaded checkpoint (epoch {})".format(epoch))
        del checkpoint

    elif args.resume or os.path.exists(os.path.join(checkpoint_dir, "content.pth")):
        # resume training
        checkpoint_file = os.path.join(checkpoint_dir, "content.pth")
        checkpoint = torch.load(checkpoint_file, map_location=torch.device(f"cuda:{device}"))
        init_epoch = checkpoint["epoch"]
        epoch = init_epoch
        model.module.load_state_dict(checkpoint["model"])
        optim.load_state_dict(checkpoint["optim"])
        ema.load_state_dict(checkpoint["ema"])
        train_steps = checkpoint["train_steps"]

        for g in optim.param_groups:
            g["lr"] = args.max_lr
        
        logger.info("=> resume checkpoint (epoch {})".format(checkpoint["epoch"]))
        del checkpoint
    else:
        # clean training run : no checkpoint resume or pretrained weights
        init_epoch = 0
        train_steps = 0
    
    # disable gradient tracking on EMA model
    requires_grad(ema, False)

    # TODO: Try streaming dataset instead if that's convenient
    dataset = get_dataset(args)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.global_seed)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // world_size),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.datadir})")

    # Prepare models for training:
    model.train()
    ema.eval() # EMA model should always be in eval mode

    # Variables for monitoring/logging:
    log_steps = 0
    running_loss = 0
    start_time = time ()
    # Class Label to condition on
    use_label = True if "imagenet" in args.dataset else False

    # TODO: Move this inside sampling loop, might be more convenient
    # noise and label to run inference
    # TODO: ARE WE EVEN USING THIS IN CELEBA? if not VAE then why 4 channel x0 ?
    inference_bs = 2
    # TODO: Figure out how it works for celeba, are we still using VAE latents
    x_inference = torch.randn (inference_bs, 4, latent_size, latent_size, device=device) # (B, C, H, W)
    y_inference = None if not use_label else torch.randint (low=0, high=args.num_classes, size=(inference_bs,), device=device) # None or (B)
    use_cfg = args.cfg_scale > 1.0
    # Setup classifier-free guidance:
    if use_cfg:
        x_inference = torch.cat((x_inference,x_inference), dim=0) # replicate the x0 for guided and unguided inference (2B, C, H, W)
        y_null = torch.tensor ([args.num_classes] * inference_bs, device=device) #(B) class labels [0, num_classes) exclusive so num_classes is designed to be the null label
        y_inference = torch.cat((y_inference, y_null), dim=0) # (2B)
        inference_model_args = dict(y_inference=y_inference, cfg_scale = args.cfg_scale)
        model_fn = ema.forward_with_cfg
    else:
        inference_model_args = dict (y_inference=y_inference)
        model_fn = ema.forward

    # TODO: verify consistency
    use_latent = True if "latent" in args.dataset and args.latents_precomputed else False
    logger.info (f"Training for {args.epochs} epochs...")

    for epoch in range (init_epoch, args.epochs + 1):
        sampler.set_epoch(epoch)
        logger.info(f"Beinning epoch {epoch}...")
        for i, (z, y) in tqdm(enumerate(loader)):
            # adjust _learning_rate(optim, i / len(loader) + epoch, args)
            x = x.to(device)
            y = None if not use_label else y.to(device)

            if not use_latent:
                with torch.no_grad():
                    # Map images to latent space + normalize latents:
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215) # (B, C, H, W)
            model_args = dict (y=y)
            memory_before_forward = torch.cuda.memory_allocated(device)
            loss_dict = flow.training_losses(model, x, model_args)
            loss = loss_dict["loss"].mean()
            memory_after_forward = torch.cuda.memory_allocated(device)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optim.step()
            memory_after_backward = torch.cuda.memory_allocated(device)
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time) # average velocity (total steps / total time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device = device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size
                logger.info(
                    f"(step={train_steps:08d}) Train loss: {avg_loss:.4f}, "
                    f"Train Steps/Sec: {steps_per_sec:.2f}, "
                    f"GPU Mem before forward: {memory_before_forward/10**9:.2f}GB, "
                    f"GPU Mem after forward: {memory_after_forward/10**9:.2f}GB, "
                    f"GPU Mem after backward: {memory_after_backward/10**9:.2f}GB, "
                )

                # Reset monitoring variables:
                runing_loss = 0
                log_steps = 0
                start_time = time()
        
        if rank == 0: # if master_process
            # latest checkpoint
            if epoch % args.save_content_every == 0:
                logger.info("Saving content.")
                content = {
                    "epoch" : epoch + 1, # eoch to resume from
                    "train_steps" : train_steps,
                    "args" : args,
                    "model" : model.module.state_dict(),
                    "optim" : optim.state_dict(),
                    "ema" : ema.state_dict(),
                }
                torch.save(content, os.path.join(checkpoint_dir, "content.pth"))
            
            if epoch % args.ckpt_every == 0 and epoch > 0:
                checkpoint = {
                    "epoch" : epoch + 1,    # next epoch upon resuming
                    "model" : model.module.state_dict(),
                    "ema" : ema.state_dict(),
                    "optim" : optim.state_dict(),
                    "args" : args,
                }
                checkpoint_path = f"{checkpoint_dir}/{epoch:08d}.pt"
                torch.save (checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        if rank == 0 and epoch % args.plot_every == 0:
            logger.info("Generating EMA samples...")
            

            




    


