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
    optim = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=0)
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
        batch_size=int(args.global_batch_size // world_size), # 8B % 8 = 0 
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.datadir})")

    # Prepare models for training:
    model.train() # this implementation uses dropout for disabling guidance, for unguided training for CFG, not recommended
    ema.eval() # EMA model should always be in eval mode

    # Variables for monitoring/logging:
    log_steps = 0
    running_loss = torch.tensor(0.0, device=device)
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
    # Setup classifier-free guidance for inference (this is not for FID eval):
    if use_cfg:
        x_inference = torch.cat((x_inference,x_inference), dim=0) # replicate the x0 for guided and unguided inference (2B, C, H, W)
        y_null = torch.tensor ([args.num_classes] * inference_bs, device=device) #(B) class labels [0, num_classes) exclusive so num_classes is designed to be the null label
        y_inference = torch.cat((y_inference, y_null), dim=0) # (2B)
        inference_model_args = dict(y=y_inference, cfg_scale = args.cfg_scale)
        model_fn = ema.forward_with_cfg
    else:
        inference_model_args = dict (y=y_inference)
        model_fn = ema.forward

    # TODO: verify consistency
    use_latent = True if "latent" in args.dataset and args.latents_precomputed else False
    logger.info (f"Training for {args.epochs} epochs...")

    for epoch in range (init_epoch, args.epochs + 1):
        sampler.set_epoch(epoch)
        logger.info(f"Beinning epoch {epoch}...")
        for i, (x, y) in tqdm(enumerate(loader)):
            # for all intents and purposes since we arent doing gradient accumulation, batch_size might as well be world_size * local_batch_size
            # and each backward syncs gradients for batch_size 8B
            # adjust _learning_rate(optim, i / len(loader) + epoch, args) # len loader = len(dataset) / (world_size * local_batch_size)
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
            loss.backward() # by default requires_backward_grad_sync is true for DDP; so every replica has synchronized gradients (basically averaged gradients) g(L)/8 for all i
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optim.step()
            memory_after_backward = torch.cuda.memory_allocated(device)
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.detach() # equivalent to loss.detach() for logging, (stops gradient tracking)
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time) # average velocity (total steps / total time)
                # Reduce loss history over all processes:
                running_loss = running_loss / log_steps
                # avg_loss = torch.tensor(running_loss / log_steps, device = device)
                dist.all_reduce(running_loss, op=dist.ReduceOp.AVG)
                #avg_loss = avg_loss.item() / world_size
                logger.info(
                    f"(step={train_steps:08d}) Train loss: {running_loss:.4f}, "
                    f"Train Steps/Sec: {steps_per_sec:.2f}, "
                    f"GPU Mem before forward: {memory_before_forward/10**9:.2f}GB, "
                    f"GPU Mem after forward: {memory_after_forward/10**9:.2f}GB, "
                    f"GPU Mem after backward: {memory_after_backward/10**9:.2f}GB, "
                )

                # Reset monitoring variables:
                running_loss.zero_()
                log_steps = 0
                start_time = time()
        
        if rank == 0: # if master_process
            # latest checkpoint
            if epoch % args.save_content_every == 0:
                logger.info("Saving content.")
                content = {
                    "epoch" : epoch + 1, # epoch to resume from
                    "train_steps" : train_steps,
                    "args" : args,
                    "model" : model.module.state_dict(),
                    "optim" : optim.state_dict(),
                    "ema" : ema.state_dict(),
                }
                torch.save(content, os.path.join(checkpoint_dir, "content.pth"))
            
            # save model checkpoint
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
        
        # run inference once in a while
        if rank == 0 and epoch % args.plot_every == 0:
            logger.info("Generating EMA samples...")
            with torch.no_grad():
                sample_fn = flow.sampler.sample_ode() # Default to ODE simulation to trace flows
                # TODO: cryptic change implementation for better readability
                # i dont like this implementation, where we send x0,y and x0,null
                # separately by construction to the ODE solver, makes it cryptic
                # better have a single function forward, looks at cfg scale if not 1.0, replicates and stacks caption or null,
                # then calls forward with cfg and forward without cfg on them, combines with cfg forward pass
                # gets the velocity and takes a small step in that direction
                samples = sample_fn(x_inference, model_fn, **inference_model_args)[-1] # TODO: look into design decision for [-1] access
                if use_cfg: # remove null samples; Gnarly, I dont like this
                    samples, _ = samples.chunk(2, dim=0)
                # TODO: VAE consistency
                samples = vae.decode(samples / 0.18215).sample
            
            # Save and display images:
            save_image(samples, f"{sample_dir}/image_{epoch:.08d}.jpg", nrow=4, normalize=True, value_range=(-1,1))
            del samples
        
        # Run eval once in a while
        if epoch % args.eval_every == 0 and epoch > 0:
            ref_dir = Path(args.eval_refdir)
            if ref_dir.exists():
                # TODO: mandate consistent flags
                eval_batch_size = args.eval_batch_size
                using_cfg = args.eval_cfg_scale > 1.0
                global_batch_size = eval_batch_size * world_size
                # total samples = next multiple of global batch size
                total_samples = int(math.ceil(args.n_eval_samples / global_batch_size) * global_batch_size)
                samples_needed_this_gpu = int (total_samples // world_size)
                iterations = int (samples_needed_this_gpu // eval_batch_size)
                pbar = range(iterations)
                pbar = tqdm(pbar) if rank == 0 else pbar
                total = 0
                p = Path(experiment_dir) / f"fid{args.n_eval_samples}_epoch{epoch}"
                # if p.exists() and rank == 0:
                #   shutil.rmtree(p.as_posix())
                p.mkdir(exist_ok=True, parents=True)

                model.eval()
                for _ in pbar:
                    # Sample inputs:
                    eval_x = torch.randn(eval_batch_size, 4, latent_size, latent_size, device=device) # (B, C, H, W)
                    eval_y = None if not use_label else torch.randint (low=0, high=args.num_classes, size=(eval_batch_size,), device=device) # (B)
                    # Setup classifier free guidance for FID/Eval:
                    if use_cfg:
                        eval_x = torch.cat((eval_x,eval_x), dim=0) #(2B, C, H, W)
                        y_null = torch.tensor([args.num_classes] * eval_batch_size, device=device) #(B)
                        eval_y = torch.cat((eval_y, y_null), dim=0) #(2B)
                        eval_model_args = dict (y = eval_y, cfg_scale=args.eval_cfg_scale)
                        eval_model_fn = ema.forward_with_cfg # model.forward_with_cfg
                    else:
                        eval_model_args = dict (y=eval_y)
                        eval_model_fn = ema.forward # model.forward
                    
                    # Sample images:
                    with torch.no_grad():
                        sample_fn = flow_sampler.sample_ode() # default to ODE sampling
                        samples = sample_fn(eval_x, eval_model_fn, eval_model_args)[-1] # TODO: look into design decision for [-1] access
                    
                    # discard null class (unguided) samples
                    if using_cfg:
                        samples, _ = samples.chunk(2, dim=0)
                    
                    # TODO: mandate VAE consistency
                    samples = vae.decode(samples / 0.18215).sample
                    samples = (
                        # TODO: FIX: the ablation to inversal to follow the repo
                        torch.clamp ((samples + 1.0) * 127.5, 0, 255)
                        .permute(0, 2, 3, 1) #(B, H, W, C)
                        .to("cpu", dtype=torch.uint8)
                        .numpy()
                    )
                    
                    # Save samples to disk as individual .png files
                    for i, sample in enumerate(samples):
                        index = i * world_size + rank + total
                        if index >= args.n_eval_samples:
                            break
                        pp = p / f"{index:06d}.jpg"
                        Image.fromarray(sample).save(pp.as_posix())
                    total += global_batch_size
                
                model.train()
                eval_args = dnnlib.EasyDict()
                # reference images
                eval_args.dataset_kwargs = dnnlib.EasyDict(
                    class_name="training.dataset.ImageFolderDataset",
                    path=ref_dir.as_posix(),
                    xflip=True,
                )
                # generated images
                eval_args.gen_dataset_kwargs = dnnlib.EasyDict(
                    class_name="training.dataset.ImageFolderDataset",
                    path=p.resolve().as_posix(),
                    xflip=True,
                )
                progress = metric_utils.ProgressMonitor(verbose=True)
                if rank == 0: # if master_process
                    print("Calculating FID...")
                result_dict = metric_main.calc_metric(
                    metric="fid2k_full",
                    dataset_kwargs=eval_args.dataset_kwargs,
                    num_gpus=world_size,
                    rank=rank,
                    device=device,
                    progress=progress,
                    gen_dataset_kwargs=eval_args.gen_dataset_kwargs,
                    cache=True,
                )

                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=experiment_dir, snapshot_pkl=p.as_posix())
                del result_dict, samples
                gc.collect()
                torch.cuda.empty_cache()
            else:
                print(f"Reference directory {ref_dir} does not exists, skipping eval...")
            dist.barrier()
    
    model.eval() #important! This disables randomized embedding dropout, hack for CFG unguided training
    # do any sampling/FID calcs with EMA or model in eval mode...
    logger.infor("Done!")
    cleanup()


def none_or_str(value):
    if value == "None":
        return None
    return value


if __name__ == "__main__":
    # Default args here will train the model with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", help="name of dataset")
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, default="MambaDiffV1_XL_2")
    parser.add_argument("--image-size", type=int, choices=[256, 512, 1024], default=256)
    parser.add_argument("--num-in-channels", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--label-dropout", type=float, default=-1)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=25)
    parser.add_argument("--save-content-every", type=int, default=5)
    parser.add_argument("--plot-every", type=int, default=5)
    parser.add_argument("--model-ckpt", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--learn-sigma", action="store_true")
    parser.add_argument(
        "--bimamba-type",
        type=str,
        default="v2",
        choices=["v2", "none", "zigma_8", "sweep_8", "jpeg_8", "sweep_4", "jpeg_2"],
    )
    parser.add_argument("--cond-mamba", action="store_true")
    parser.add_argument("--scanning-continuity", action="store_true")
    parser.add_argument("--enable-fourier-layers", action="store_true")
    parser.add_argument("--rms-norm", action="store_true")
    parser.add_argument("--fused-add-norm", action="store_true")
    parser.add_argument("--drop-path", type=float, default=0.0)
    parser.add_argument("--use-final-norm", action="store_true")
    parser.add_argument(
        "--use-attn-every-k-layers",
        type=int,
        default=-1,
    )
    parser.add_argument("--not-use-gated-mlp", action="store_true")
    # parser.add_argument("--skip", action="store_true")
    parser.add_argument("--max-lr", type=float, default=1e-4)
    parser.add_argument("--pe-type", type=str, default="ape", choices=["ape", "cpe", "rope"])
    parser.add_argument("--learnable-pe", action="store_true")
    parser.add_argument(
        "--block-type",
        type=str,
        default="linear",
        choices=["linear", "raw", "wave", "combined", "window", "combined_fourier", "combined_einfft"],
    )
    parser.add_argument("--no-lr-decay", action="store_true", default=False)
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=2.0,
    )

    group = parser.add_argument_group("Eval")
    group.add_argument("--eval-every", type=int, default=100)
    group.add_argument("--eval-refdir", type=str, default=None)
    group.add_argument("--n-eval-samples", type=int, default=1000)
    group.add_argument("--eval-batch-size", type=int, default=4)
    group.add_argument("--eval-cfg-scale", type=float, default=1.0)

    group = parser.add_argument_group("MoE arguments")
    group.add_argument("--num-moe-experts", type=int, default=8)
    group.add_argument("--mamba-moe-layers", type=none_or_str, nargs="*", default=None)
    group.add_argument("--is-moe", action="store_true")
    group.add_argument(
        "--routing-mode", type=str, choices=["sinkhorn", "top1", "top2", "sinkhorn_top2"], default="top1"
    )
    group.add_argument("--gated-linear-unit", action="store_true")

    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument("--loss-weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
    group.add_argument("--sample-eps", type=float)
    group.add_argument("--train-eps", type=float)
    group.add_argument(
        "--diffusion-form",
        type=str,
        default="none",
        choices=["none", "constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing", "log"],
        help="form of diffusion coefficient in the SDE",
    )
    group.add_argument("--t-sample-mode", type=str, default="uniform")
    group.add_argument("--use-blurring", action="store_true")
    group.add_argument("--blur-sigma-max", type=int, default=3)
    group.add_argument("--blur-upscale", type=int, default=4)

    args = parser.parse_args()
    main(args)



    


