import argparse
import gc
import math
import os
import sys
from pathlib import Path

import numpy as np

import torch
import torch.distributed as dist

from create_model import create_model
from diffusers.models import AutoencoderKL

from download import find_model
from PIL import Image
from tqdm import tqdm

from flowMatching import Sampler, create_flow

# set eval toolbox in the system path for convenient imports
eval_import_path = (Path(__file__).parent.parent / "eval_toolbox").resolve().as_posix()
sys.path.append(eval_import_path)
import dnnlib # from eval toolbox
from pytorch_fid import metric_main, metric_utils #from eval toolbox

def create_npz_from_sample_folder (sample_dir, image_ext, num=50_000):
    """
    Builds a single .npz file from a folder of .image_ext inference samples.

    Args:
    sample_dir : str; directory of inference samples
    image_ext : str; extension type of the inference samples (.png, .jpeg)
    num :int = 50000; number of images in the inference directory 
    """

    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.{image_ext}")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3) # (B, H, W, 3(RGB))
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print (f"Saved .npz file to {npz_path}: Shape {samples.shape}")
    return npz_path

def main(mode, args):
    """
    Run sampling
    """

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    assert torch.cuda.is_available(), f"Sampling with DDP requires atleast one GPU, sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank:{rank}, seed:{seed}, world_size:{dist.get_world_size()}")

    # Load model:
    latent_size = args.image_size // 8 # VAE compression factor
    model = create_model(args).to(device)
    # Auto-download a pretrained model or load a custom checkpoint from train.py
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    flow = create_flow(args.path_type, args.prediction, args.loss_weight, args.train_eps, args.sample_eps)
    sampler = Sampler(flow)
    if mode == "ODE":
        if args.likelihood:
            assert args.cfg_scale == 1, f"Likelihood is incompatible with guidance"
            sample_fn = sampler.sample_ode_likelihood(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
            )
        else:
            sample_fn = sampler.sample_ode(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
                reverse=args.reverse,
            )
    elif mode == "SDE":
        sample_fn = sampler.sample_sde(
            sampling_method=args.sampling_method,
            diffusion_form=args.diffusion_form,
            diffusion_norm=args.diffusion_norm,
            last_step=args.last_step,
            # last_step_size: 1/num_steps by default
            last_step_size=args.last_step_size,
            num_steps=args.num_sampling_steps,
        )
    
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, f"CFG scale should be greater than or equal to 1 instead of: {args.cfg_scale}"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/","-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    if mode == "ODE":
        folder_name = (
            f"{model_string_name}-{ckpt_string_name}-"
            f"cfg-{args.cfg_scale}-{args.local_batch_size}-"
            f"{mode}-{args.num_sampling_steps}-{args.sampling_method}"
        )
    elif mode == "SDE":
        folder_name = (
            f"{model_string_name}-{ckpt_string_name}-"
            f"cfg-{args.cfg_scale}-{args.local_batch_size}-"
            f"{mode}-{args.num_sampling_steps}-{args.sampling_method}-"
            f"{args.diffusion_form}-{args.last_step}-{args.last_step_size}"
        )
    if args.use_even_classes:
        folder_name = folder_name + "-even-classes"
    # TODO: make sure it reads args.inference_dir everywhere
    inference_folder_dir = f"{args.inference_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(inference_folder_dir, exist_ok=True)
        print (f"Saving .{args.image_ext} samples at {inference_folder_dir}")
    dist.barrier()

    # How many samples we need to generate on each GPU and how many iterations we need to run:
    B = args.local_batch_size
    global_batch_size = B * dist.get_world_size()
    total_samples = int (math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print (f"Total number of images sampled : {total_samples}")
    # total samples 800 = 704 / 80
    assert total_samples % dist.get_world_size() == 0, f"total_samples must be divisible by world_size"
    samples_for_this_gpu = int (total_samples // dist.get_world_size())
    assert samples_for_this_gpu % B == 0, f"Samples to be processed by this GPU:{samples_for_this_gpu} must be divisible by the local_batch_size{B}"
    
    mini_batches = int (samples_for_this_gpu // B)
    pbar = range(mini_batches)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0

    use_label = True if args.num_classes > 1 else False
    if use_label:
        real_num_classes = args.num_classes - 1 # dont count uncond cls
    else:
        real_num_classes = args.num_classes
    
    if args.use_even_classes: # evenly sample labels across all classes
        # [0, 1, 2] * repeat each class (samples_for_this_gpu / num classes) times so that len(classes_list) = samples_for_this_gpu
        CLASSES_LIST = list(range(real_num_classes)) * math.ceil(samples_for_this_gpu / real_num_classes)
    
    for i in pbar:
        # sample inputs:
        x0 = torch.randn(B, model.in_channels, latent_size, latent_size, device=device)
        if args.use_even_classes:
            # end is exclusive [i*B to (i+1)B)
            y = torch.tensor(CLASSES_LIST[i * B: (i+1)*B], device=device)
        else:
            y = None if not use_label else torch.randint(0, real_num_classes, (B,), device=device)
        
        # Setup classifier-free guidance (CFG)
        if using_cfg:
            x0 = torch.cat((x0, x0), dim=0)
            y_null = torch.tensor([real_num_classes] * B, device=device) # (B) tensor of B integers, each of which are null_class token
            y = torch.cat((y,y_null), dim=0)
            model_kwargs = dict (y=y, cfg_scale=args.cfg_Scale)
            model_fn = model.forward_with_cfg if not args.ada_cfg else model.forward_with_adacfg
        else:
            model_kwargs = dict(y=y)
            model_fn = model.forward
        
        samples = sample_fn(x0, model_fn, **model_kwargs)[-1]
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0) # Remove null class samples
            del _
        
        samples = vae.decode(samples / 0.18215).sample

        samples = torch.clamp (127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy() # 0 is centered at 128 uint value approximation of 127.5 * (image + 1)
        # distortion from actual expression is negligible after clamping

        # Save samples to disk as individual .jpg files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{inference_folder_dir}/{index:06d}.{args.image_ext}")
        total += global_batch_size
        dist.barrier()

    # Make sure all processes have finished saving their samples before attempting to conver to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(inference_folder_dir, args.image_ext, args.num_fid_samples)
        print("Done.")
    
    # test FID:
    eval_args = dnnlib.EasyDict()
    eval_args.dataset_kwargs = dnnlib.EasyDict(
        class_name = "training.dataset.ImageFolderDataset",
        path=args.eval_refdir,
        xflip=True,
    )
    eval_args.gen_dataset_kwargs = dnnlib.EasyDict(
        class_name = "training.dataset.ImageFolderDataset",
        path=inference_folder_dir,
        xflip=True,
    )
    progress = metric_utils.ProgressMonitor(verbose=True)
    if rank == 0:
        print ("Calculating FID...")
    eval_metrics = args.eval_metric.split(",")
    for metric in eval_metrics:
        result_dict = metric_main.calc_metric(
            metric=metric,
            dataset_kwargs=eval_args.dataset_kwargs,
            num_gpus=dist.get_world_size(),
            rank = rank,
            device=device,
            progress=progress,
            gen_dataset_kwargs=eval_args.gen_dataset_kwargs,
            cache=True,
        )
        if rank == 0:
            metric_dir = Path(inference_folder_dir) / "metrics"
            metric_dir.mkdir(exist_ok=True, parents=True)
            metric_main.report_metric(result_dict, run_dir=metric_dir.as_posix(), snapshot_pkl=inference_folder_dir)
    
    del result_dict
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    dist.destroy_process_group()

def none_or_str(value):
    if value == "None" or value == "none":
        return None
    return value



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    if len(sys.argv) < 2:
        print("Usage: program.py <mode> [options]")
        sys.exit(1)
    
    mode = sys.argv[1]

    assert mode[:2] != "--", f"Usage: program.py <mode> [options]"
    assert mode in ["ODE", "SDE"], f"Invalid mode. Please choose 'ODE' or 'SDE'"

    parser.add_argument("--model", type=str, default="Fleurdelys-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--inference-dir", type=str, default="Inferences")
    parser.add_argument("--local-batch-size", type=int, default=4)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512, 1024], default=256)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=1337)
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="By default, use TF32 matmuls. Faster passes on Ampere GPUs"
    )
    parser.add_argument("--learn-sigma", action="store_true")
    parser.add_argument("--num-in-channels", type=int, default=4) # VAE latent space channels
    parser.add_argument("--label-dropout", type=float, default=-1)
    parser.add_argument("--use-final-norm", action="store_true")
    parser.add_argument(
        "--use-attn-every-k-layers",
        type=int,
        default=-1,
    )
    parser.add_argument("--not-use-gated-mlp", action="store_true")
    parser.add_argument("--use-even-classes", action="store_true")
    parser.add_argument("--image-ext", type=str, default="jpg")
    parser.add_argument("--ada-cfg", action="store_true", help="flag for adaptive cfg")

    parser.add_argument(
        "--bimamba-type", type=str, default="v2", choices=["v2", "none", "zigma_8", "sweep_8", "jpeg_8", "sweep_4"]
    )
    parser.add_argument("--pe-type", type=str, default="ape", choices=["ape", "cpe", "rope"])
    parser.add_argument(
        "--block-type",
        type=str,
        default="linear",
        choices=["linear", "raw", "wave", "combined", "window", "combined_fourier", "combined_einfft"],
    )
    parser.add_argument("--cond-mamba", action="store_true")
    parser.add_argument("--scanning-continuity", action="store_true")
    parser.add_argument("--enable-fourier-layers", action="store_true")
    parser.add_argument("--rms-norm", action="store_true")
    parser.add_argument("--fused-add-norm", action="store_true")
    parser.add_argument("--drop-path", type=float, default=0.0)
    parser.add_argument("--learnable-pe", action="store_true")

    parser.add_argument("--eval-refdir", type=str, default=None)
    parser.add_argument(
        "--eval-metric",
        type=str,
        default="fid50k_full",
        help="Metrics to compute, separated by comma (e.g. fid50k_full, pr50k3_full)"
    )

    # groups for better aesthetics for help
    group = parser.add_argument_group("MoE arguments")
    group.add_argument("--num-moe-experts", type=int, default=8)
    group.add_argument("--mamba-moe-layers", type=none_or_str, nargs="*", default=None) # multiple string arguments for this flag, stored as string or none
    group.add_argument("--is-moe", action="store_true")
    # TODO: integrate support for ecmoe in single GPU sampling script and training script
    group.add_argument(
        "--routing-mode", type=str, choices=["sinkhorn", "top1", "top2", "sinkhorn_top2", "ecmoe"], default="top1"
    )
    group.add_argument("--gated-linear-unit", action="store_true")

    group = parser.add_argument_group("Flow Matching arguments")
    group.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument("--loss-weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
    group.add_argument("--sample-eps", type=float)
    group.add_argument("--train-eps", type=float)

    if mode == "ODE":
        group = parser.add_argument_group("ODE Arguments")
        group.add_argument(
            "--sampling-method",
            type=str,
            default="dopri5",
            help="blackbox ODE solver methods; full list check https://github.com/rtqichen/torchdiffeq",
        )
        group.add_argument("--atol", type=float, default=1e-6, help="absolute tolerance")
        group.add_argument("--rtol", type=float, default=1e-3, help="relative tolenrance")
        group.add_argument("--reverse", action="store_true")
        group.add_argument("--likelihood", action="store_true")
    elif mode == "SDE":
        group = parser.add_argument_group("SDE arguments")
        group.add_argument("--sampling-method", type=str, default="Euler", choices=["Euler", "Heun"])
        group.add_argument(
            "--diffusion-form",
            type=str,
            default="none",
            choices=["none", "constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing", "log"],
            help="form of diffusion coefficient in the SDE",
        )
        group.add_argument("--diffusion-norm", type=float, default=1.0)
        group.add_argument(
            "--last-step",
            type=none_or_str,
            default="Mean",
            choices=[None, "Mean", "Tweedie", "Euler"],
            help="Form of the last step in  simulation",
        )
        group.add_argument("--last-step-size", type=float, default=-1, help="size of the last step in SDE simulation")
    
    args = parser.parse_args()
    main(mode, args)

    
