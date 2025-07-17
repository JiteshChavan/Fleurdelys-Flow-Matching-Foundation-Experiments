# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.

import os

import torch
import torch.nn as nn

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import argparse
import sys
from time import time

import numpy as np
import torchvision

from create_model import create_model
from diffusers.models import AutoencoderKL
from download import find_model

from PIL import Image
from tqdm import tqdm
from flowMatching import Sampler, create_flow

class NFECount(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("nfe", torch.tensor(0.0))
    
    def forward (self, x, t, *args, **kwargs):
        self.nfe += 1.0
        return self.model(x, t, *args, **kwargs)
    
    def forward_with_cfg(self, x, t, *args, **kwargs):
        self.nfe += 1.0
        return self.model.forward_with_cfg(x, t, *args, **kwargs)
    
    def forward_with_adacfg(self, x, t, *args, **kwargs):
        self.nfe += 1.0
        return self.model.forward_with_adacfg(x, t, *args, **kwargs)
    
    def reset_nfe(self):
        self.nfe = torch.tensor(0.0)
    
def main(mode, args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = 'cuda' if torch.cuda.is_available() else "cpu"

    # Load Model:
    latent_size = args.image_size // 8
    model = create_model(args).to(device)
    ckpt_path = args.ckpt
    # Auto download a pretrained model or load a custom checkpoint from train.py
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()

    if args.compute_nfe:
        # model.count_nfe = True
        model = NFECount(model).to(device) # count wrapper

    flow = create_flow(args.path_type, args.prediction, args.loss_weight, args.train_eps, args.sample_eps)
    sampler = Sampler(flow)
    if mode == "ODE":
        if args.likelihood:
            assert args.cfg_scale == 1, "Likelihood evaluation is incompatible with guidance"
            sample_fn = sampler.sample_ode_likelihood(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
            )
        else:
            sample_fn = sampler.sample_ode(
                sampling_method=args.sampling_method, # will default to dopri15 either way
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
            # last_step_size 1 / num_steps by default
            last_step_size=args.last_step_size,
            num_steps=args.num_sampling_steps,
        )
    
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model:
    use_label = True if args.num_classes > 1 else False
    if use_label:
        real_num_classes = args.num_classes - 1 # don't count the uncond cls
    else:
        real_num_classes = args.num_classes
    
    use_cfg = args.cfg_scale > 1.0
    # better to set this up in args
    # global batch size has to be > 16 for this setup to work
    class_labels = [207, 360, 387, 974, 88, 393, 979, 417, 279, 972, 973, 980, 270, 33, 344, 345] * (
        args.global_batch_size // 16 # TODO: move this thing to args
    )  # 355
    n = len(class_labels) if use_label else args.global_batch_size

    # Sample from Pinit
    x0 = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = None if not use_label else torch.tensor(class_labels, device=device)

    # setup classifier free guidance
    if use_cfg:
        x0 = torch.cat([x0,x0], dim=0)
        y_null = torch.tensor([real_num_classes] * n, device=device)
        y = torch.cat([y, y_null], dim=0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
        model_fn = model.forward_with_cfg if not args.ada_cfg else model.forward_with_adacfg
    else:
        model_kwargs = dict(y=y)
        model_fn = model.forward
    
    if args.compute_nfe:
        print("Compute nfe")
        average_nfe = 0.0
        num_trials = 30
        for i in tqdm(range(num_trials)):
            x0 = torch.randn(n, 4, latent_size, latent_size, device=device)
            y = None if not use_label else torch.tensor(class_labels, device=device)
            if use_cfg:
                x0 = torch.cat((x0, x0), dim=0)
                y_null = torch.tensor([real_num_classes] * n, device=device)
                y = torch.cat((y, y_null), dim=0)
                model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
                model_fn = model.forward_with_cfg
            else:
                model_kwargs = dict(y=y)
                model_fn = model.forward
            # sample_fn is alrady initiated with num_steps, t0, t1
            _ = sample_fn(x0, model_fn, **model_kwargs)[-1] # extract only the last point (in case of num steps = 4 it will be the outcome evaluated at t=0.75 with dt=0.25)
            average_nfe += model.nfe / num_trials
            model.reset_nfe()
        print(f"Average NFE over {num_trials} trials: {int(average_nfe)}")
        exit(0)
    
    if args.measure_time:
        print("Measuring time")
        # INIT LOGGERS
        repetitions = 30
        timings = np.zeros((repetitions, 1))
        # GPU-WARM-UP
        for _ in range(10):
            _ = model_fn(x0, torch.ones((n), device=device), **model_kwargs)
        torch.cuda.synchronize()

        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in tqdm(range(repetitions)):
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                starter.record()
                _ = sample_fn(x0, model_fn, **model_kwargs)[-1]
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        print (f"Time for {repetitions} inferences over batch_size:{n} is {mean_syn:.2f}+/-{std_syn:.2f}")
        exit(0)

    # Sample images
    start_time = time()
    samples = sample_fn(x0, model_fn, **model_kwargs)[-1]
    if use_cfg: # remove null samples
        samples, _ = samples.chunk(2, dim=0) # remove null class samples
        del _
    samples = vae.decode(samples / 0.18215).sample
    print(f"Sampling took {time() - start_time} seconds.")

    # Save and disaplay images:
    os.makedirs(args.inference_path, exist_ok=True)
    if use_cfg:
        torchvision.utils.save_image(samples, f"sample_cfg{args.cfg_scale}.png", nrow=8, normalize=True, value_range=(-1,1), pad_value=1.)
    else:
        torchvision.utils.save_image(samples, "sample.png", nrow=8, normalize=True, value_range=(-1,1), pad_value=1.)
        
def none_or_str(value):
    if value == "None":
        return None
    return value

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    if len(sys.argv) < 2:
        print("Usage: program.py <mode> [options]")
        sys.exit(1)
    
    mode = sys.argv[1]
    assert mode[:2] != "--", "Usage: program.py <mode> [options]"
    assert mode in ["ODE", "SDE"], "Invalid mode. Please choose 'ODE' or 'SDE'"

    parser.add_argument("--inference-path", type=str, default=".")
    # irrelevant for our construction
    # parser.add_argument ("--sampler-type", type=str, default="ODE", choices=["ODE", "SDE"])

    parser.add_argument("--model", type=str, default="Fleurdelys_XL_2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Optional path to a pretrained checkpoint (default: auto-download a pre-trained SiT-XL/2 model)."
    )
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--learn-sigma", action="store_true")
    parser.add_argument(
        "--use-attn-every-k-layers",
        type=int,
        default=-1
    )
    parser.add_argument("--not-use-gated-mlp", action="store_true")
    parser.add_argument("--ada-cfg", action="store_true", help="Use adaptive cfg as MDT")

    parser.add_argument(
        "--bimamba-type", type=str, default="v2", choices=["v2", "none", "zigma_8", "sweep_8", "jpeg_8", "sweep_4"]
    )
    parser.add_argument("--pe-type", type=str, default="ape", choices=["ape", "cpe", "rope"])
    parser.add_argument(
        "--block-type",
        type=str,
        default="liner",
        choices=["linear", "raw", "wave", "combined", "window", "combined_fourier", "combined_einfft"]        
    )

    # important!!!
    parser.add_argument("--cond-mamba", action="store_true")
    parser.add_argument("--scanning-continuity", action="store_true")
    parser.add_argument("--rms-norm", action="store_true")
    parser.add_argument("--fused-add-norm", action="store_true")
    parser.add_argument("--drop-path", type=float, default=0.0)
    parser.add_argument("--learnable-pe", action="store_true")
    parser.add_argument("--measure-time", action="store_true")
    parser.add_argument("--compute-nfe", action="store_true")

    group = parser.add_argument_group("MoE arguments")
    group.add_argument("--num-moe-experts", type=int, default=8)
    # can pass multiple values to this argument eg: 0 1 2 3 4 which gets converted to list
    group.add_argument("--mamba-moe-layers", type=none_or_str, nargs="*", default=None)
    group.add_argument("--is-moe", action="store_true")
    group.add_argument(
        "--routing-mode", type=str, choices=["sinkhorn", "top1", "top2", "sinkhorn_top2", "ECMoE"], default="top1"
    )
    group.add_argument("--gated-linear-unit", action="store_true")

    group = parser.add_argument_group("FlowMatching arguments")
    group.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument("--loss-weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
    group.add_argument("--sample-eps", type=float)
    group.add_argument("--train-eps", type=float)

    if mode == "ODE":
        group = parser.add_argument_group("ODE arguments")
        group.add_argument(
            "--sampling-method", type=str, default="dopri15", help="blackbox ODE solver methods; for full list check https://github.com/rtqichen/torchdiffeq",
        )
        group.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
        group.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
        group.add_argument("--reverse", action="store_true")
        group.add_argument("--likelihood", action="store_true")
    elif mode == "SDE":
        group = parser.add_argument_group("SDE arguments")
        group.add_argument("--sampling-method", type=str, default="Euler", choices=["Euler", "Heun"])
        group.add_argument(
            "--diffusion-form",
            type=str, default="none",
            choices=["none", "constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing", "log"],
            help="form of diffusion coefficient in the SDE",
        )
        group.add_argument("--diffusion-norm", type=float, default=1.0)
        group.add_argument(
            "--last-step",
            type=none_or_str,
            default="Mean",
            choices=[None, "Mean", "Tweedie", "Euler"],
            help="form of the last step while silumating the SDE",
        )
        group.add_argument("--last-step-size", type=float, default=-1, help="dt for the last step of SDE")

    
    args = parser.parse_args()
    main(mode, args)