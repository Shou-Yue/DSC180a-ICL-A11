import argparse
import os
import sys

import torch

_script_dir = os.path.dirname(os.path.abspath(__file__))
_src = os.path.dirname(_script_dir)
sys.path.insert(0, _src)

from icl_reproduction.experiments.runner import run_one
from icl_reproduction.experiments.plots import generate_all_plots

D_VALS = [50, 100, 200, 500, 1000]
N_VALS = [5, 10, 20, 40, 80]
B_VALS = [50, 100, 250, 500, 1000, 2000]
D_GRID = [100, 500, 1000]
N_GRID = [10, 20, 40]
B_GRID = [100, 500, 2000]
D0, N0, B0 = 500, 20, 1000
R_CONST = 6.45
R_SNR_C = 0.3
MAX_STEPS = 1000
LR = 1e-2
EVAL_EVERY = 50
LOG_EVERY = 10
DEFAULT_FLIP_TRAIN = 0.2
DEFAULT_FLIP_VAL = 0.2


def _runs(seeds):
    seen = set()
    out = []
    for r_mode, r_val in [("const", R_CONST), ("snr", R_SNR_C)]:
        for d in D_VALS:
            key = (d, N0, B0, r_mode, r_val)
            if key not in seen:
                seen.add(key)
                for seed in seeds:
                    out.append((d, N0, B0, r_mode, r_val, seed))
        for n in N_VALS:
            key = (D0, n, B0, r_mode, r_val)
            if key not in seen:
                seen.add(key)
                for seed in seeds:
                    out.append((D0, n, B0, r_mode, r_val, seed))
        for b in B_VALS:
            key = (D0, N0, b, r_mode, r_val)
            if key not in seen:
                seen.add(key)
                for seed in seeds:
                    out.append((D0, N0, b, r_mode, r_val, seed))
    for r_mode, r_val in [("const", R_CONST), ("snr", R_SNR_C)]:
        for d in D_GRID:
            for n in N_GRID:
                key = (d, n, B0, r_mode, r_val)
                if key not in seen:
                    seen.add(key)
                    for seed in seeds:
                        out.append((d, n, B0, r_mode, r_val, seed))
        for b in B_GRID:
            for n in N_GRID:
                key = (D0, n, b, r_mode, r_val)
                if key not in seen:
                    seen.add(key)
                    for seed in seeds:
                        out.append((D0, n, b, r_mode, r_val, seed))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", default="rq1_linear_scaling")
    p.add_argument("--output_root", default="results")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--steps", type=int, default=MAX_STEPS)
    p.add_argument("--noise", action="store_true")
    p.add_argument("--flip_train", type=float, default=DEFAULT_FLIP_TRAIN)
    p.add_argument("--flip_val", type=float, default=DEFAULT_FLIP_VAL)
    p.add_argument("--only_plots", action="store_true")
    args = p.parse_args()
    steps = min(args.steps, MAX_STEPS)
    seeds = [int(x.strip()) for x in args.seeds.split(",")]
    if not args.only_plots:
        for d, n, b, r_mode, r_val, seed in _runs(seeds):
            run_one(d, n, b, r_mode, r_val, seed, args.output_root, args.exp_name, steps, LR, EVAL_EVERY, LOG_EVERY, args.device, flip_train=0.0, flip_val=0.0, subdir="")
        if args.noise:
            for d, n, b, r_mode, r_val, seed in _runs(seeds):
                run_one(d, n, b, r_mode, r_val, seed, args.output_root, args.exp_name, steps, LR, EVAL_EVERY, LOG_EVERY, args.device, flip_train=args.flip_train, flip_val=args.flip_val, subdir="noise")
    generate_all_plots(args.output_root, args.exp_name, D0, N0, B0, D_VALS, N_VALS, B_VALS, None, subdir=None)
    if args.noise:
        generate_all_plots(args.output_root, args.exp_name, D0, N0, B0, D_VALS, N_VALS, B_VALS, None, subdir="noise")
    base = os.path.join(args.output_root, "rq1", args.exp_name)
    print(f"Results: {os.path.abspath(base)}")
    print(f"Plots:   {os.path.abspath(base)}/_plots/")
    if args.noise:
        print(f"Noise:   {os.path.abspath(base)}/noise/")
        print(f"Noise plots: {os.path.abspath(base)}/noise/_plots/")
    if args.only_plots:
        print("(Plots-only run; use without --only_plots to train.)")


if __name__ == "__main__":
    main()
