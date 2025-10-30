import os
from pathlib import Path
from random import randint
import uuid

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml

from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model

import wandb


def get_device():
    """
    Prefer MPS on Apple Silicon, then CUDA, else CPU.
    Handle older PyTorch builds that may not expose torch.backends.mps.
    """
    if hasattr(torch.backends, "mps"):
        try:
            if torch.backends.mps.is_built() and torch.backends.mps.is_available():
                return torch.device("mps")
        except Exception:
            pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Enable backend-specific optimizations only when relevant
_device = get_device()
if _device.type == "cuda" and hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.benchmark = True


def to_device(obj, device):
    """
    Move tensors (or collections of tensors) to the selected device.
    Safely no-ops for non-tensor objects.
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj


def train_step(model, xs, ys, optimizer, loss_func):
    optimizer.zero_grad()
    output = model(xs, ys)
    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args, device):
    optimizer = torch.optim.Adam(model.parameters(), lr = args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)

    # Ensure run directory exists even on resume or if external process deletes it
    out_dir_path = Path(args.out_dir).resolve()
    out_dir_path.mkdir(parents = True, exist_ok = True)

    starting_step = 0
    state_path = out_dir_path / "state.pt"
    if state_path.exists():
        # Ensure checkpoints saved on another device can load on MPS/CPU
        state = torch.load(str(state_path), map_location = device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for _ in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = model.n_dims
    bsize = args.training.batch_size
    data_sampler = get_data_sampler(args.training.data, n_dims = n_dims)
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        num_tasks = args.training.num_tasks,
        **args.training.task_kwargs,
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples

    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}

        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )
        task = task_sampler(**task_sampler_args)
        ys = task.evaluate(xs)

        # Move inputs and labels to the selected device
        xs = to_device(xs, device)
        ys = to_device(ys, device)

        loss_func = task.get_training_metric()
        loss, output = train_step(model, xs, ys, optimizer, loss_func)

        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys).mean(dim = 0)

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "overall_loss": loss,
                    "excess_loss": loss / baseline_loss,
                    "pointwise/loss": dict(
                        zip(point_wise_tags, point_wise_loss.detach().cpu().numpy())
                    ),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                },
                step = i,
            )

        curriculum.update()

        pbar.set_description(f"loss {loss}")

        # Periodic checkpoint
        if i % args.training.save_every_steps == 0 and not args.test_run:
            state_path.parent.mkdir(parents = True, exist_ok = True)
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, str(state_path))

        # Optional permanent checkpoints
        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            ckpt_path = out_dir_path / f"model_{i}.pt"
            ckpt_path.parent.mkdir(parents = True, exist_ok = True)
            torch.save(model.state_dict(), str(ckpt_path))


def main(args):
    device = _device

    # Re-ensure run dir and config exist before any side-effects
    out_dir_path = Path(args.out_dir).resolve()
    out_dir_path.mkdir(parents = True, exist_ok = True)
    cfg_path = out_dir_path / "config.yaml"
    if not cfg_path.exists():
        with open(cfg_path, "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style = False)

    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.init(
            dir = args.out_dir,  # absolute path set below in __main__
            project = args.wandb.project,
            entity = args.wandb.entity,
            config = args.__dict__,
            notes = args.wandb.notes,
            name = args.wandb.name,
            resume = True,
        )

    model = build_model(args.model)
    model.to(device)
    model.train()

    # Optional: improve matmul throughput on supported backends
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    train(model, args, device)

    if not args.test_run:
        # Ensure config exists before eval; skip gracefully if not
        if not cfg_path.exists():
            with open(cfg_path, "w") as yaml_file:
                yaml.dump(args.__dict__, yaml_file, default_flow_style = False)
        if cfg_path.exists():
            _ = get_run_metrics(args.out_dir)  # precompute metrics for eval
        else:
            print(f"[warn] Skipping eval: missing {cfg_path}")


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema = schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "llama", "lstm"]
    print(f"Running with: {args}")

    if not args.test_run:
        run_id = args.training.resume_id or str(uuid.uuid4())

        # Build an absolute run directory and ensure it exists
        base_out = Path(args.out_dir).resolve()
        out_dir = (base_out / run_id).resolve()
        out_dir.mkdir(parents = True, exist_ok = True)
        args.out_dir = str(out_dir)

        # Persist resolved config immediately
        with open(out_dir / "config.yaml", "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style = False)

    main(args)
