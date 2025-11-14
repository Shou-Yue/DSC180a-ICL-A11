import os
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import GPT2Config, GPT2Model


#############################################
# Hyperparameters and config
#############################################

OUT_DIR = "models/linear_regression"

# model settings
N_DIMS = 20
N_POSITIONS = 101  # this just has to be > the max points in the curriculum
N_EMBD = 256
N_ATT_LAYERS = 12
N_ATT_HEADS = 8

# training settings
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
TRAIN_STEPS = 500_001
SAVE_EVERY_STEPS = 1_000
KEEP_EVERY_STEPS = 100_000

# curriculum settings
CURR_DIMS_START = 5
CURR_DIMS_END = 20
CURR_DIMS_INC = 1
CURR_POINTS_START = 11
CURR_POINTS_END = 41
CURR_POINTS_INC = 2
CURR_INTERVAL = 2_000  # how often to update dims/points


#############################################
# Utilities
#############################################

def mean_squared_error(pred, target):
    return ((pred - target) ** 2).mean()


class Curriculum:
    """Simple curriculum on (effective dimension, number of points)."""

    def __init__(
        self,
        dims_start,
        dims_end,
        dims_inc,
        points_start,
        points_end,
        points_inc,
        interval,
    ):
        self.dims_trunc = dims_start
        self.points = points_start

        self.dims_start = dims_start
        self.dims_end = dims_end
        self.dims_inc = dims_inc

        self.points_start = points_start
        self.points_end = points_end
        self.points_inc = points_inc

        self.interval = interval
        self.step = 0

    def update(self):
        self.step += 1
        if self.step % self.interval == 0:
            self.dims_trunc = min(self.dims_trunc + self.dims_inc, self.dims_end)
            self.points = min(self.points + self.points_inc, self.points_end)


class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd = 256, n_layer = 12, n_head = 8):
        super().__init__()
        config = GPT2Config(
            n_positions = 2 * n_positions,
            n_embd = n_embd,
            n_layer = n_layer,
            n_head = n_head,
            resid_pdrop = 0.0,
            embd_pdrop = 0.0,
            attn_pdrop = 0.0,
            use_cache = False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims

        self.read_in = nn.Linear(n_dims, n_embd)
        self.backbone = GPT2Model(config)
        self.read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b, ys_b):
        """
        Interleave x's and y's into a single sequence.

        xs_b: [B, T, D]
        ys_b: [B, T]

        Returns: [B, 2T, D] where positions are x_1, y_1, x_2, y_2, ...
        """
        bsize, points, dim = xs_b.shape

        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device = ys_b.device),
            ),
            dim = 2,
        )

        # zs[:, 0] = x_1, zs[:, 1] = y_1, zs[:, 2] = x_2, zs[:, 3] = y_2, ...
        zs = torch.stack((xs_b, ys_b_wide), dim = 2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys):
        """
        xs: [B, T, D]
        ys: [B, T]

        Returns predictions for all y positions: [B, T]
        """
        zs = self._combine(xs, ys)
        embeds = self.read_in(zs)
        output = self.backbone(inputs_embeds = embeds).last_hidden_state
        prediction = self.read_out(output)          # [B, 2T, 1]
        prediction = prediction[:, ::2, 0]          # keep only x positions -> [B, T]
        return prediction


#############################################
# Data generation (Gaussian + linear regression)
#############################################

def sample_gaussian_xs(batch_size, n_points, n_dims, n_dims_trunc, device):
    """
    Sample Gaussian inputs and optionally truncate to a lower-dimensional
    subspace by zeroing out the last coordinates.
    """
    xs = torch.randn(batch_size, n_points, n_dims, device = device)
    if n_dims_trunc is not None and n_dims_trunc < n_dims:
        xs[:, :, n_dims_trunc:] = 0.0
    return xs


def sample_linear_regression_weights(batch_size, n_dims, device):
    """
    Sample one weight vector per batch element, w ~ N(0, I).
    """
    return torch.randn(batch_size, n_dims, 1, device = device)


def evaluate_linear_regression(xs, w, scale = 1.0):
    """
    xs: [B, T, D]
    w: [B, D, 1]

    Returns ys: [B, T]
    """
    ys = scale * (xs @ w)   # [B, T, 1]
    return ys[:, :, 0]


#############################################
# Training
#############################################

def train(model):
    device = next(model.parameters()).device

    os.makedirs(OUT_DIR, exist_ok = True)
    state_path = os.path.join(OUT_DIR, "state.pt")

    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

    curriculum = Curriculum(
        dims_start = CURR_DIMS_START,
        dims_end = CURR_DIMS_END,
        dims_inc = CURR_DIMS_INC,
        points_start = CURR_POINTS_START,
        points_end = CURR_POINTS_END,
        points_inc = CURR_POINTS_INC,
        interval = CURR_INTERVAL,
    )

    # Resume if state exists
    starting_step = 0
    if os.path.exists(state_path):
        state = torch.load(state_path, weights_only = True)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]

        # Fast forward curriculum
        for _ in range(starting_step + 1):
            curriculum.update()

    ema_loss = None
    alpha = 0.01

    pbar = tqdm(range(starting_step, TRAIN_STEPS))
    for step in pbar:
        n_points = curriculum.points
        n_dims_trunc = curriculum.dims_trunc

        # Sample data
        xs = sample_gaussian_xs(
            batch_size = BATCH_SIZE,
            n_points = n_points,
            n_dims = model.n_dims,
            n_dims_trunc = n_dims_trunc,
            device = device,
        )
        w = sample_linear_regression_weights(
            batch_size = BATCH_SIZE,
            n_dims = model.n_dims,
            device = device,
        )
        ys = evaluate_linear_regression(xs, w)  # [B, T]

        # Training step
        model.train()
        optimizer.zero_grad()
        preds = model(xs, ys)
        loss = mean_squared_error(preds, ys)
        loss.backward()
        optimizer.step()

        loss_val = loss.detach().item()
        if ema_loss is None:
            ema_loss = loss_val
        else:
            ema_loss = alpha * loss_val + (1.0 - alpha) * ema_loss

        curriculum.update()

        desc_dict = {
            "loss": round(loss_val, 4),
            "ema_loss": round(ema_loss, 4),
            "d": n_dims_trunc,
            "n": n_points,
        }
        pbar.set_description(str(desc_dict))

        # Save training state (for resume)
        if step % SAVE_EVERY_STEPS == 0:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": step,
            }
            torch.save(training_state, state_path)

        # Save model snapshot every KEEP_EVERY_STEPS
        if step % KEEP_EVERY_STEPS == 0 and step > 0:
            torch.save(
                model.state_dict(),
                os.path.join(OUT_DIR, f"model_{step}.pt"),
            )


#############################################
# Main
#############################################

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel(
        n_dims = N_DIMS,
        n_positions = N_POSITIONS,
        n_embd = N_EMBD,
        n_layer = N_ATT_LAYERS,
        n_head = N_ATT_HEADS,
    )
    model.to(device)
    train(model)
