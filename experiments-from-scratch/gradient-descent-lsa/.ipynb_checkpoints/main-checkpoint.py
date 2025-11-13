from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: Apple Silicon")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Device: GPU")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    return device

def generate_batch(batch_size, n, curr_d, max_d, device, s = None):
    xs = torch.zeros(batch_size, n, max_d, device = device)
    xs[:, :, :curr_d] = torch.randn(batch_size, n, curr_d, device = device)

    ws = torch.zeros(batch_size, max_d, device = device)

    if s is None or s >= curr_d:
        ws[:, :curr_d] = torch.randn(batch_size, curr_d, device = device)
    else:
        for b in range(batch_size):
            idx = torch.randperm(curr_d, device = device)[:s]
            ws[b, idx] = torch.randn(s, device = device)

    ys = torch.einsum("bnd,bd->bn", xs, ws)

    return xs, ys

class Curriculum:
    def __init__(self, dims_start: int, dims_end: int, dims_inc: int, points_start: int, points_end: int, points_inc: int, interval: int):
        self.n_dims = dims_start
        self.dims_end = dims_end
        self.dims_inc = dims_inc

        self.n_points = points_start
        self.points_end = points_end
        self.points_inc = points_inc

        self.interval = interval
        self.step_count = 0

    def update(self):
        self.step_count += 1

        if self.step_count % self.interval == 0:
            self.n_dims = min(self.n_dims + self.dims_inc, self.dims_end)
            self.n_points = min(self.n_points + self.points_inc, self.points_end)

class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
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
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis = 2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim = 2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds = embeds).last_hidden_state
        prediction = self._read_out(output)
        return prediction[:, ::2, 0][:, inds]  # predict only on xs

device = get_device()

# hyperparams
max_d = 20          # ambient dimension
max_points = 41     # max number of points in curriculum
batch_size = 64
num_steps = 500_000
lr = 1e-4
sparsity_s = None   # or e.g. 5 for sparse w

# curriculum: example similar to Garg-style
curriculum = Curriculum(
    dims_start = 5,
    dims_end = max_d,
    dims_inc = 1,
    points_start = 11,
    points_end = max_points,
    points_inc = 2,
    interval = 2000,
)

model = TransformerModel(
    n_dims = max_d,
    n_positions = max_points,
    n_embd = 256,
    n_layer = 12,
    n_head = 8,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = lr)
loss_fn = F.mse_loss

model.train()

pbar = tqdm(range(num_steps), desc="training", ncols=100)

for step in pbar:
    n = curriculum.n_points
    curr_d = curriculum.n_dims

    xs, ys = generate_batch(
        batch_size = batch_size,
        n = n,
        curr_d = curr_d,
        max_d = max_d,
        device = device,
        s = sparsity_s,
    )

    # zero out y at the query position for input (optional but explicit)
    ys_in = ys.clone()
    ys_in[:, -1] = 0.0

    # last index is the query point
    inds = [n - 1]

    # forward
    optimizer.zero_grad()
    y_pred = model(xs, ys_in, inds = inds).squeeze(-1)  # (B,)
    y_target = ys[:, -1]                                # (B,)

    loss = loss_fn(y_pred, y_target)
    loss.backward()
    optimizer.step()

    curriculum.update()

    if step % 100 == 0:
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "n": n,
            "d": curr_d
        })

save_path = "standard_linear_regression.pt"
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "curriculum_n_dims": curriculum.n_dims,
        "curriculum_n_points": curriculum.n_points,
    },
    save_path,
)
print(f"Saved checkpoint to {save_path}")