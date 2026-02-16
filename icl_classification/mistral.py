import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import os
import pandas as pd
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import multiprocessing as mp 
from functools import partial 
import filelock
from transformers import MistralModel
from icl_classification.datasets import GaussianMixtureDataset


@dataclass
class ExperimentConfig:
    """Configuration class for experiment parameters"""
    d: int  # Input dimension
    N: int  # Number of examples per task
    B: int  # Number of tasks
    B_val: int # Number of validation tasks
    R_train: float  # Signal-to-noise ratio during training
    R_val: float  # ...at test time
    max_steps: int  # Maximum training steps
    checkpoint_steps: List[int]  # Steps at which to save checkpoints
    label_flip_p: float=0.0
    learning_rate: float = 1e-2
    use_cuda: bool = True  # Flag for using CUDA
    use_wandb: bool = False # Disable wandb by default
    wandb_project: Optional[str] = "linear-transformer"
    save_checkpoints: bool = False
    save_results: bool = False
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
    experiment_name: Optional[str] = None

    def __post_init__(self):
        """Setup device based on CUDA availability"""
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.use_cuda else "cpu")

        # Generate default experiment name if none provided
        if self.experiment_name is None and self.save_results:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"d{self.d}_N{self.N}_B{self.B}_R{self.R}_{timestamp}"


class MistralTransformer(nn.Module):
    """
    Mistral version of the LinearTransformer.
    - Operates on vector tokens (context_x, target_x)
    - Preserves your original API (_predict_single, forward, compute_in_context_preds)
    """
    def __init__(self, d: int, model_name="mistralai/Mistral-7B-Instruct-v0.3"):
        super().__init__()
        self.d = d

        # Load pretrained Mistral
        self.mistral = MistralModel.from_pretrained(model_name)

        H = self.mistral.config.hidden_size

        # Project your d-dimensional vectors -> hidden size
        self.input_proj = nn.Linear(d, H)

        # Binary classification head
        self.head = nn.Linear(H, 1)

    def _mistral_encode(self, tokens):
        x = self.input_proj(tokens)  # (B, T, H)
        out = self.mistral(inputs_embeds=x).last_hidden_state
        return out

    # ----------------------------
    # Matches original API
    # ----------------------------
    def _predict_single(self, context_x, context_y, target_x):
        B, N, d = context_x.shape

        # Convert labels {0,1} -> {-1,+1}
        y_signal = 2 * context_y.float() - 1
        y_signal = y_signal.unsqueeze(-1)

        # Form (x_i, y_i) tokens
        context_tokens = torch.cat([context_x, y_signal], dim=-1)
        if context_tokens.shape[-1] != d:
            context_tokens = context_tokens[..., :d]

        # Target token (placeholder)
        target_token = target_x.unsqueeze(1)
        tokens = torch.cat([context_tokens, target_token], dim=1)

        # Encode sequence
        h = self._mistral_encode(tokens)
        final = h[:, -1, :]
        logits = self.head(final).squeeze(-1)
        return logits

    def forward(self, context_x, context_y, target_x):
        return self._predict_single(context_x, context_y, target_x)

    def compute_in_context_preds(self, context_x, context_y):
        context_x = context_x / torch.norm(context_x, dim=2, keepdim=True)
        y_signal = 2 * context_y.float() - 1
        y_signal = y_signal.unsqueeze(-1)

        tokens = torch.cat([context_x, y_signal], dim=-1)
        if tokens.shape[-1] != self.d:
            tokens = tokens[..., :self.d]

        h = self._mistral_encode(tokens)
        logits = self.head(h).squeeze(-1)
        preds = (logits > 0).float()
        return preds
    

class Trainer:
    """Trainer class for linear transformer"""
    def __init__(self, config: ExperimentConfig):
        self.config = config
        if config.save_checkpoints or config.save_results:
            self.setup_directories()

        # Initialize model and optimizer
        self.model = MistralTransformer(config.d).to(config.device)
        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=config.learning_rate,
            momentum=0.0)

        # Initialize datasets
        self.train_dataset = GaussianMixtureDataset(
            config.d, config.N, config.B, config.R_train, is_validation=False, label_flip_p=0.0 # no noise during pre-training
        )
        self.val_dataset = GaussianMixtureDataset(
            config.d, config.N, config.B, config.R_val, is_validation=True, label_flip_p=config.label_flip_p
        )

        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=None, 
            shuffle=True,
            pin_memory=True if config.device.type == "cuda" else False
        )

        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=None, 
            shuffle=False,
            pin_memory=True if config.device.type == "cuda" else False
        )

        # Initialize logging
        self.setup_wandb()
        self.metrics = {
            'step': [], 
            'train_loss': [], 
            'train_acc': [],
            'in_context_acc': [],
            'val_loss': [],
            'val_acc': [],
            'batch_time': [],
            'samples_per_second': []
        }

        print(f"Using device: {config.device}")


    def evaluate(self) -> Tuple[float, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                context_x, context_y, target_x, target_y = [t.to(self.config.device) for t in batch]

                # Forward pass
                pred = self.model(context_x, context_y, target_x)
                val_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    pred, target_y.float())

                # Compute accuracy
                val_acc = ((pred > 0).float() == target_y).float().mean()

                # Compute in-context training accuracy
                in_context_preds = self.model.compute_in_context_preds(context_x, context_y)
                in_context_acc = (in_context_preds == context_y).float().mean()

                return val_loss.item(), val_acc.item(), in_context_acc.item()

    def setup_directories(self):
        """Create directories for checkpoints and results"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.results_dir, exist_ok=True)

    def setup_wandb(self):
        """Initialize W&B logging"""
        if not self.config.use_wandb or self.config.wandb_project is None:
            return 

        wandb.init(
            project=self.config.wandb_project,
            config={
                'd': self.config.d,
                'N': self.config.N,
                'B': self.config.B,
                'R': self.config.R,
                'max_steps': self.config.max_steps,
                'batch_size': self.config.B,
                'learning_rate': self.config.learning_rate,
                'device': str(self.config.device)
            }
        )

    def save_checkpoint(self, step: int):
        """Save model checkpoint"""
        if not self.config.save_checkpoints:
            return 

        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics,
            'config': self.config
        }
        path = os.path.join(self.config.checkpoint_dir, f'checkpoint_{self.config.experiment_name}_step_{step}.pt')
        print(f'Saving checkpoint at {path}')
        torch.save(checkpoint, path)

    def load_checkpoint(self, step: int):
        """Load model checkpoint"""
        path = os.path.join(self.config.checkpoint_dir, f'checkpoint_step_{step}.pt')
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.metrics = checkpoint['metrics']
        return checkpoint['step']

    def save_metrics(self):
        """Save metrics to CSV"""
        if not self.config.save_results:
            return 

        # Convert metrics dict to DataFrame
        df = pd.DataFrame(self.metrics)

        # Add experiment parameters as columns
        df['experiment_name'] = self.config.experiment_name
        df['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
        df['d'] = self.config.d
        df['N'] = self.config.N
        df['B'] = self.config.B
        df['R_train'] = self.config.R_train

        # Create filename with experiment name
        filename = f"metrics_{self.config.experiment_name}.csv"
        path = os.path.join(self.config.results_dir, filename)


        # Create lock file path
        lock_path = os.path.join(self.config.results_dir, "results.lock")

        # Save individual experiment results (no lock needed as filename is unique)
        df.to_csv(path, index=False)

        # Safely append to master results file using file lock
        with filelock.FileLock(lock_path):
            master_path = os.path.join(self.config.results_dir, "all_results.csv")
            if os.path.exists(master_path):
                # Read existing data to get column order
                existing_df = pd.read_csv(master_path)
                # Ensure columns match
                df = df[existing_df.columns]
                # Append without writing header
                df.to_csv(master_path, mode='a', header=False, index=False)
            else:
                # First time creating the file
                df.to_csv(master_path, index=False)

    def train(self):
        """Training loop with timing measurements"""
        self.model.train()
        step = 0
        total_start_time = time.time()
        num_samples = 0

        print(f"\nStarting training on {self.config.device}")
        print(f"Input dimension (d): {self.config.d}")
        print(f"Training tasks: {self.config.B}")
        print(f"Validation tasks: {self.config.B_val}")
        print(f"Batch size: {self.config.B}")
        print("-" * 50)

        while step < self.config.max_steps:
            for batch in self.train_loader:
                batch_start_time = time.time()
                # Shapes after batching:
                # context_x: (batch_size, N, d)
                # context_y: (batch_size, N)
                # target_x: (batch_size, d)
                # target_y: (batch_size,)
                context_x, context_y, target_x, target_y = [t.to(self.config.device) for t in batch]

                # Forward pass - pred shape: (batch_size,)
                pred = self.model(context_x, context_y, target_x)
                train_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    pred, target_y.float())

                # Backward pass
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

                # Compute train accuracy
                train_acc = ((pred > 0).float() == target_y).float().mean()

                # Compute validation metrics
                val_loss, val_acc, in_context_acc = self.evaluate()

                # Compute timing metrics
                batch_time = time.time() - batch_start_time
                num_samples += len(context_x)
                avg_samples_per_second = num_samples / (time.time() - total_start_time)

                # Log metrics
                self.metrics['step'].append(step)
                self.metrics['train_loss'].append(train_loss.item())
                self.metrics['train_acc'].append(train_acc.item())
                self.metrics['val_loss'].append(val_loss)
                self.metrics['val_acc'].append(val_acc)
                self.metrics['in_context_acc'].append(in_context_acc)
                self.metrics['batch_time'].append(batch_time)
                self.metrics['samples_per_second'].append(avg_samples_per_second)

                if self.config.use_wandb:
                    wandb.log({
                        'train_loss': train_loss.item(),
                        'train_acc': train_acc.item(),
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'in_context_acc_acc': in_context_acc,
                        'batch_time': batch_time,
                        'samples_per_second': avg_samples_per_second,
                        'step': step
                    })

                # Print progress every 100 steps
                if step % 10 == 0:
                    print(f"Step {step}/{self.config.max_steps} | "
                          f"Train Loss: {train_loss.item():.4f} | "
                          f"Train Acc: {train_acc.item():.4f} | "
                          f"Val Loss: {val_loss:.4f} | "
                          f"Val Acc: {val_acc:.4f} | "
                          f"In-context Acc: {in_context_acc:.4f} | "
                          f"Batch time: {batch_time*1000:.2f}ms | "
                          f"Samples/sec: {avg_samples_per_second:.2f}")

                # Save checkpoint if needed
                if step in self.config.checkpoint_steps:
                    self.save_checkpoint(step)

                step += 1
                if step >= self.config.max_steps:
                    break

        # Print final timing statistics
        total_time = time.time() - total_start_time
        print("\nTraining completed!")
        print(f"Total training time: {total_time:.2f} seconds")
        print(f"Final train accuracy: {self.metrics['train_acc'][-1]:.4f}")
        print(f"Final validation accuracy: {self.metrics['val_acc'][-1]:.4f}") 
        print(f"Final in-context accuracy: {self.metrics['in_context_acc'][-1]:.4f}") 
        print(f"Average samples/second: {num_samples/total_time:.2f}")
        print(f"Average batch time: {np.mean(self.metrics['batch_time'])*1000:.2f}ms")

        # Save final metrics and checkpoint
        self.save_metrics()
        self.save_checkpoint(step)
        if self.config.use_wandb:
            wandb.finish()

        return self.metrics['train_acc'][-1], self.metrics['val_acc'][-1], self.metrics['in_context_acc'][-1]


def run_single_experiment(params, base_results_dir: str, use_cuda: bool = False, checkpoint_dir: str = "checkpoints/"):
    """
    Run a single experiment with given hyperparameters.

    Args:
        params: tuple of (d, B, R_train, R_val, steps) hyperparameters
        base_results_dir: base directory for results
    """
    d, B, R_train, R_val, steps, label_flip_p = params
    # (dimension, batch size = tasks, R for training, \tilde R for validation, num steps)

    # Create unique experiment name based on hyperparameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"d{d}_B{B}_R{int(R_val)}_{timestamp}"

    # Configure experiment
    config = ExperimentConfig(
        d=d,
        N=40, # with R_train = 5 sqrt(d), N = \Omega(1) suffices
        B=B,
        B_val=50, # don't really need validation data since we'll be evaluating checkpoints later
        R_train=R_train,
        R_val=R_val,
        max_steps=steps,
        checkpoint_steps=[steps],
        label_flip_p=label_flip_p,
        use_cuda=use_cuda,
        save_checkpoints=True,
        save_results=True,
        checkpoint_dir=checkpoint_dir,
        results_dir=base_results_dir,
        experiment_name=experiment_name
    )

    # Initialize and run trainer
    trainer = Trainer(config)
    train_acc, val_acc, in_context_acc = trainer.train()
    
    return experiment_name, train_acc, val_acc, in_context_acc


def run_parallel_cpu_experiments(num_processes: int = None):
    """
    Run hyperparameter search using multiprocessing (CPU only).

    Args:
        num_processes: Number of processes to use. If None, uses CPU count.
    """
    # Create base results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = f"mistral_results_{timestamp}"
    checkpoint_dir = "mistral_checkpoints"
    os.makedirs(base_results_dir, exist_ok=True)

    # Create master results file with lock
    master_results_path = os.path.join(base_results_dir, "all_results.csv")
    lock_path = os.path.join(base_results_dir, "results.lock")

    # Generate hyperparameter combinations
    param_combinations = []
    steps = 300
    dimensions = [10, 50, 100, 200, 400, 600, 800, 1000, 1250, 1500]
    # os.makedirs(base_results_dir, exist_ok=True)

    # First do d-varying
    for d in dimensions:
        B = d
        R_train = 5 * d**0.5
        param_combinations.append((d, B, R_train, d ** 0.3, steps))

    # Then compute d-fixed, vary B
    d = 1000
    B_list = [int(d**0.1), int(d**0.3), int(d**0.5), int(d**0.7), int(d**0.9)]
    for B in B_list:
        param_combinations.append((d, B, R_train, d**0.3, steps))

    # Set up multiprocessing
    if num_processes is None:
        num_processes = mp.cpu_count() - 2

    print(f"Starting hyperparameter search with {num_processes} processes")
    print(f"Total experiments to run: {len(param_combinations)}")
    print(f"Results will be saved in: {base_results_dir}")

    # Create partial function with fixed base_results_dir
    run_experiment = partial(run_single_experiment, use_cuda=False, base_results_dir=base_results_dir, checkpoint_dir=checkpoint_dir)

    # Run experiments in parallel
    with mp.Pool(processes=num_processes) as pool:
        experiment_names = pool.map(run_experiment, param_combinations)

    # Combine all individual results files into master results file
    with filelock.FileLock(lock_path):
        all_results = []
        for exp_name in experiment_names:
            results_path = os.path.join(base_results_dir, f"metrics_{exp_name}.csv")
            if os.path.exists(results_path):
                df = pd.read_csv(results_path)
                all_results.append(df)

        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
            combined_results.to_csv(master_results_path, index=False)

    print("\nHyperparameter search completed!")
    print(f"Combined results saved to: {master_results_path}")


if __name__ == "__main__":

    USE_CPU = False
    
    if torch.cuda.is_available() and not USE_CPU:
        print('Running sequential GPU experiments')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_results_dir = f"mistral_results_{timestamp}"
        checkpoint_dir = "mistral_checkpoints"
    
        # First get collection of varying-d, B fixed
        d_list = [10, 50, 100, 200, 400, 600, 800, 1000, 1250, 1500]
        # os.makedirs(base_results_dir, exist_ok=True)
    
        for d in d_list:
            B = d
            R_train = 5 * d**0.5
            params = (d, B, R_train, d**0.35, 300)
            run_single_experiment(params=params, base_results_dir=base_results_dir, use_cuda=True, checkpoint_dir=checkpoint_dir)
    
        # Then compute d-fixed, vary B
        d = 1000
        B_list = [int(d**0.1), int(d**0.3), int(d**0.5), int(d**0.7), int(d**0.9)]
        for B in B_list:
            R_train = 5 * d**0.5
            params = (d, B, R_train, d**0.35, 300)
            run_single_experiment(params=params, base_results_dir=base_results_dir, use_cuda=True, checkpoint_dir=checkpoint_dir)
    
    else: 
        print('Running parallel CPU experiments')
        run_parallel_cpu_experiments() 
