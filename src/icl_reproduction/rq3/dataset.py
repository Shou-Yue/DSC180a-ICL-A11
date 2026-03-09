"""
BinaryClassificationDataset wrapper for RQ3 experiments.
Wraps the data_gen() function from data.py for use in commercial LLM testing.
"""

from __future__ import annotations
import sys
import os
from typing import Tuple, Dict, Any

# Add parent directory to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_src = os.path.dirname(_script_dir)
sys.path.insert(0, _src)

# Lazy imports to handle torch
torch = None

def _ensure_torch():
    global torch
    if torch is None:
        import torch as _torch
        torch = _torch
    return torch

def _ensure_data_gen():
    try:
        from data import data_gen as _data_gen
        return _data_gen
    except ImportError:
        # Try alternate import paths
        try:
            from icl_reproduction.data import data_gen as _data_gen
            return _data_gen
        except ImportError:
            raise ImportError("Could not import data_gen from data.py")


class BinaryClassificationDataset:
    """
    Dataset for generating binary classification tasks for in-context learning.
    
    Each task consists of:
    - context_x: (N, d) tensor of context features
    - context_y: (N,) tensor of context labels
    - query_x: (d,) tensor of query features
    - query_y: scalar query label
    
    Args:
        d (int): Dimension of each feature vector
        N (int): Number of context examples (in-context learning length)
        num_tasks (int): Number of tasks to generate
        R (float): Signal strength parameter
        flip_prob (float): Label flip probability for noisy variants
        seed (int): Random seed for reproducibility
        device (str): Device to place tensors on ('cpu' or 'cuda')
    """
    
    def __init__(
        self,
        d: int,
        N: int,
        num_tasks: int = 1,
        R: float = 6.45,
        flip_prob: float = 0.0,
        seed: int = None,
        device: str = "cpu"
    ):
        _ensure_torch()
        data_gen = _ensure_data_gen()
        
        self.d = d
        self.N = N
        self.num_tasks = num_tasks
        self.R = R
        self.flip_prob = flip_prob
        self.seed = seed
        self.device = device
        
        # Generate all tasks
        # data_gen returns (context_x, context_y, query_x, query_y) where:
        # context_x: (B, N, d), context_y: (B, N), query_x: (B, d), query_y: (B,)
        context_x, context_y, query_x, query_y = data_gen(
            d=d,
            N=N,
            B=num_tasks,
            R=R,
            flip_prob=flip_prob,
            device=device,
            seed=seed
        )
        
        self.context_x = context_x
        self.context_y = context_y
        self.query_x = query_x
        self.query_y = query_y
    
    def __len__(self) -> int:
        """Number of tasks in the dataset"""
        return self.num_tasks
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single task.
        
        Returns:
            Dict with keys:
            - 'context_x': (N, d) tensor
            - 'context_y': (N,) tensor
            - 'query_x': (d,) tensor
            - 'query_y': scalar tensor
        """
        return {
            'context_x': self.context_x[idx],
            'context_y': self.context_y[idx],
            'query_x': self.query_x[idx],
            'query_y': self.query_y[idx]
        }
    
    def format_for_prompt(self, task_idx: int = 0) -> str:
        """
        Format a task as a human-readable string for LLM prompts.
        
        Args:
            task_idx (int): Index of task to format
            
        Returns:
            Formatted string with labeled context examples and unlabeled query point
        """
        _ensure_torch()  # Ensure torch is available
        
        context_x = self.context_x[task_idx]
        context_y = self.context_y[task_idx]
        query_x = self.query_x[task_idx]
        
        # Format context examples
        dataset_str = "Labeled Context Examples:\n"
        for i in range(self.N):
            features = [f"{context_x[i, j].item():.4f}" for j in range(self.d)]
            label = int(context_y[i].item())
            dataset_str += f"Example {i+1}: Features = [{', '.join(features)}], Label = {label}\n"
        
        # Format query point
        dataset_str += "\nQuery Point (unlabeled):\n"
        query_features = [f"{query_x[j].item():.4f}" for j in range(self.d)]
        dataset_str += f"Features = [{', '.join(query_features)}]\n"
        
        return dataset_str
    
    def get_task_dict(self, task_idx: int = 0) -> Dict[str, Any]:
        """Get a task as a dictionary of tensors and formatted string."""
        task = self[task_idx]
        task['formatted_prompt'] = self.format_for_prompt(task_idx)
        return task
