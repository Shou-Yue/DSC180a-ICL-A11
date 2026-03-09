"""
TinyLlama probing on linear regression tasks.

Evaluate TinyLlama-1.1B on linear regression in-context learning:
- Compare model predictions to GD-1 (one-step gradient descent from zero)
- Compare to GD++ (preconditioned gradient descent variant)
- Test on-distribution (linear tasks) and OOD (sine-wave tasks)
"""

import torch
import numpy as np
import sys
import os
from typing import List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_src = os.path.dirname(_script_dir)
sys.path.insert(0, _src)


def create_linear_regression_prompt(
    context_x: torch.Tensor,
    context_y: torch.Tensor,
    query_x: torch.Tensor,
    include_answer: bool = False,
    answer: Optional[float] = None
) -> str:
    """
    Create a prompt for linear regression task.
    
    Format: in-context examples followed by query point
    
    Args:
        context_x: (N, d) context features
        context_y: (N,) context labels
        query_x: (d,) query features
        include_answer: Whether to include the correct answer
        answer: The correct answer (optional)
        
    Returns:
        Formatted prompt string
    """
    N, d = context_x.shape
    
    prompt = "Linear Regression Task\n"
    prompt += "Given in-context examples, predict the output for the query point.\n\n"
    
    # Context examples
    prompt += "Context Examples:\n"
    for i in range(N):
        x_str = ", ".join([f"{x:.3f}" for x in context_x[i].cpu().numpy()])
        y_val = context_y[i].item()
        prompt += f"Input: [{x_str}], Output: {y_val:.3f}\n"
    
    # Query
    prompt += "\nQuery:\n"
    query_str = ", ".join([f"{x:.3f}" for x in query_x.cpu().numpy()])
    prompt += f"Input: [{query_str}], Output: ?"
    
    if include_answer and answer is not None:
        prompt += f"\nAnswer: {answer:.3f}"
    
    return prompt


def compute_gd_baseline(
    context_x: torch.Tensor,
    context_y: torch.Tensor,
    query_x: torch.Tensor,
    num_steps: int = 1,
    learning_rate: float = 0.01,
    preconditioned: bool = False
) -> float:
    """
    Compute gradient descent solution as baseline.
    
    Args:
        context_x: (N, d) context features
        context_y: (N,) context labels/outputs
        query_x: (d,) query features
        num_steps: Number of GD steps
        learning_rate: Learning rate
        preconditioned: Whether to use preconditioning (GD++)
        
    Returns:
        Predicted output on query_x
    """
    context_x = context_x.float()
    context_y = context_y.float().unsqueeze(1)
    query_x = query_x.float()
    
    # Initialize weights to zero
    w = torch.zeros(context_x.shape[1], 1, device=context_x.device)
    
    for step in range(num_steps):
        # Compute predictions and loss
        preds = context_x @ w  # (N, 1)
        residuals = preds - context_y  # (N, 1)
        
        # Gradient
        grad = context_x.T @ residuals / context_x.shape[0]  # (d, 1)
        
        # Update with optional preconditioning
        if preconditioned:
            # Use Hessian-based preconditioning (pseudo-inverse)
            hessian = (context_x.T @ context_x) / context_x.shape[0]
            try:
                hessian_inv = torch.linalg.inv(hessian)
                w = w - learning_rate * (hessian_inv @ grad)
            except:
                # Fall back to standard GD if inversion fails
                w = w - learning_rate * grad
        else:
            # Standard GD
            w = w - learning_rate * grad
    
    # Predict on query
    pred = (query_x @ w).item()
    return pred


def eval_linear_regression(
    d: int,
    N: int,
    num_tasks: int = 10,
    lower_bound: float = -1.0,
    upper_bound: float = 1.0,
    seed: int = 42,
    device: str = "cpu",
    num_gd_steps: int = 1,
    learning_rate: float = 0.01
) -> List[Dict]:
    """
    Evaluate TinyLlama on linear regression tasks.
    
    Args:
        d: Input dimension
        N: Number of context examples
        num_tasks: Number of tasks to generate
        lower_bound: Lower bound for input sampling
        upper_bound: Upper bound for input sampling
        seed: Random seed
        device: Device ('cpu' or 'cuda')
        num_gd_steps: Number of gradient descent steps for baselines
        learning_rate: Learning rate for baselines
        
    Returns:
        List of result dicts with keys:
        - task_idx, mse_model, mse_gd1, mse_gdpp
    """
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"Loading TinyLlama-1.1B...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_name = "TinyLlama/TinyLlama-1.1B-Chat"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            device_map=device,
            trust_remote_code=True
        )
        model.eval()
    except Exception as e:
        print(f"❌ Failed to load TinyLlama: {str(e)}")
        print("   Make sure to install: pip install transformers")
        return []
    
    results = []
    
    for task_idx in range(num_tasks):
        try:
            # Generate random linear task: y = W*x + noise
            # W is (1, d), x is (N+1, d)
            w_true = torch.randn(1, d)
            context_x = torch.uniform_(
                torch.zeros(N, d),
                lower=lower_bound,
                upper=upper_bound
            )
            query_x = torch.uniform_(
                torch.zeros(d),
                lower=lower_bound,
                upper=upper_bound
            )
            
            # Generate outputs with small noise
            context_y = (context_x @ w_true.T).squeeze() + 0.01 * torch.randn(N)
            query_y_true = (query_x @ w_true.T).squeeze()
            
            # Create prompt
            prompt = create_linear_regression_prompt(
                context_x, context_y, query_x,
                include_answer=False
            )
            
            # Get TinyLlama's prediction
            try:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=20,
                        temperature=0.0,
                        top_p=1.0,
                        do_sample=False
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Try to extract numerical prediction (look for pattern like "0.123" or "Answer: 0.123")
                response_text = response.split("Output: ?")[-1].strip()
                
                # Extract first float number from response
                import re
                matches = re.findall(r'-?\d+\.?\d*', response_text)
                if matches:
                    pred_model = float(matches[0])
                else:
                    pred_model = 0.0
                
            except Exception as e:
                print(f"  ⚠️  Error in inference task {task_idx}: {str(e)}")
                pred_model = 0.0
            
            # Compute baselines
            pred_gd1 = compute_gd_baseline(
                context_x, context_y, query_x,
                num_steps=1, learning_rate=learning_rate,
                preconditioned=False
            )
            
            pred_gdpp = compute_gd_baseline(
                context_x, context_y, query_x,
                num_steps=1, learning_rate=learning_rate,
                preconditioned=True
            )
            
            # Compute MSE
            mse_model = float((pred_model - query_y_true.item()) ** 2)
            mse_gd1 = float((pred_gd1 - query_y_true.item()) ** 2)
            mse_gdpp = float((pred_gdpp - query_y_true.item()) ** 2)
            
            results.append({
                'task_idx': task_idx,
                'mse_model': mse_model,
                'mse_gd1': mse_gd1,
                'mse_gdpp': mse_gdpp,
                'pred_model': pred_model,
                'pred_gd1': pred_gd1,
                'pred_gdpp': pred_gdpp,
                'true_output': query_y_true.item()
            })
            
            if (task_idx + 1) % max(1, num_tasks // 5) == 0:
                print(f"  Task {task_idx + 1}/{num_tasks} - Model MSE: {mse_model:.4f}, GD-1 MSE: {mse_gd1:.4f}")
        
        except Exception as e:
            print(f"  ❌ Task {task_idx} failed: {str(e)}")
            results.append({
                'task_idx': task_idx,
                'mse_model': float('nan'),
                'mse_gd1': float('nan'),
                'mse_gdpp': float('nan'),
                'error': str(e)
            })
    
    return results


def eval_ood_sine_tasks(
    d: int,
    N: int,
    num_tasks: int = 5,
    frequency: float = 1.0,
    amplitude: float = 1.0,
    lower_bound: float = -np.pi,
    upper_bound: float = np.pi,
    seed: int = 42,
    device: str = "cpu"
) -> List[Dict]:
    """
    Evaluate TinyLlama on out-of-distribution sine-wave tasks.
    
    Models trained on linear functions may not generalize to non-linear functions.
    
    Args:
        d: Input dimension (only first dimension is used for sine)
        N: Number of context examples
        num_tasks: Number of OOD tasks
        frequency: Sine wave frequency
        amplitude: Sine wave amplitude
        lower_bound: Lower bound for sampling
        upper_bound: Upper bound for sampling
        seed: Random seed
        device: Device
        
    Returns:
        List of result dicts
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"Loading TinyLlama for OOD evaluation...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_name = "TinyLlama/TinyLlama-1.1B-Chat"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            device_map=device,
            trust_remote_code=True
        )
        model.eval()
    except Exception as e:
        print(f"❌ Failed to load TinyLlama: {str(e)}")
        return []
    
    results = []
    
    for task_idx in range(num_tasks):
        try:
            # Generate sine task: y = A * sin(f * x[0])
            context_x = torch.linspace(lower_bound, upper_bound, N).unsqueeze(-1)
            # Pad with zeros for other dimensions
            if d > 1:
                context_x = torch.cat([
                    context_x,
                    torch.zeros(N, d - 1)
                ], dim=1)
            
            context_y = amplitude * torch.sin(frequency * context_x[:, 0])
            
            query_x = torch.tensor([
                np.random.uniform(lower_bound, upper_bound)
            ] + [0.0] * (d - 1), dtype=torch.float32)
            
            query_y_true = amplitude * torch.sin(frequency * query_x[0])
            
            # Create prompt
            prompt = create_linear_regression_prompt(
                context_x, context_y, query_x,
                include_answer=False
            )
            
            # Get prediction
            try:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=20,
                        temperature=0.0
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                import re
                response_text = response.split("Output: ?")[-1].strip()
                matches = re.findall(r'-?\d+\.?\d*', response_text)
                if matches:
                    pred = float(matches[0])
                else:
                    pred = 0.0
                
            except Exception as e:
                pred = 0.0
            
            mse = float((pred - query_y_true.item()) ** 2)
            
            results.append({
                'task_idx': task_idx,
                'mse': mse,
                'prediction': pred,
                'true_output': query_y_true.item(),
                'ood_type': 'sine'
            })
            
        except Exception as e:
            results.append({
                'task_idx': task_idx,
                'mse': float('nan'),
                'error': str(e),
                'ood_type': 'sine'
            })
    
    return results
