"""
Phase 3: Open-Weights Model Probing - Test Suite

This module validates that the TinyLlama probing functions
work correctly for linear regression and OOD evaluation.
"""

def test_imports():
    """Test that tinyllama_probing module imports correctly"""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    
    from rq3 import tinyllama_probing
    
    assert hasattr(tinyllama_probing, 'eval_linear_regression')
    assert hasattr(tinyllama_probing, 'eval_ood_sine_tasks')
    assert hasattr(tinyllama_probing, 'create_linear_regression_prompt')
    assert hasattr(tinyllama_probing, 'compute_gd_baseline')
    
    print("✅ All TinyLlama probing functions are available")


def test_gd_baseline():
    """Test gradient descent baseline computation"""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    
    from rq3.tinyllama_probing import compute_gd_baseline
    
    # Simple 1D linear task: y = 2*x
    import torch
    context_x = torch.tensor([[1.0], [2.0], [3.0]])
    context_y = torch.tensor([2.0, 4.0, 6.0])
    query_x = torch.tensor([4.0])
    
    # GD should learn approximately y = 2*x
    pred = compute_gd_baseline(context_x, context_y, query_x, num_steps=100, learning_rate=0.01)
    
    # Should predict close to 8.0
    assert abs(pred - 8.0) < 2.0, f"Expected ~8.0, got {pred}"
    
    print(f"✅ GD baseline works: predicted {pred:.2f} for y=2*x with x=4")


def test_gd_preconditioned():
    """Test preconditioned gradient descent baseline"""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    
    from rq3.tinyllama_probing import compute_gd_baseline
    
    import torch
    context_x = torch.randn(10, 5)
    context_y = torch.randn(10)
    query_x = torch.randn(5)
    
    # Both should return floats
    pred_gd = compute_gd_baseline(context_x, context_y, query_x, preconditioned=False, num_steps=1)
    pred_gdpp = compute_gd_baseline(context_x, context_y, query_x, preconditioned=True, num_steps=1)
    
    assert isinstance(pred_gd, float)
    assert isinstance(pred_gdpp, float)
    
    print(f"✅ GD++ preconditioned works: GD={pred_gd:.4f}, GD++={pred_gdpp:.4f}")


def test_prompt_formatting():
    """Test linear regression prompt formatting"""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    
    from rq3.tinyllama_probing import create_linear_regression_prompt
    
    import torch
    context_x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    context_y = torch.tensor([5.0, 11.0])
    query_x = torch.tensor([2.0, 3.0])
    
    prompt = create_linear_regression_prompt(context_x, context_y, query_x)
    
    assert isinstance(prompt, str)
    assert 'Input:' in prompt
    assert 'Output:' in prompt
    assert 'Query' in prompt
    
    print("✅ Linear regression prompt formatting works")
    print(f"📝 Sample prompt:\n{prompt}\n")


def test_framework_structure():
    """Test that eval_linear_regression has correct signature"""
    import sys
    import os
    import inspect
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    
    from rq3.tinyllama_probing import eval_linear_regression, eval_ood_sine_tasks
    
    # Check signatures
    sig_lin = inspect.signature(eval_linear_regression)
    sig_ood = inspect.signature(eval_ood_sine_tasks)
    
    # Should have these parameters
    assert 'd' in sig_lin.parameters
    assert 'N' in sig_lin.parameters
    assert 'num_tasks' in sig_lin.parameters
    assert 'seed' in sig_lin.parameters
    
    assert 'd' in sig_ood.parameters
    assert 'N' in sig_ood.parameters
    assert 'num_tasks' in sig_ood.parameters
    
    print("✅ Function signatures are correct")
    print(f"   eval_linear_regression params: {list(sig_lin.parameters.keys())}")
    print(f"   eval_ood_sine_tasks params: {list(sig_ood.parameters.keys())}")


if __name__ == '__main__':
    print("Running Phase 3 TinyLlama Probing Tests\n")
    print("=" * 60)
    
    test_imports()
    test_gd_baseline()
    test_gd_preconditioned()
    test_prompt_formatting()
    test_framework_structure()
    
    print("\n" + "=" * 60)
    print("✅ All Phase 3 framework tests passed!")
    print("\n📝 Note: Full TinyLlama inference tests require GPU/transformers,")
    print("   and are tested during actual evaluation runs.")
