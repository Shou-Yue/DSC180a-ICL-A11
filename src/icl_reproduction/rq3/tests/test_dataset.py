"""
Phase 2: Data Generation - Test Suite

This module validates that the BinaryClassificationDataset wrapper
works correctly for RQ3 experiments.
"""

def test_dataset_instantiation():
    """Test that dataset can be created without errors"""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    
    from rq3.dataset import BinaryClassificationDataset
    
    # Create small dataset
    dataset = BinaryClassificationDataset(
        d=10,
        N=5,
        num_tasks=2,
        R=6.45,
        flip_prob=0.0,
        seed=42
    )
    
    assert dataset.d == 10
    assert dataset.N == 5
    assert len(dataset) == 2
    print("✅ Dataset instantiation works")


def test_dataset_shapes():
    """Test that dataset tensors have correct shapes"""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    
    from rq3.dataset import BinaryClassificationDataset
    
    d, N, num_tasks = 20, 10, 5
    dataset = BinaryClassificationDataset(
        d=d,
        N=N,
        num_tasks=num_tasks,
        R=3.0,
        seed=123
    )
    
    assert dataset.context_x.shape == (num_tasks, N, d), f"Expected ({num_tasks}, {N}, {d}), got {dataset.context_x.shape}"
    assert dataset.context_y.shape == (num_tasks, N), f"Expected ({num_tasks}, {N}), got {dataset.context_y.shape}"
    assert dataset.query_x.shape == (num_tasks, d), f"Expected ({num_tasks}, {d}), got {dataset.query_x.shape}"
    assert dataset.query_y.shape == (num_tasks,), f"Expected ({num_tasks},), got {dataset.query_y.shape}"
    
    print("✅ Dataset tensor shapes are correct")


def test_dataset_access():
    """Test that dataset indexing works correctly"""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    
    from rq3.dataset import BinaryClassificationDataset
    
    dataset = BinaryClassificationDataset(
        d=5,
        N=3,
        num_tasks=2,
        R=1.0,
        seed=0
    )
    
    # Test dictionary access
    task = dataset[0]
    assert isinstance(task, dict)
    assert 'context_x' in task
    assert 'context_y' in task
    assert 'query_x' in task
    assert 'query_y' in task
    
    print("✅ Dataset indexing works")


def test_prompt_formatting():
    """Test that prompt formatting returns a valid string"""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    
    from rq3.dataset import BinaryClassificationDataset
    
    dataset = BinaryClassificationDataset(
        d=5,
        N=3,
        num_tasks=1,
        R=1.0,
        seed=0
    )
    
    prompt = dataset.format_for_prompt(task_idx=0)
    
    assert isinstance(prompt, str)
    assert 'Labeled Context Examples' in prompt
    assert 'Query Point (unlabeled)' in prompt
    assert 'Example 1:' in prompt
    assert 'Features' in prompt
    assert 'Label' in prompt
    
    print("✅ Prompt formatting works")
    print(f"\n📝 Sample prompt:\n{prompt}")


def test_flip_probability():
    """Test that label flipping works"""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    
    from rq3.dataset import BinaryClassificationDataset
    
    # Create clean dataset
    clean = BinaryClassificationDataset(
        d=100,
        N=50,
        num_tasks=1,
        R=1.0,
        flip_prob=0.0,
        seed=999
    )
    
    # Create noisy dataset with same seed
    noisy = BinaryClassificationDataset(
        d=100,
        N=50,
        num_tasks=1,
        R=1.0,
        flip_prob=0.3,
        seed=999
    )
    
    # Context labels should be different (some flipped)
    label_diffs = (clean.context_y != noisy.context_y).sum().item()
    
    # Expect roughly 30% flipped (15 ± some variance out of 50)
    assert label_diffs > 5, f"Expected ~15 flips, got {label_diffs}"
    
    print(f"✅ Label flipping works: {label_diffs} labels flipped out of 50")


if __name__ == '__main__':
    print("Running Phase 2 Data Generation Tests\n")
    print("=" * 50)
    
    test_dataset_instantiation()
    test_dataset_shapes()
    test_dataset_access()
    test_prompt_formatting()
    test_flip_probability()
    
    print("\n" + "=" * 50)
    print("✅ All Phase 2 tests passed!")
