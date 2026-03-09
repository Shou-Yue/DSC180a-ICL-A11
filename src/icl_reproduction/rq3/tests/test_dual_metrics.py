#!/usr/bin/env python
"""
Test suite for RQ3 dual metrics (query accuracy vs in-context accuracy).

Validates:
- compute_dual_metrics() function
- Results storage format
- Commercial classification pipeline with dual metrics
- Data consistency across metrics
"""

import sys
import os
import json
from pathlib import Path

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def test_compute_dual_metrics_basic():
    """Test that compute_dual_metrics runs without errors"""
    print("Testing compute_dual_metrics basic functionality...")
    
    from rq3.dataset import BinaryClassificationDataset
    from rq3.llm_providers import MockLLMProvider
    
    # Create small dataset
    dataset = BinaryClassificationDataset(
        d=5,
        N=3,
        num_tasks=1,
        R=1.0,
        seed=42
    )
    
    # Create mock provider
    mock_provider = MockLLMProvider()
    
    # Import the function
    from rq3.run_rq3 import compute_dual_metrics
    
    # Run computation
    try:
        result = compute_dual_metrics(
            dataset=dataset,
            provider=mock_provider,
            task_idx=0,
            system_prompt="Test prompt"
        )
        
        # Check return structure
        assert isinstance(result, dict), "Result should be a dict"
        assert 'query_acc' in result, "Missing query_acc"
        assert 'icl_acc' in result, "Missing icl_acc"
        assert 'icl_correct' in result, "Missing icl_correct"
        assert 'icl_total' in result, "Missing icl_total"
        
        # Check types
        assert isinstance(result['query_acc'], int), "query_acc should be int"
        assert isinstance(result['icl_acc'], float), "icl_acc should be float"
        assert isinstance(result['icl_correct'], int), "icl_correct should be int"
        assert isinstance(result['icl_total'], int), "icl_total should be int"
        
        # Check ranges
        assert result['query_acc'] in [0, 1], "query_acc should be 0 or 1"
        assert 0.0 <= result['icl_acc'] <= 1.0, "icl_acc should be in [0, 1]"
        assert 0 <= result['icl_correct'] <= result['icl_total'], "icl_correct should be <= icl_total"
        
        print(f"  ✅ Result: query_acc={result['query_acc']}, icl_acc={result['icl_acc']:.2%}")
        return True
    
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dual_metrics_consistency():
    """Test that metrics are computed consistently across multiple calls"""
    print("Testing dual metrics consistency...")
    
    from rq3.dataset import BinaryClassificationDataset
    from rq3.llm_providers import MockLLMProvider
    from rq3.run_rq3 import compute_dual_metrics
    
    dataset = BinaryClassificationDataset(
        d=10,
        N=5,
        num_tasks=1,
        R=1.0,
        seed=42
    )
    
    mock_provider = MockLLMProvider()
    
    # Run twice with same seed/task
    result1 = compute_dual_metrics(dataset, mock_provider, 0)
    result2 = compute_dual_metrics(dataset, mock_provider, 0)
    
    # Should get same results (mock provider is deterministic)
    assert result1['query_acc'] == result2['query_acc'], "query_acc should be consistent"
    assert abs(result1['icl_acc'] - result2['icl_acc']) < 1e-6, "icl_acc should be consistent"
    
    print(f"  ✅ Metrics are consistent across calls")
    return True


def test_dual_metrics_bounds():
    """Test that metrics are within expected bounds"""
    print("Testing dual metrics bounds...")
    
    from rq3.dataset import BinaryClassificationDataset
    from rq3.llm_providers import MockLLMProvider
    from rq3.run_rq3 import compute_dual_metrics
    
    dataset = BinaryClassificationDataset(
        d=20,
        N=10,
        num_tasks=5,
        R=3.0,
        seed=99
    )
    
    mock_provider = MockLLMProvider()
    
    for task_idx in range(5):
        result = compute_dual_metrics(dataset, mock_provider, task_idx)
        
        # Query accuracy: 0 or 1
        assert result['query_acc'] in [0, 1], f"Invalid query_acc: {result['query_acc']}"
        
        # ICL accuracy: 0.0 to 1.0
        assert 0.0 <= result['icl_acc'] <= 1.0, f"Invalid icl_acc: {result['icl_acc']}"
        
        # ICL correct/total relationship
        assert result['icl_correct'] >= 0
        assert result['icl_total'] > 0
        assert result['icl_correct'] <= result['icl_total']
        
        # Recompute accuracy from correct/total
        computed_icl_acc = result['icl_correct'] / result['icl_total']
        assert abs(computed_icl_acc - result['icl_acc']) < 1e-6, "icl_acc not consistent with icl_correct/total"
    
    print(f"  ✅ All metrics within expected bounds")
    return True


def test_dual_metrics_gap_analysis():
    """Test that we can compute generalization gap (query_acc - icl_acc)"""
    print("Testing generalization gap analysis...")
    
    from rq3.dataset import BinaryClassificationDataset
    from rq3.llm_providers import MockLLMProvider
    from rq3.run_rq3 import compute_dual_metrics
    
    dataset = BinaryClassificationDataset(
        d=10,
        N=5,
        num_tasks=3,
        R=1.0,
        seed=42
    )
    
    mock_provider = MockLLMProvider()
    gaps = []
    
    for task_idx in range(3):
        result = compute_dual_metrics(dataset, mock_provider, task_idx)
        query_acc = float(result['query_acc'])
        icl_acc = result['icl_acc']
        gap = query_acc - icl_acc
        gaps.append(gap)
        
        # Gap should be in valid range
        assert -1.0 <= gap <= 1.0, f"Invalid gap: {gap}"
    
    mean_gap = sum(gaps) / len(gaps)
    print(f"  ✅ Generalization gaps computed")
    print(f"     Mean gap: {mean_gap:+.4f}")
    print(f"     Min gap:  {min(gaps):+.4f}")
    print(f"     Max gap:  {max(gaps):+.4f}")
    
    return True


if __name__ == '__main__':
    print("="*60)
    print("🧪 RQ3 DUAL METRICS TEST SUITE")
    print("="*60)
    
    tests = [
        ("Basic Functionality", test_compute_dual_metrics_basic),
        ("Consistency", test_dual_metrics_consistency),
        ("Bounds Checking", test_dual_metrics_bounds),
        ("Gap Analysis", test_dual_metrics_gap_analysis),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"✅ Tests Passed: {passed}/{len(tests)}")
    if failed > 0:
        print(f"❌ Tests Failed: {failed}/{len(tests)}")
    else:
        print("🎉 All tests passed!")
    print("="*60)
    
    sys.exit(0 if failed == 0 else 1)