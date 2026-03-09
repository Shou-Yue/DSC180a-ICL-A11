#!/usr/bin/env python
"""
RQ3 Results Analysis & Visualization Script
Reproducible analysis of commercial LLM sweep results across seeds

Usage:
    python rq3/analyze_results.py             # Interactive menu
    python rq3/analyze_results.py --summary   # Quick summary
    python rq3/analyze_results.py --plots     # Generate plots
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def load_results(results_root: str = 'results/rq3/commercial') -> List[Dict]:
    """Load all results from seed directories."""
    results_root = Path(results_root)
    if not results_root.exists():
        print(f"❌ Results directory not found: {results_root}")
        return []
    
    all_results = []
    
    for provider_dir in sorted(results_root.glob('*')):
        if not provider_dir.is_dir():
            continue
            
        provider = provider_dir.name.split('_')[0]
        config_name = provider_dir.name
        
        for seed_dir in sorted(provider_dir.glob('seed_*')):
            if not seed_dir.is_dir():
                continue
                
            try:
                seed_num = int(seed_dir.name.split('_')[1])
                
                # Load summary
                summary_file = seed_dir / 'summary.json'
                if summary_file.exists():
                    with open(summary_file) as f:
                        summary = json.load(f)
                    
                    # Support both old format (accuracy) and new format (query_accuracy, mean_icl_accuracy)
                    all_results.append({
                        'provider': provider,
                        'config': config_name,
                        'seed': seed_num,
                        'query_accuracy': summary.get('query_accuracy', summary.get('accuracy', 0)),
                        'icl_accuracy': summary.get('mean_icl_accuracy', 0),
                        'query_correct': summary.get('query_correct', summary.get('correct', 0)),
                        'total': summary.get('total', 0),
                        'timestamp': summary.get('timestamp', ''),
                        'seed_dir': seed_dir
                    })
            except (ValueError, json.JSONDecodeError) as e:
                print(f"⚠️  Skipping {seed_dir}: {e}")
                continue
    
    return all_results


def print_summary(results: List[Dict]) -> None:
    """Print quick summary statistics."""
    if not results:
        print("❌ No results found")
        return
    
    print("\n" + "="*80)
    print("📊 RQ3 COMMERCIAL LLM RESULTS SUMMARY (Query Accuracy vs In-Context Accuracy)")
    print("="*80)
    
    # Group by provider and config
    by_config = {}
    for r in results:
        key = (r['provider'], r['config'])
        if key not in by_config:
            by_config[key] = {'query': [], 'icl': []}
        by_config[key]['query'].append(r['query_accuracy'])
        by_config[key]['icl'].append(r['icl_accuracy'])
    
    print(f"\nTotal configurations: {len(by_config)}")
    print(f"Total seeds: {len(results)}")
    print(f"Total tasks evaluated: {sum(r['total'] for r in results)}\n")
    
    # Sort by provider
    by_provider = {}
    for (provider, config), accs in by_config.items():
        if provider not in by_provider:
            by_provider[provider] = {'query': [], 'icl': []}
        by_provider[provider]['query'].extend(accs['query'])
        by_provider[provider]['icl'].extend(accs['icl'])
    
    print("Per-Provider Statistics (Query Acc | In-Context Acc):")
    print("-" * 80)
    for provider in sorted(by_provider.keys()):
        query_accs = by_provider[provider]['query']
        icl_accs = by_provider[provider]['icl']
        query_mean = sum(query_accs) / len(query_accs)
        icl_mean = sum(icl_accs) / len(icl_accs)
        query_std = (sum((x - query_mean)**2 for x in query_accs) / len(query_accs))**0.5 if len(query_accs) > 1 else 0
        icl_std = (sum((x - icl_mean)**2 for x in icl_accs) / len(icl_accs))**0.5 if len(icl_accs) > 1 else 0
        
        num_configs = len(set(r['config'] for r in results if r['provider'] == provider))
        print(f"\n{provider.upper():10} | {num_configs:2} configs | {len(query_accs):2} seeds")
        print(f"           | Query Accuracy   : {query_mean:.4f} ± {query_std:.4f}")
        print(f"           | In-Context Acc   : {icl_mean:.4f} ± {icl_std:.4f}")
        print(f"           | Generalization Gap: {query_mean - icl_mean:+.4f} (Query - ICL)")
    
    print("\n" + "-"*80)
    print("Per-Configuration (Mean ± Std across seeds):")
    print("-" * 80)
    
    for (provider, config), accs in sorted(by_config.items()):
        query_accs = accs['query']
        icl_accs = accs['icl']
        query_mean = sum(query_accs) / len(query_accs) if query_accs else 0
        icl_mean = sum(icl_accs) / len(icl_accs) if icl_accs else 0
        
        query_std = (sum((x - query_mean)**2 for x in query_accs) / len(query_accs))**0.5 if len(query_accs) > 1 else 0
        icl_std = (sum((x - icl_mean)**2 for x in icl_accs) / len(icl_accs))**0.5 if len(icl_accs) > 1 else 0
        
        print(f"\n{provider:8} | {config:40}")
        print(f"         | Query: {query_mean:.4f} ± {query_std:.4f} | ICL: {icl_mean:.4f} ± {icl_std:.4f}")


def generate_plots(results_root: str = 'results') -> None:
    """Generate all RQ3 visualization plots."""
    print("\n" + "="*80)
    print("📈 GENERATING PLOTS")
    print("="*80)
    
    try:
        # Import from parent directory (icl_reproduction)
        parent_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(parent_dir))
        from experiments.plots import generate_all_rq3_plots
        
        output_dir = Path('results/rq3/_plots')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🔄 Generating plots in {output_dir}...")
        generate_all_rq3_plots(results_root, str(output_dir))
        
        print("\n✅ Plot generation complete!")
        print(f"   📁 Output directory: {output_dir}")
        
        # List generated files
        plot_files = list(output_dir.glob('*.png'))
        if plot_files:
            print(f"\n   Generated plots:")
            for f in sorted(plot_files):
                print(f"     - {f.name}")
        
    except Exception as e:
        print(f"❌ Error generating plots: {e}")
        import traceback
        traceback.print_exc()


def interactive_menu(results: List[Dict]) -> None:
    """Interactive menu for exploring results."""
    while True:
        print("\n" + "="*80)
        print("🔍 RQ3 Results Explorer")
        print("="*80)
        print("\n1. View summary statistics")
        print("2. View per-provider breakdown")
        print("3. View per-config breakdown")
        print("4. View all results with seeds")
        print("5. Generate plots")
        print("6. Check file locations")
        print("0. Exit")
        
        choice = input("\nSelect option (0-6): ").strip()
        
        if choice == '1':
            print_summary(results)
        
        elif choice == '2':
            print("\n" + "-"*80)
            print("Per-Provider Accuracy (all seeds combined)")
            print("-"*80)
            by_provider = {}
            for r in results:
                if r['provider'] not in by_provider:
                    by_provider[r['provider']] = {'query': [], 'icl': []}
                by_provider[r['provider']]['query'].append(r['query_accuracy'])
                by_provider[r['provider']]['icl'].append(r['icl_accuracy'])
            
            for provider in sorted(by_provider.keys()):
                query_accs = by_provider[provider]['query']
                icl_accs = by_provider[provider]['icl']
                query_mean = sum(query_accs) / len(query_accs)
                icl_mean = sum(icl_accs) / len(icl_accs)
                print(f"  {provider:10} | Query: {query_mean:.4f} | ICL: {icl_mean:.4f} | Gap: {query_mean - icl_mean:+.4f}")
        
        elif choice == '3':
            print("\n" + "-"*80)
            print("Per-Configuration (Mean Accuracy across seeds)")
            print("-"*80)
            by_config = {}
            for r in results:
                key = r['config']
                if key not in by_config:
                    by_config[key] = {'query': [], 'icl': []}
                by_config[key]['query'].append(r['query_accuracy'])
                by_config[key]['icl'].append(r['icl_accuracy'])
            
            for config in sorted(by_config.keys()):
                query_accs = by_config[config]['query']
                icl_accs = by_config[config]['icl']
                query_mean = sum(query_accs) / len(query_accs) if query_accs else 0
                icl_mean = sum(icl_accs) / len(icl_accs) if icl_accs else 0
                print(f"  {config:50} | Query: {query_mean:.4f} | ICL: {icl_mean:.4f}")
        
        elif choice == '4':
            print("\n" + "-"*80)
            print("All Results with Seeds")
            print("-"*80)
            for r in sorted(results, key=lambda x: (x['provider'], x['config'], x['seed'])):
                print(f"  {r['provider']:8} | {r['config']:40} | Seed {r['seed']} | Query: {r['query_accuracy']:.4f} | ICL: {r['icl_accuracy']:.4f}")
        
        elif choice == '5':
            generate_plots()
        
        elif choice == '6':
            print("\n" + "-"*80)
            print("Results File Locations")
            print("-"*80)
            results_root = Path('results/rq3/commercial')
            print(f"\n  Base dir: {results_root.absolute()}")
            print(f"  Exists: {results_root.exists()}")
            
            if results_root.exists():
                provider_dirs = list(results_root.glob('*'))
                print(f"  Provider configs found: {len(provider_dirs)}")
                for pdir in sorted(provider_dirs)[:5]:  # Show first 5
                    seed_dirs = list(pdir.glob('seed_*'))
                    print(f"    - {pdir.name} ({len(seed_dirs)} seeds)")
        
        elif choice == '0':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid option")


def main():
    parser = argparse.ArgumentParser(description='RQ3 Results Analysis')
    parser.add_argument('--summary', action='store_true', help='Print summary and exit')
    parser.add_argument('--plots', action='store_true', help='Generate plots and exit')
    parser.add_argument('--interactive', action='store_true', help='Interactive menu (default)')
    parser.add_argument('--results-root', default='results/rq3/commercial', help='Results directory')
    
    args = parser.parse_args()
    
    print("\n🔄 Loading results...")
    results = load_results(args.results_root)
    print(f"✅ Loaded {len(results)} result entries\n")
    
    if args.summary:
        print_summary(results)
    elif args.plots:
        generate_plots()
    elif args.interactive or (not args.summary and not args.plots):
        if results:
            interactive_menu(results)
        else:
            print("❌ No results found to analyze")


if __name__ == '__main__':
    main()
