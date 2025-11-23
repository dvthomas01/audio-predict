"""
Generate human-readable reports from benchmark results.
"""

import json
from pathlib import Path
from typing import Dict, List


def load_benchmark_results(results_dir: str) -> Dict:
    """Load all benchmark result files."""
    results_dir = Path(results_dir)
    results = {}
    
    for i in range(1, 5):
        result_file = results_dir / f"benchmark{i}_*.json"
        files = list(results_dir.glob(f"benchmark{i}_*.json"))
        if files:
            with open(files[0], 'r') as f:
                results[f'benchmark_{i}'] = json.load(f)
    
    return results


def generate_benchmark_1_report(results: Dict) -> str:
    """Generate report for Benchmark 1: Vary prediction horizon."""
    if 'benchmark_1' not in results:
        return "Benchmark 1 results not found."
    
    data = results['benchmark_1']
    report = []
    report.append("="*70)
    report.append("BENCHMARK 1: Fixed Context Duration → Vary Prediction Horizon (k)")
    report.append("="*70)
    report.append(f"Fixed context duration: {data['fixed_context_duration_seconds']}s ({data['fixed_context_size_frames']} frames)")
    report.append(f"Prediction horizons tested: {data['prediction_horizons']}")
    report.append("")
    
    # Create comparison table
    report.append(f"{'k':<6} {'Model':<12} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'Cosine Sim':<12}")
    report.append("-"*70)
    
    for k_str, k_results in data['results'].items():
        k_val = int(k_str.split('=')[1])
        for model_type, metrics in k_results.items():
            report.append(
                f"{k_val:<6} {model_type:<12} {metrics['mse']:<12.6f} "
                f"{metrics['mae']:<12.6f} {metrics['rmse']:<12.6f} "
                f"{metrics['cosine_similarity']:<12.6f}"
            )
    
    report.append("="*70)
    return "\n".join(report)


def generate_benchmark_2_report(results: Dict) -> str:
    """Generate report for Benchmark 2: Vary context size."""
    if 'benchmark_2' not in results:
        return "Benchmark 2 results not found."
    
    data = results['benchmark_2']
    report = []
    report.append("="*70)
    report.append("BENCHMARK 2: Vary Context Duration → Fixed Prediction Horizon (k)")
    report.append("="*70)
    report.append(f"Fixed k: {data['fixed_k']} frames")
    report.append(f"Context durations tested: {data['context_durations_seconds']}s")
    report.append("")
    
    # Create comparison table
    report.append(f"{'Duration':<12} {'Model':<12} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'Cosine Sim':<12}")
    report.append("-"*70)
    
    for duration_str, duration_results in data['results'].items():
        for model_type, metrics in duration_results.items():
            report.append(
                f"{duration_str:<12} {model_type:<12} {metrics['mse']:<12.6f} "
                f"{metrics['mae']:<12.6f} {metrics['rmse']:<12.6f} "
                f"{metrics['cosine_similarity']:<12.6f}"
            )
    
    report.append("="*70)
    return "\n".join(report)


def generate_benchmark_3_report(results: Dict) -> str:
    """Generate report for Benchmark 3: Vary model size."""
    if 'benchmark_3' not in results:
        return "Benchmark 3 results not found."
    
    data = results['benchmark_3']
    report = []
    report.append("="*70)
    report.append("BENCHMARK 3: Vary Model Size")
    report.append("="*70)
    report.append(f"Context size: {data['context_size']} frames")
    report.append(f"Prediction horizon: {data['prediction_horizon']} frames")
    report.append("")
    
    # Create comparison table
    report.append(f"{'Model':<12} {'Params':<15} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'Cosine Sim':<12}")
    report.append("-"*70)
    
    for model_type, metrics in data['results'].items():
        num_params = metrics.get('num_parameters', 'N/A')
        if isinstance(num_params, int):
            num_params = f"{num_params:,}"
        report.append(
            f"{model_type:<12} {num_params:<15} {metrics['mse']:<12.6f} "
            f"{metrics['mae']:<12.6f} {metrics['rmse']:<12.6f} "
            f"{metrics['cosine_similarity']:<12.6f}"
        )
    
    report.append("="*70)
    return "\n".join(report)


def generate_benchmark_4_report(results: Dict) -> str:
    """Generate report for Benchmark 4: Vary dataset size."""
    if 'benchmark_4' not in results:
        return "Benchmark 4 results not found."
    
    data = results['benchmark_4']
    report = []
    report.append("="*70)
    report.append("BENCHMARK 4: Vary Dataset Size")
    report.append("="*70)
    report.append(f"Context size: {data['context_size']} frames")
    report.append(f"Prediction horizon: {data['prediction_horizon']} frames")
    report.append(f"Total files in dataset: {data['total_files']}")
    report.append("")
    
    # Create comparison table
    report.append(f"{'Dataset':<12} {'Model':<12} {'Train Files':<15} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'Cosine Sim':<12}")
    report.append("-"*70)
    
    for size_name, size_results in data['results'].items():
        for model_type, metrics in size_results.items():
            train_files = metrics.get('num_training_files', 'N/A')
            report.append(
                f"{size_name:<12} {model_type:<12} {train_files:<15} {metrics['mse']:<12.6f} "
                f"{metrics['mae']:<12.6f} {metrics['rmse']:<12.6f} "
                f"{metrics['cosine_similarity']:<12.6f}"
            )
    
    report.append("="*70)
    return "\n".join(report)


def generate_full_report(results_dir: str, output_file: str = "benchmark_report.txt"):
    """Generate a full report from all benchmark results."""
    results = load_benchmark_results(results_dir)
    
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("COMPREHENSIVE BENCHMARK REPORT")
    report_lines.append("="*70)
    report_lines.append("")
    
    if 'benchmark_1' in results:
        report_lines.append(generate_benchmark_1_report(results))
        report_lines.append("")
    
    if 'benchmark_2' in results:
        report_lines.append(generate_benchmark_2_report(results))
        report_lines.append("")
    
    if 'benchmark_3' in results:
        report_lines.append(generate_benchmark_3_report(results))
        report_lines.append("")
    
    if 'benchmark_4' in results:
        report_lines.append(generate_benchmark_4_report(results))
        report_lines.append("")
    
    report_text = "\n".join(report_lines)
    
    # Save to file
    output_path = Path(results_dir) / output_file
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print(f"Report saved to {output_path}")
    print("\n" + report_text)
    
    return report_text


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="benchmark_report.txt")
    
    args = parser.parse_args()
    generate_full_report(args.results_dir, args.output)

