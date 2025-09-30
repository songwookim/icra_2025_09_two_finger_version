#!/usr/bin/env python3
"""
Example script demonstrating programmatic use of StiffnessCalculator.

This shows how to use the StiffnessCalculator class in your own Python code
for batch processing or custom analysis.
"""

import sys
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

from calculate_stiffness import StiffnessCalculator
import numpy as np


def example_basic_usage():
    """Example 1: Basic usage with default parameters."""
    print("="*60)
    print("Example 1: Basic Usage")
    print("="*60)
    
    # Generate sample data first
    import generate_sample_data
    csv_path = "/tmp/example_trial.csv"
    generate_sample_data.generate_sample_data(csv_path, duration=5.0, rate_hz=100.0)
    
    # Create calculator
    calc = StiffnessCalculator(csv_path)
    
    # Load data
    if not calc.load_data():
        print("Failed to load data")
        return
    
    # Calculate stiffness with default parameters
    stiffness = calc.calculate_stiffness()
    
    # Print statistics
    print(f"\nStiffness Statistics:")
    print(f"  Mean: {np.mean(stiffness):.2f}")
    print(f"  Std: {np.std(stiffness):.2f}")
    print(f"  Min: {np.min(stiffness):.2f}")
    print(f"  Max: {np.max(stiffness):.2f}")
    
    # Save results
    calc.save_results("/tmp/example_trial_stiffness.csv")
    calc.plot_results("/tmp/example_trial_plot.png")
    
    print("\n✓ Example 1 completed successfully\n")


def example_custom_parameters():
    """Example 2: Custom parameters for specific application."""
    print("="*60)
    print("Example 2: Custom Parameters")
    print("="*60)
    
    # Generate sample data
    import generate_sample_data
    csv_path = "/tmp/example_trial2.csv"
    generate_sample_data.generate_sample_data(csv_path, duration=8.0, rate_hz=100.0)
    
    # Create calculator
    calc = StiffnessCalculator(csv_path)
    calc.load_data()
    
    # Calculate with custom parameters
    # Use case: EMG-dominant control for rehabilitation
    stiffness = calc.calculate_stiffness(
        emg_weight=0.6,        # High EMG weight - follow user intent
        force_weight=0.2,      # Lower force weight
        velocity_weight=0.2,   # Lower velocity weight
        k_min=20.0,            # Higher minimum for safety
        k_max=500.0,           # Lower maximum for comfort
        window_size=20         # More smoothing
    )
    
    print(f"\nCustom Stiffness Statistics:")
    print(f"  Mean: {np.mean(stiffness):.2f}")
    print(f"  Std: {np.std(stiffness):.2f}")
    print(f"  Min: {np.min(stiffness):.2f}")
    print(f"  Max: {np.max(stiffness):.2f}")
    
    calc.save_results("/tmp/example_trial2_stiffness.csv")
    calc.plot_results("/tmp/example_trial2_plot.png")
    
    print("\n✓ Example 2 completed successfully\n")


def example_batch_processing():
    """Example 3: Batch processing multiple trials."""
    print("="*60)
    print("Example 3: Batch Processing")
    print("="*60)
    
    # Generate multiple sample files
    import generate_sample_data
    
    trials = []
    for i in range(3):
        csv_path = f"/tmp/batch_trial_{i+1}.csv"
        generate_sample_data.generate_sample_data(
            csv_path, 
            duration=5.0 + i*2,  # Varying durations
            rate_hz=100.0
        )
        trials.append(csv_path)
    
    # Process all trials
    results = []
    for trial_path in trials:
        print(f"\nProcessing: {trial_path}")
        calc = StiffnessCalculator(trial_path)
        
        if not calc.load_data():
            continue
        
        stiffness = calc.calculate_stiffness()
        
        # Store statistics
        results.append({
            'file': trial_path,
            'mean_stiffness': np.mean(stiffness),
            'std_stiffness': np.std(stiffness),
            'max_activation': np.max(calc.muscle_activation),
            'max_force': np.max(calc.force_mag),
        })
        
        # Save results
        output_csv = trial_path.replace('.csv', '_stiffness.csv')
        output_plot = trial_path.replace('.csv', '_plot.png')
        calc.save_results(output_csv)
        calc.plot_results(output_plot)
    
    # Print summary
    print("\n" + "="*60)
    print("Batch Processing Summary")
    print("="*60)
    for r in results:
        print(f"\n{Path(r['file']).name}:")
        print(f"  Mean Stiffness: {r['mean_stiffness']:.2f}")
        print(f"  Std Stiffness: {r['std_stiffness']:.2f}")
        print(f"  Max Activation: {r['max_activation']:.3f}")
        print(f"  Max Force: {r['max_force']:.3f}")
    
    print("\n✓ Example 3 completed successfully\n")


def example_access_intermediate_data():
    """Example 4: Access and analyze intermediate processing results."""
    print("="*60)
    print("Example 4: Access Intermediate Data")
    print("="*60)
    
    # Generate sample data
    import generate_sample_data
    csv_path = "/tmp/example_intermediate.csv"
    generate_sample_data.generate_sample_data(csv_path, duration=10.0, rate_hz=100.0)
    
    # Create and run calculator
    calc = StiffnessCalculator(csv_path)
    calc.load_data()
    calc.calculate_stiffness()
    
    # Access intermediate processing results
    print(f"\nIntermediate Data Available:")
    print(f"  Time points: {len(calc.time)}")
    print(f"  Muscle activation shape: {calc.muscle_activation.shape}")
    print(f"  Force magnitude shape: {calc.force_mag.shape}")
    print(f"  Velocity shape: {calc.velocity.shape}")
    print(f"  Displacement shape: {calc.displacement.shape}")
    
    # Custom analysis example: Find peak activity moments
    activation_peaks = np.where(calc.muscle_activation > 0.7)[0]
    if len(activation_peaks) > 0:
        print(f"\nHigh activation detected at {len(activation_peaks)} time points")
        print(f"  First peak at t={calc.time[activation_peaks[0]]:.2f}s")
        print(f"  Stiffness at peak: {calc.stiffness[activation_peaks[0]]:.2f}")
    
    # Find maximum force moment
    max_force_idx = np.argmax(calc.force_mag)
    print(f"\nMaximum force event:")
    print(f"  Time: {calc.time[max_force_idx]:.2f}s")
    print(f"  Force: {calc.force_mag[max_force_idx]:.3f}N")
    print(f"  Stiffness: {calc.stiffness[max_force_idx]:.2f}")
    print(f"  Activation: {calc.muscle_activation[max_force_idx]:.3f}")
    
    print("\n✓ Example 4 completed successfully\n")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("StiffnessCalculator Usage Examples")
    print("="*60 + "\n")
    
    try:
        example_basic_usage()
        example_custom_parameters()
        example_batch_processing()
        example_access_intermediate_data()
        
        print("="*60)
        print("All examples completed successfully!")
        print("="*60)
        print("\nGenerated files in /tmp/:")
        import subprocess
        subprocess.run("ls -lh /tmp/example* /tmp/batch* 2>/dev/null | tail -20", shell=True)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
