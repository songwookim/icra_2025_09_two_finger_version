#!/usr/bin/env python3
"""
Generate sample CSV data for testing the stiffness calculator.
This creates a synthetic dataset similar to what data_logger_node produces.
"""

import csv
import numpy as np
from pathlib import Path


def generate_sample_data(output_path: str, duration: float = 10.0, rate_hz: float = 100.0):
    """
    Generate sample CSV data with synthetic EMG, force, and position data.
    
    Args:
        output_path: Path to save the CSV file
        duration: Duration of the trial in seconds
        rate_hz: Sampling rate in Hz
    """
    n_samples = int(duration * rate_hz)
    t = np.linspace(0, duration, n_samples)
    
    # Generate synthetic data
    # EMG: Simulated muscle activation with some noise
    emg_data = []
    for ch in range(8):
        # Different frequency for each channel
        freq = 0.5 + ch * 0.2
        emg = 100 + 50 * np.sin(2 * np.pi * freq * t) + np.random.randn(n_samples) * 10
        emg = np.clip(emg, 0, 1024)
        emg_data.append(emg)
    
    # Force sensors: Simulated force with gradual increase
    force_s2 = {
        'fx': 0.5 * t + 0.3 * np.sin(2 * np.pi * 0.3 * t) + np.random.randn(n_samples) * 0.1,
        'fy': 0.3 * t + 0.2 * np.sin(2 * np.pi * 0.5 * t) + np.random.randn(n_samples) * 0.1,
        'fz': 0.8 * t + 0.4 * np.sin(2 * np.pi * 0.4 * t) + np.random.randn(n_samples) * 0.1,
        'tx': 0.1 * np.sin(2 * np.pi * 0.8 * t) + np.random.randn(n_samples) * 0.05,
        'ty': 0.1 * np.sin(2 * np.pi * 0.6 * t) + np.random.randn(n_samples) * 0.05,
        'tz': 0.1 * np.sin(2 * np.pi * 0.7 * t) + np.random.randn(n_samples) * 0.05,
    }
    
    force_s3 = {
        'fx': 0.4 * t + 0.25 * np.sin(2 * np.pi * 0.35 * t) + np.random.randn(n_samples) * 0.1,
        'fy': 0.35 * t + 0.15 * np.sin(2 * np.pi * 0.45 * t) + np.random.randn(n_samples) * 0.1,
        'fz': 0.7 * t + 0.35 * np.sin(2 * np.pi * 0.42 * t) + np.random.randn(n_samples) * 0.1,
        'tx': 0.08 * np.sin(2 * np.pi * 0.75 * t) + np.random.randn(n_samples) * 0.05,
        'ty': 0.08 * np.sin(2 * np.pi * 0.65 * t) + np.random.randn(n_samples) * 0.05,
        'tz': 0.08 * np.sin(2 * np.pi * 0.72 * t) + np.random.randn(n_samples) * 0.05,
    }
    
    # End-effector position: Simulated movement trajectory
    ee_pos = {
        'px': 0.5 + 0.1 * t + 0.02 * np.sin(2 * np.pi * 0.5 * t),
        'py': 0.3 + 0.08 * t + 0.015 * np.sin(2 * np.pi * 0.4 * t),
        'pz': 0.2 + 0.05 * t + 0.01 * np.sin(2 * np.pi * 0.6 * t),
    }
    
    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header (matching data_logger_node format)
        header = ['t_sec', 't_nanosec']
        # s2 force
        header += ['s2_fx', 's2_fy', 's2_fz', 's2_tx', 's2_ty', 's2_tz', 
                   's2_stamp_sec', 's2_stamp_nsec']
        # s3 force
        header += ['s3_fx', 's3_fy', 's3_fz', 's3_tx', 's3_ty', 's3_tz', 
                   's3_stamp_sec', 's3_stamp_nsec']
        # EE position
        header += ['ee_px', 'ee_py', 'ee_pz', 'ee_stamp_sec', 'ee_stamp_nsec']
        # EE MF position (not used in this example)
        header += ['ee_mf_px', 'ee_mf_py', 'ee_mf_pz', 'ee_mf_stamp_sec', 'ee_mf_stamp_nsec']
        # EE TH position (not used in this example)
        header += ['ee_th_px', 'ee_th_py', 'ee_th_pz', 'ee_th_stamp_sec', 'ee_th_stamp_nsec']
        # Deformity (not used in this example)
        header += ['deform_circ', 'deform_circ_stamp_sec', 'deform_circ_stamp_nsec']
        header += ['deform_ecc', 'deform_ecc_stamp_sec', 'deform_ecc_stamp_nsec']
        # EMG
        header += [f'emg_ch{i+1}' for i in range(8)] + ['emg_stamp_sec', 'emg_stamp_nsec']
        
        writer.writerow(header)
        
        # Data rows
        for i in range(n_samples):
            t_sec = int(t[i])
            t_nanosec = int((t[i] - t_sec) * 1e9)
            
            row = [t_sec, t_nanosec]
            
            # s2 force
            row += [force_s2['fx'][i], force_s2['fy'][i], force_s2['fz'][i],
                   force_s2['tx'][i], force_s2['ty'][i], force_s2['tz'][i],
                   t_sec, t_nanosec]
            
            # s3 force
            row += [force_s3['fx'][i], force_s3['fy'][i], force_s3['fz'][i],
                   force_s3['tx'][i], force_s3['ty'][i], force_s3['tz'][i],
                   t_sec, t_nanosec]
            
            # EE position
            row += [ee_pos['px'][i], ee_pos['py'][i], ee_pos['pz'][i],
                   t_sec, t_nanosec]
            
            # EE MF/TH (empty for now)
            row += ['', '', '', '', '']
            row += ['', '', '', '', '']
            
            # Deformity (empty for now)
            row += ['', '', '']
            row += ['', '', '']
            
            # EMG
            for ch in range(8):
                row.append(emg_data[ch][i])
            row += [t_sec, t_nanosec]
            
            writer.writerow(row)
    
    print(f"Generated sample data: {output_path}")
    print(f"  Duration: {duration}s, Samples: {n_samples}, Rate: {rate_hz}Hz")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample CSV data for testing')
    parser.add_argument('--output', '-o', type=str, 
                       default='/tmp/sample_trial.csv',
                       help='Output CSV path')
    parser.add_argument('--duration', '-d', type=float, default=10.0,
                       help='Duration in seconds')
    parser.add_argument('--rate', '-r', type=float, default=100.0,
                       help='Sampling rate in Hz')
    
    args = parser.parse_args()
    generate_sample_data(args.output, args.duration, args.rate)
