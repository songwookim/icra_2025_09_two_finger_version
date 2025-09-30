#!/usr/bin/env python3
"""
Variable Stiffness Calculation from CSV Data

This script reads logged data from the data_logger_node CSV files and calculates
variable stiffness K based on:
- EMG signals (muscle activation)
- Force sensor measurements
- End-effector position/displacement

References:
- Learning Target-Directed Skill and Variable Impedance Control From Interactive Demonstrations
- Tele-Impedance control of a virtual avatar based on EMG and M-IMU sensors
- Patient's Healthy-Limb Motion Characteristic-Based Assist-As-Needed Control Strategy

Author: Generated for ICRA 2025 Two-Finger Version
"""

import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.interpolate import interp1d


class StiffnessCalculator:
    """Calculate variable stiffness K from EMG, force, and position data."""
    
    def __init__(self, csv_path: str):
        """
        Initialize the stiffness calculator.
        
        Args:
            csv_path: Path to the CSV file from data_logger_node
        """
        self.csv_path = Path(csv_path)
        self.data = None
        self.time = None
        self.emg = None
        self.force = None
        self.position = None
        self.stiffness = None
        
    def load_data(self) -> bool:
        """
        Load and parse CSV data.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read CSV file
            with open(self.csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            if not rows:
                print(f"Error: No data in {self.csv_path}")
                return False
            
            print(f"Loaded {len(rows)} rows from {self.csv_path}")
            
            # Parse time
            t_sec = np.array([float(r['t_sec']) if r['t_sec'] else np.nan for r in rows])
            t_nanosec = np.array([float(r['t_nanosec']) if r['t_nanosec'] else np.nan for r in rows])
            self.time = t_sec + t_nanosec * 1e-9
            
            # Remove NaN values from beginning
            valid_idx = ~np.isnan(self.time)
            if not np.any(valid_idx):
                print("Error: No valid timestamps")
                return False
            
            # Normalize time to start from 0
            self.time = self.time[valid_idx]
            self.time -= self.time[0]
            
            # Parse EMG data (8 channels)
            emg_channels = []
            for i in range(1, 9):
                ch_data = []
                for r in rows:
                    val = r.get(f'emg_ch{i}', '')
                    ch_data.append(float(val) if val else np.nan)
                emg_channels.append(np.array(ch_data)[valid_idx])
            self.emg = np.array(emg_channels).T  # Shape: (n_samples, 8)
            
            # Parse force sensor data (s2 and s3)
            force_components = {}
            for sensor in ['s2', 's3']:
                for comp in ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']:
                    key = f'{sensor}_{comp}'
                    vals = []
                    for r in rows:
                        val = r.get(key, '')
                        vals.append(float(val) if val else np.nan)
                    force_components[key] = np.array(vals)[valid_idx]
            
            # Combine force data
            self.force = {
                's2': np.array([force_components['s2_fx'], 
                               force_components['s2_fy'], 
                               force_components['s2_fz']]).T,
                's3': np.array([force_components['s3_fx'], 
                               force_components['s3_fy'], 
                               force_components['s3_fz']]).T,
            }
            
            # Parse end-effector position
            ee_pos = []
            for comp in ['ee_px', 'ee_py', 'ee_pz']:
                vals = []
                for r in rows:
                    val = r.get(comp, '')
                    vals.append(float(val) if val else np.nan)
                ee_pos.append(np.array(vals)[valid_idx])
            self.position = np.array(ee_pos).T  # Shape: (n_samples, 3)
            
            print(f"Data shapes - Time: {self.time.shape}, EMG: {self.emg.shape}, "
                  f"Position: {self.position.shape}")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_emg(self, cutoff_freq: float = 10.0, fs: float = 100.0) -> np.ndarray:
        """
        Process EMG signals: rectification, low-pass filtering, normalization.
        
        Args:
            cutoff_freq: Cutoff frequency for low-pass filter (Hz)
            fs: Sampling frequency (Hz)
            
        Returns:
            Processed EMG activation levels (n_samples, 8)
        """
        # Rectify EMG (absolute value)
        emg_rect = np.abs(self.emg)
        
        # Design low-pass Butterworth filter
        nyquist = fs / 2.0
        normalized_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(4, normalized_cutoff, btype='low')
        
        # Apply filter to each channel
        emg_filtered = np.zeros_like(emg_rect)
        for i in range(emg_rect.shape[1]):
            # Skip if all NaN
            if np.all(np.isnan(emg_rect[:, i])):
                emg_filtered[:, i] = np.zeros(emg_rect.shape[0])
                continue
            
            # Handle NaN values by interpolation
            valid_mask = ~np.isnan(emg_rect[:, i])
            if np.sum(valid_mask) > 10:  # Need at least 10 valid points
                valid_indices = np.where(valid_mask)[0]
                valid_values = emg_rect[valid_mask, i]
                
                # Interpolate missing values
                if len(valid_indices) > 1:
                    interp_func = interp1d(valid_indices, valid_values, 
                                          kind='linear', fill_value='extrapolate')
                    emg_interp = interp_func(np.arange(len(emg_rect)))
                else:
                    emg_interp = np.zeros(len(emg_rect))
                
                # Apply filter
                emg_filtered[:, i] = signal.filtfilt(b, a, emg_interp)
            else:
                emg_filtered[:, i] = np.zeros(emg_rect.shape[0])
        
        # Normalize to [0, 1] for each channel
        emg_normalized = np.zeros_like(emg_filtered)
        for i in range(emg_filtered.shape[1]):
            max_val = np.max(emg_filtered[:, i])
            if max_val > 1e-6:
                emg_normalized[:, i] = emg_filtered[:, i] / max_val
            else:
                emg_normalized[:, i] = emg_filtered[:, i]
        
        return emg_normalized
    
    def calculate_force_magnitude(self, sensor: str = 's2') -> np.ndarray:
        """
        Calculate force magnitude from force sensor data.
        
        Args:
            sensor: Sensor name ('s2' or 's3')
            
        Returns:
            Force magnitude array (n_samples,)
        """
        force_data = self.force[sensor]
        # Euclidean norm of force vector
        force_mag = np.linalg.norm(force_data, axis=1)
        return force_mag
    
    def calculate_displacement(self) -> np.ndarray:
        """
        Calculate displacement from initial position.
        
        Returns:
            Displacement magnitude array (n_samples,)
        """
        if self.position is None or len(self.position) == 0:
            return np.zeros(len(self.time))
        
        # Handle NaN values
        valid_mask = ~np.isnan(self.position).any(axis=1)
        if not np.any(valid_mask):
            return np.zeros(len(self.time))
        
        # Use first valid position as reference
        first_valid_idx = np.where(valid_mask)[0][0]
        initial_pos = self.position[first_valid_idx]
        
        # Calculate displacement from initial position
        displacement = np.zeros(len(self.position))
        for i in range(len(self.position)):
            if valid_mask[i]:
                displacement[i] = np.linalg.norm(self.position[i] - initial_pos)
            else:
                # Use last valid displacement
                if i > 0:
                    displacement[i] = displacement[i-1]
        
        return displacement
    
    def calculate_stiffness(self, 
                           emg_weight: float = 0.3,
                           force_weight: float = 0.4,
                           velocity_weight: float = 0.3,
                           k_min: float = 10.0,
                           k_max: float = 1000.0,
                           window_size: int = 10) -> np.ndarray:
        """
        Calculate variable stiffness K based on EMG, force, and motion characteristics.
        
        This implements an assist-as-needed strategy where:
        - Higher EMG activation -> Higher stiffness (user is actively engaged)
        - Higher force -> Higher stiffness (task requires more support)
        - Higher velocity -> Lower stiffness (allow natural motion)
        
        Args:
            emg_weight: Weight for EMG contribution (0-1)
            force_weight: Weight for force contribution (0-1)
            velocity_weight: Weight for velocity contribution (0-1)
            k_min: Minimum stiffness value
            k_max: Maximum stiffness value
            window_size: Window size for velocity calculation
            
        Returns:
            Variable stiffness array K (n_samples,)
        """
        n_samples = len(self.time)
        
        # Process EMG to get activation levels
        emg_activation = self.process_emg()
        
        # Use mean EMG activation across all channels as muscle activation index
        muscle_activation = np.mean(emg_activation, axis=1)
        
        # Calculate force magnitude (using s2 sensor)
        force_mag = self.calculate_force_magnitude('s2')
        force_mag_normalized = force_mag / (np.max(force_mag) + 1e-6)
        
        # Calculate velocity from position
        displacement = self.calculate_displacement()
        velocity = np.gradient(displacement, self.time)
        velocity_mag = np.abs(velocity)
        
        # Smooth velocity
        if window_size > 1:
            velocity_smoothed = np.convolve(velocity_mag, 
                                           np.ones(window_size)/window_size, 
                                           mode='same')
        else:
            velocity_smoothed = velocity_mag
        
        velocity_normalized = velocity_smoothed / (np.max(velocity_smoothed) + 1e-6)
        
        # Calculate stiffness components
        # EMG component: higher activation -> higher stiffness
        k_emg = muscle_activation
        
        # Force component: higher force -> higher stiffness
        k_force = force_mag_normalized
        
        # Velocity component: higher velocity -> lower stiffness (inverted)
        k_velocity = 1.0 - velocity_normalized
        
        # Normalize weights
        total_weight = emg_weight + force_weight + velocity_weight
        if total_weight > 0:
            emg_weight /= total_weight
            force_weight /= total_weight
            velocity_weight /= total_weight
        
        # Combine components with weights
        k_normalized = (emg_weight * k_emg + 
                       force_weight * k_force + 
                       velocity_weight * k_velocity)
        
        # Map to [k_min, k_max]
        self.stiffness = k_min + k_normalized * (k_max - k_min)
        
        # Store intermediate results for visualization
        self.muscle_activation = muscle_activation
        self.force_mag = force_mag
        self.velocity = velocity_smoothed
        self.displacement = displacement
        
        return self.stiffness
    
    def save_results(self, output_path: Optional[str] = None) -> str:
        """
        Save calculated stiffness and related data to CSV.
        
        Args:
            output_path: Optional output path. If None, uses default based on input file.
            
        Returns:
            Path to the saved file
        """
        if output_path is None:
            output_path = self.csv_path.parent / f"{self.csv_path.stem}_stiffness.csv"
        else:
            output_path = Path(output_path)
        
        # Prepare data for output
        output_data = {
            'time': self.time,
            'stiffness_K': self.stiffness,
            'muscle_activation': self.muscle_activation,
            'force_magnitude': self.force_mag,
            'velocity': self.velocity,
            'displacement': self.displacement,
        }
        
        # Add EMG channels
        for i in range(self.emg.shape[1]):
            output_data[f'emg_ch{i+1}'] = self.emg[:, i]
        
        # Write to CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            headers = list(output_data.keys())
            writer.writerow(headers)
            
            # Data rows
            n_rows = len(self.time)
            for i in range(n_rows):
                row = [output_data[h][i] for h in headers]
                writer.writerow(row)
        
        print(f"Results saved to: {output_path}")
        return str(output_path)
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Generate visualization of stiffness calculation results.
        
        Args:
            save_path: Optional path to save the figure
        """
        if self.stiffness is None:
            print("Error: No stiffness data to plot. Run calculate_stiffness() first.")
            return
        
        fig, axes = plt.subplots(5, 1, figsize=(12, 14))
        
        # Plot 1: Stiffness K
        axes[0].plot(self.time, self.stiffness, 'b-', linewidth=2)
        axes[0].set_ylabel('Stiffness K', fontsize=12, fontweight='bold')
        axes[0].set_title('Variable Stiffness K Calculation Results', 
                         fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Muscle Activation (from EMG)
        axes[1].plot(self.time, self.muscle_activation, 'r-', linewidth=1.5)
        axes[1].set_ylabel('Muscle Activation\n(normalized)', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1.1)
        
        # Plot 3: Force Magnitude
        axes[2].plot(self.time, self.force_mag, 'g-', linewidth=1.5)
        axes[2].set_ylabel('Force Magnitude\n(N)', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Velocity
        axes[3].plot(self.time, self.velocity, 'm-', linewidth=1.5)
        axes[3].set_ylabel('Velocity\n(m/s)', fontsize=12, fontweight='bold')
        axes[3].grid(True, alpha=0.3)
        
        # Plot 5: Displacement
        axes[4].plot(self.time, self.displacement, 'c-', linewidth=1.5)
        axes[4].set_ylabel('Displacement\n(m)', fontsize=12, fontweight='bold')
        axes[4].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        axes[4].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        else:
            # Save to default location
            fig_path = self.csv_path.parent / f"{self.csv_path.stem}_stiffness_plot.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {fig_path}")
        
        plt.close()


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Calculate variable stiffness K from logged CSV data'
    )
    parser.add_argument(
        'csv_path',
        type=str,
        help='Path to CSV file from data_logger_node'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output CSV path (default: <input>_stiffness.csv)'
    )
    parser.add_argument(
        '--plot', '-p',
        type=str,
        default=None,
        help='Save plot to specified path (default: <input>_stiffness_plot.png)'
    )
    parser.add_argument(
        '--k-min',
        type=float,
        default=10.0,
        help='Minimum stiffness value (default: 10.0)'
    )
    parser.add_argument(
        '--k-max',
        type=float,
        default=1000.0,
        help='Maximum stiffness value (default: 1000.0)'
    )
    parser.add_argument(
        '--emg-weight',
        type=float,
        default=0.3,
        help='Weight for EMG contribution (default: 0.3)'
    )
    parser.add_argument(
        '--force-weight',
        type=float,
        default=0.4,
        help='Weight for force contribution (default: 0.4)'
    )
    parser.add_argument(
        '--velocity-weight',
        type=float,
        default=0.3,
        help='Weight for velocity contribution (default: 0.3)'
    )
    
    args = parser.parse_args()
    
    # Create calculator
    calc = StiffnessCalculator(args.csv_path)
    
    # Load data
    print("Loading data...")
    if not calc.load_data():
        print("Failed to load data. Exiting.")
        return 1
    
    # Calculate stiffness
    print("Calculating stiffness...")
    calc.calculate_stiffness(
        emg_weight=args.emg_weight,
        force_weight=args.force_weight,
        velocity_weight=args.velocity_weight,
        k_min=args.k_min,
        k_max=args.k_max
    )
    
    # Save results
    print("Saving results...")
    calc.save_results(args.output)
    
    # Generate plot
    print("Generating plot...")
    calc.plot_results(args.plot)
    
    print("Done!")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
