# Scripts Directory

This directory contains utility scripts for the ICRA 2025 Two-Finger Version project.

## Available Scripts

### 🎯 Variable Stiffness Calculation

Calculate variable stiffness K from logged CSV data using EMG, force sensors, and end-effector position.

**Main Script:**
- `calculate_stiffness.py` - Calculate stiffness K from CSV data

**Supporting Scripts:**
- `generate_sample_data.py` - Generate synthetic test data
- `example_stiffness_usage.py` - Usage examples and demonstrations

**Documentation:**
- `QUICKSTART.md` - Quick start guide (Korean)
- `README_STIFFNESS.md` - Comprehensive documentation
- `IMPLEMENTATION_SUMMARY.md` - Technical details and theory

**Quick Start:**
```bash
# Calculate stiffness from your data
python3 scripts/calculate_stiffness.py /path/to/your/trial_data.csv

# See help for all options
python3 scripts/calculate_stiffness.py --help

# Run examples
python3 scripts/example_stiffness_usage.py
```

### 📡 Other Scripts

- `falcon_position_reader.py` - Falcon device position reader utility

## Dependencies

For stiffness calculation:
```bash
pip install numpy scipy matplotlib
```

## More Information

See individual documentation files for detailed usage instructions.
