# Setup Guide for BSPF Package

## Installation

The package is installed in **editable mode**, which means changes to the source code are immediately available without reinstalling.

### Basic Installation

```bash
# From the repository root directory
cd /home/moulin/Dropbox/workspace/bspf
python -m pip install -e .
```

### Verify Installation

```python
# Test basic import
from bspf import bspf1d
print("✓ Package installed successfully!")

# Test GPU version (if CuPy is available)
try:
    from bspf.bspf1d_gpu import bspf1d as bspf1d_gpu
    print("✓ GPU version available")
except ImportError:
    print("⚠ GPU version not available (CuPy not installed)")
```

## Usage

### CPU Version (Standard)

```python
from bspf import bspf1d
import numpy as np

# Create grid
x = np.linspace(0.0, 1.0, 1000)
f = np.sin(2.0 * np.pi * x)

# Create BSPF instance
bspf = bspf1d.from_grid(degree=3, x=x)

# Compute derivative
df, f_spline = bspf.differentiate(f, k=1)
```

### GPU Version (CuPy)

```python
from bspf.bspf1d_gpu import bspf1d
import numpy as np

# Create grid
x = np.linspace(0.0, 1.0, 10000)
f = np.sin(2.0 * np.pi * x)

# Create GPU-accelerated BSPF instance
bspf = bspf1d.from_grid(degree=3, x=x, use_gpu=True)

# Compute derivative (runs on GPU)
df, f_spline = bspf.differentiate(f, k=1)
```

## Optional: Install CuPy for GPU Acceleration

To use the GPU-accelerated version, install CuPy:

```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x
pip install cupy-cuda12x

# Or use conda (handles CUDA automatically)
conda install -c conda-forge cupy
```

## Running Examples

```bash
# Run CPU example
python examples/1d_differentiation.py

# Run GPU example (if CuPy is installed)
python examples/bspf1d_gpu_example.py
```

## Troubleshooting

### ModuleNotFoundError: No module named 'bspf'

If you get this error, make sure you've installed the package:
```bash
python -m pip install -e .
```

### ImportError when using GPU version

If you get an error about CuPy not being available:
- Install CuPy (see above)
- Or use `use_gpu=False` to use the CPU version

### Changes not reflected

If code changes aren't being picked up:
- Make sure you installed with `-e` (editable mode)
- Restart your Python interpreter/kernel



