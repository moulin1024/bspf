# Debugging Implicit Type Conversions

This guide explains how to catch implicit NumPy/CuPy conversion errors during development, even when running on CPU.

## The Problem

When writing GPU-compatible code, it's easy to accidentally mix NumPy and CuPy arrays. On GPU, CuPy will raise errors like:

```
TypeError: Implicit conversion to a NumPy array is not allowed. 
Please use `.get()` to construct a NumPy array explicitly.
```

However, on CPU with NumPy, these errors won't appear because NumPy allows implicit conversions. This makes it hard to catch bugs during development.

## Solution: Use CuPy's Strict Mode

CuPy provides a strict mode that disallows implicit conversions between NumPy and CuPy arrays. You can enable this even when running on CPU to catch these errors early.

## Usage

### Method 1: Environment Variable

Set the `BSPF_STRICT_MODE` environment variable before running your script:

```bash
export BSPF_STRICT_MODE=1
python your_script.py
```

Or inline:

```bash
BSPF_STRICT_MODE=1 python your_script.py
```

### Method 2: Enable in Code

Import and enable strict mode at the start of your script:

```python
from bspf.debug_utils import enable_strict_mode

# Enable strict mode
enable_strict_mode()

# Now your code will raise errors on implicit conversions
import cupy as cp
import numpy as np

x = cp.array([1, 2, 3])
y = np.array([4, 5, 6])
z = x + y  # This will raise TypeError!
```

### Method 3: Use in Tests

Add strict mode to your test setup:

```python
import pytest
from bspf.debug_utils import enable_strict_mode

@pytest.fixture(autouse=True)
def setup_strict_mode():
    """Enable strict mode for all tests."""
    try:
        enable_strict_mode()
    except RuntimeError:
        # CuPy not available, skip strict mode
        pass
```

## Example: Testing Your Code

Here's a complete example:

```python
#!/usr/bin/env python3
"""Example script with strict mode enabled."""

from bspf.debug_utils import enable_strict_mode
import cupy as cp
import numpy as np

# Enable strict mode
enable_strict_mode()

# This will work: both CuPy arrays
x = cp.array([1.0, 2.0, 3.0])
y = cp.array([4.0, 5.0, 6.0])
z = x + y  # OK
print(f"CuPy + CuPy: {z.get()}")

# This will FAIL: mixing CuPy and NumPy
x = cp.array([1.0, 2.0, 3.0])
y = np.array([4.0, 5.0, 6.0])
try:
    z = x + y  # TypeError!
except TypeError as e:
    print(f"Caught error: {e}")
    # Fix: use explicit conversion
    z = x + cp.asarray(y)  # OK
```

## Running the Test Script

A test script is provided to verify strict mode is working:

```bash
# Run with strict mode enabled
BSPF_STRICT_MODE=1 python test/test_debug_strict_mode.py

# Or enable in the script itself
python test/test_debug_strict_mode.py
```

## What Gets Caught

Strict mode will catch:

1. **Mixing CuPy and NumPy in operations:**
   ```python
   x = cp.array([1, 2, 3])
   y = np.array([4, 5, 6])
   z = x + y  # TypeError!
   ```

2. **Using `np.asarray()` on CuPy arrays:**
   ```python
   x = cp.array([1, 2, 3])
   y = np.asarray(x)  # TypeError!
   ```

3. **Passing CuPy arrays to NumPy functions:**
   ```python
   x = cp.array([1, 2, 3])
   y = np.sum(x)  # May raise TypeError depending on NumPy version
   ```

## What Still Works

Explicit conversions are still allowed:

```python
# These are OK:
x = cp.array([1, 2, 3])
y = cp.asarray(np.array([4, 5, 6]))  # Explicit conversion
z = x.get()  # Explicit conversion to NumPy
```

## Requirements

- CuPy must be installed (even if not using GPU)
- CuPy version that supports `cupyx.disable_implicit_conversion()`

## Limitations

- Strict mode only works with CuPy installed
- Some older CuPy versions may not support strict mode
- This is a development/debugging tool, not for production use

## Tips

1. **Enable strict mode in CI/CD:** Add `BSPF_STRICT_MODE=1` to your test environment
2. **Use in development:** Enable strict mode when writing new GPU-compatible code
3. **Disable for production:** Don't enable strict mode in production code (it adds overhead)

## See Also

- `test/test_debug_strict_mode.py` - Example test script
- `src/bspf/debug_utils.py` - Implementation details

















