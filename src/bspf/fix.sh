#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 path/to/bspf1d_file.py" >&2
  exit 1
fi

FILE="$1"

python - "$FILE" << 'PYCODE'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()

marker = "    # ---------- private solvers ----------"
if marker not in text:
    sys.stderr.write("ERROR: marker for private solvers not found in file.\n")
    sys.exit(1)

# ---------------------------------------------------------------------
# 1) Insert or replace _spectral_derivative helper
# ---------------------------------------------------------------------
helper = '''    def _spectral_derivative(
        self,
        residual,
        k: int,
        is_complex: bool,
        enforce_zero_boundary: bool = True,
    ):
        """
        Spectral k-th derivative of 'residual'.

        - If enforce_zero_boundary is True: use an even reflection (Neumann-style
          extension) so the resulting correction has (approximately) zero values
          at the endpoints. This lets the B-spline part carry all boundary data
          (Dirichlet or Neumann).
        - Otherwise: apply a standard periodic spectral derivative on the
          original grid.
        """
        bk = self._bk
        xp, fft = bk.xp, bk.fft
        n = self.grid.n
        dx = self.grid.dx

        # Boundary-preserving path: even reflection about the right endpoint
        if enforce_zero_boundary and n >= 3:
            r = residual
            # r_ext = [r0, r1, ..., r_{n-1}, r_{n-2}, ..., r1]
            r_ext = xp.concatenate([r, r[-2:0:-1]], axis=0)
            N_ext = int(r_ext.shape[0])

            if is_complex:
                R_ext = fft.fft(r_ext)
                omega = 2.0 * xp.pi * xp.fft.fftfreq(N_ext, d=dx)
                corr_ext = fft.ifft(R_ext * (1j * omega) ** k)
            else:
                R_ext = fft.rfft(r_ext)
                k_idx = xp.arange(N_ext // 2 + 1, dtype=xp.float64)
                omega = 2.0 * xp.pi * k_idx / (N_ext * dx)
                corr_ext = fft.irfft(R_ext * (1j * omega) ** k, n=N_ext)

            # Restrict back to the original interval
            return corr_ext[:n]

        # Standard periodic spectral derivative on the original grid
        if is_complex:
            R = fft.fft(residual)
            omega = 2.0 * xp.pi * xp.fft.fftfreq(n, d=dx)
            corr = fft.ifft(R * (1j * omega) ** k)
        else:
            R = fft.rfft(residual)
            k_idx = xp.arange(n // 2 + 1, dtype=xp.float64)
            omega = 2.0 * xp.pi * k_idx / (n * dx)
            corr = fft.irfft(R * (1j * omega) ** k, n=n)

        return corr
'''

if "def _spectral_derivative(" in text:
    # Replace existing helper between its def and the private-solvers marker
    start = text.index("    def _spectral_derivative(")
    end = text.index(marker, start)
    text = text[:start] + helper + "\n\n" + text[end:]
else:
    # Insert new helper before private solvers
    text = text.replace(marker, helper + "\n\n" + marker, 1)

# ---------------------------------------------------------------------
# 2) Patch differentiate(): spectral correction block
# ---------------------------------------------------------------------
old_block_diff_v0 = """        # Use appropriate FFT based on input type
        if is_complex:
            R = fft.fft(residual)
            corr = fft.ifft(R * (1j * om) ** k)
        else:
            R = fft.rfft(residual)
            corr = fft.irfft(R * (1j * om) ** k, n=self.grid.n)

        df_final = (df + corr)
        f_spline_out = f_spline
"""

old_block_diff_v1 = """        # Spectral correction that optionally preserves Neumann BC at endpoints
        corr = self._spectral_derivative(
            residual,
            k=k,
            is_complex=is_complex,
            enforce_neumann=(neumann_bc is not None and k == 1),
        )

        df_final = (df + corr)
        f_spline_out = f_spline
"""

new_block_diff = """        # Spectral correction: zero at endpoints; B-spline handles boundary values
        corr = self._spectral_derivative(
            residual,
            k=k,
            is_complex=is_complex,
        )

        df_final = (df + corr)
        f_spline_out = f_spline
"""

if old_block_diff_v0 in text:
    text = text.replace(old_block_diff_v0, new_block_diff, 1)
elif old_block_diff_v1 in text:
    text = text.replace(old_block_diff_v1, new_block_diff, 1)
elif new_block_diff in text:
    sys.stderr.write("NOTE: differentiate() spectral block already patched.\n")
else:
    sys.stderr.write("WARNING: could not find spectral block in differentiate().\n")

# ---------------------------------------------------------------------
# 3) Patch differentiate_1_2(): spectral correction block
# ---------------------------------------------------------------------
old_block_diff12_v0 = """        residual = f_x - f_spline
        
        # Use appropriate FFT based on input type
        if is_complex:
            R = fft.fft(residual)
            corr1 = fft.ifft(R * (1j * om))
            corr2 = fft.ifft(R * (1j * om) ** 2)
        else:
            R = fft.rfft(residual)
            corr1 = fft.irfft(R * (1j * om), n=self.grid.n)
            corr2 = fft.irfft(R * (1j * om) ** 2, n=self.grid.n)

        df1 = df1_spline + corr1
        df2 = df2_spline + corr2
"""

old_block_diff12_v1 = """        residual = f_x - f_spline
        
        # Spectral correction that optionally preserves Neumann BC at endpoints
        corr1 = self._spectral_derivative(
            residual,
            k=1,
            is_complex=is_complex,
            enforce_neumann=(neumann_bc is not None),
        )
        corr2 = self._spectral_derivative(
            residual,
            k=2,
            is_complex=is_complex,
            enforce_neumann=(neumann_bc is not None),
        )

        df1 = df1_spline + corr1
        df2 = df2_spline + corr2
"""

new_block_diff12 = """        residual = f_x - f_spline
        
        # Spectral corrections: zero at endpoints; B-spline handles boundary values
        corr1 = self._spectral_derivative(
            residual,
            k=1,
            is_complex=is_complex,
        )
        corr2 = self._spectral_derivative(
            residual,
            k=2,
            is_complex=is_complex,
        )

        df1 = df1_spline + corr1
        df2 = df2_spline + corr2
"""

if old_block_diff12_v0 in text:
    text = text.replace(old_block_diff12_v0, new_block_diff12, 1)
elif old_block_diff12_v1 in text:
    text = text.replace(old_block_diff12_v1, new_block_diff12, 1)
elif new_block_diff12 in text:
    sys.stderr.write("NOTE: differentiate_1_2() spectral block already patched.\n")
else:
    sys.stderr.write("WARNING: could not find spectral block in differentiate_1_2().\n")

# ---------------------------------------------------------------------
# 4) Write back
# ---------------------------------------------------------------------
path.write_text(text)
print(f"Patched {path}")
PYCODE
