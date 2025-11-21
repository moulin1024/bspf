"""
Padé-type compact finite difference methods for differentiation.

Compact (Padé-type) first derivative on uniform, non-periodic 1D grids
using DST-I diagonalization for the interior tridiagonal system.
"""

import numpy as np
import numpy.typing as npt

try:
    from scipy.fft import dst, idst
except ImportError:
    raise ImportError("padefd requires scipy.fft. Install via: pip install scipy")

try:
    from findiff import FinDiff
except ImportError:
    raise ImportError("padefd requires findiff. Install via: pip install findiff")

Array = npt.NDArray[np.float64]


def derive_tridiag_compact_coeffs(K, as_float=False):
    """
    Symbolically derive coefficients for the centered compact (Padé-type) first-derivative:
        f'_i + a1*(f'_{i-1} + f'_{i+1})
        = (1/h) * sum_{k=1..K} b_k * (f_{i+k} - f_{i-k})
    by matching Taylor series so that terms up to h^{2K} match exactly.
    This yields a formal order p = 2K + 2 (error O(h^{p})) for the interior scheme.

    Parameters
    ----------
    K : int
        Half-width of the RHS stencil (K = 1..).
    as_float : bool
        If True, return floats; otherwise return exact SymPy rationals.

    Returns
    -------
    a1, b_dict, order
        a1 : number
            LHS coefficient multiplying (f'_{i-1}+f'_{i+1}).
        b_dict : dict[int, number]
            Mapping k -> b_k for k=1..K.
        order : int
            Formal interior order p = 2*K + 2.
    """
    try:
        import sympy as sp
    except ImportError as e:
        raise ImportError("This helper requires sympy. Install via: pip install sympy") from e

    # Unknowns: a1 and b1..bK
    a1 = sp.symbols('a1')
    b_syms = sp.symbols(' '.join([f'b{k}' for k in range(1, K+1)]))
    if K == 1:
        b_syms = (b_syms,)  # ensure tuple for K=1

    # Build linear equations by matching coefficients of h^{2q} f^{(2q+1)}, q=0..K
    eqs = []
    for q in range(0, K+1):
        # LHS coefficient of h^{2q} f^{(2q+1)}
        if q == 0:
            lhs = 1 + 2*a1
        else:
            lhs = 2*a1 / sp.factorial(2*q)

        # RHS coefficient of h^{2q} f^{(2q+1)}
        # (1/h) * sum_k b_k [f(x+kh)-f(x-kh)]
        # = 2 * sum_k b_k * k^{2q+1} * h^{2q} / (2q+1)!
        rhs = 0
        for k, bk in enumerate(b_syms, start=1):
            rhs += bk * (k**(2*q + 1))
        rhs = 2 * rhs / sp.factorial(2*q + 1)

        eqs.append(sp.Eq(lhs, rhs))

    sol = sp.solve(eqs, (a1, *b_syms), dict=True)
    if not sol:
        raise RuntimeError("No solution found for the given K.")
    sol = sol[0]

    a1_val = sp.nsimplify(sol[a1])
    b_vals = {k: sp.nsimplify(sol[b_syms[k-1]]) for k in range(1, K+1)}

    if as_float:
        a1_val = float(a1_val)
        b_vals = {k: float(v) for k, v in b_vals.items()}

    order = 2*K + 2
    return a1_val, b_vals, order


def build_schemes_table(K_list=(5, 4, 3, 2, 1), as_float=True):
    """
    Construct the schemes dict {order: {'a1':..., 'b':{...}, 'K':K}} for the given K values.
    Defaults to K=5..1, yielding orders 12,10,8,6,4 that match the standard table.
    """
    table = {}
    for K in K_list:
        a1, b_dict, order = derive_tridiag_compact_coeffs(K, as_float=as_float)
        table[order] = dict(a1=a1, b=b_dict, K=K)
    return dict(sorted(table.items(), reverse=True))


class padefd:
    """
    Compact (Padé-type) first derivative on a uniform, NON-PERIODIC 1D grid.
    Simplicity-first implementation using a DST-I diagonalization for the
    interior tridiagonal system. No TDMA/JIT paths.

    Choose the scheme by desired 'order':
        order=12  → 11-point (±5)  tridiagonal LHS (a1=5/12), RHS b_k for k=1..5   (interior ~12th superconvergence)
        order=10  →  9-point (±4)  tridiagonal LHS (a1=2/5),  RHS b_k for k=1..4
        order=8   →  7-point (±3)  tridiagonal LHS (a1=3/8),  RHS b_k for k=1..3
        order=6   →  5-point (±2)  tridiagonal LHS (a1=1/3),  RHS b_k for k=1..2
        order=4   →  3-point (±1)  tridiagonal LHS (a1=1/4),  RHS b_1

    Boundaries: handled by one-sided stencils from `findiff` over the first/last K points,
    where K = max RHS offset (half-stencil width). Interior unknowns: i = K .. N-K-1.

    Interior solve: A y = rhs via DST-I:
        A = I + a1*T1  ⇒  λ_j = 1 + 2 a1 cos(jπ/(M+1)),  j=1..M
        y = IDST_I( DST_I(rhs) / λ )

    Parameters
    ----------
    N : int
        Number of grid points.
    h : float
        Grid spacing.
    order : {12, 10, 8, 6, 4}
        Selects the stencil/coefficients.
    acc : int or None
        Accuracy for `findiff` boundary rows. Default: min(order, 10).

    Usage
    -----
    op = padefd(N=513, h=1/512, order=12)
    df = op(f)                     # f shape (N,) or (N, nfields)
    """

    _SCHEMES = {
        12: dict(a1=5.0/12.0, b={1:7.0/9.0, 2:5.0/63.0, 3:-5.0/672.0, 4:1.0/1512.0, 5:-1.0/30240.0}, K=5),
        10: dict(a1=2.0/5.0,  b={1:39.0/50.0, 2:1.0/15.0, 3:-1.0/210.0, 4:1.0/4200.0},                 K=4),
        8:  dict(a1=3.0/8.0,  b={1:25.0/32.0, 2:1.0/20.0, 3:-1.0/480.0},                                K=3),
        6:  dict(a1=1.0/3.0,  b={1:7.0/9.0,  2:1.0/36.0},                                               K=2),
        4:  dict(a1=1.0/4.0,  b={1:3.0/4.0},                                                            K=1),
    }

    def __init__(self, N, h, order=12, acc=None):
        if order not in self._SCHEMES:
            raise ValueError(f"Unsupported order {order}. Choose from {sorted(self._SCHEMES)}.")
        params = self._SCHEMES[order]

        self.N = int(N)
        self.h = float(h)
        self.order = int(order)
        # Higher 'acc' isn't always more stable; cap by default
        self.acc = int(acc if acc is not None else min(self.order, 10))

        self.a1 = float(params["a1"])
        self.ks = np.array(sorted(params["b"].keys()), dtype=int)
        self.bs = np.array([params["b"][k] for k in self.ks], dtype=float)
        self.K  = int(params["K"])

        # Minimal N so interior & boundary blocks exist: N >= 4K + 1
        if self.N < 4*self.K + 1:
            raise ValueError(f"N must be >= {4*self.K + 1} for order {self.order} (K={self.K}).")

        # Interior range and size
        self.i0, self.i1 = self.K, self.N - self.K - 1
        self.M = self.i1 - self.i0 + 1

        # DST-I eigenvalues λ_j for A = I + a1 T1
        j = np.arange(1, self.M + 1, dtype=float)
        self._lam = 1.0 + 2.0*self.a1 * np.cos(np.pi * j / (self.M + 1))

        # Boundary derivative operators: keep only rows we need
        D = FinDiff(0, self.h, 1, acc=self.acc).matrix((self.N,)).tocsr()
        self.D_top   = D[0:self.K, :]          # fp[0..K-1]
        self.D_bot   = D[-self.K:, :]          # fp[N-K..N-1]
        self.row_lhs = D[self.K - 1, :]        # fp[K-1]  (for left interior coupling)
        self.row_rhs = D[self.N - self.K, :]   # fp[N-K]  (for right interior coupling)

    # Vectorized RHS assembly via direct slices (faster than sparse matvec)
    def _assemble_rhs(self, f):
        rhs = np.zeros(self.M, dtype=float)
        for k, c in zip(self.ks, self.bs):
            rhs += c * f[self.i0 + k : self.i1 + 1 + k]
            rhs -= c * f[self.i0 - k : self.i1 + 1 - k]
        rhs /= self.h
        # Coupling to known boundary derivative values adjacent to interior
        rhs[0]  -= self.a1 * (self.row_lhs @ f)
        rhs[-1] -= self.a1 * (self.row_rhs @ f)
        return rhs

    def __call__(self, f):
        f = np.asarray(f, dtype=float)

        if f.ndim == 1:
            if f.size != self.N:
                raise ValueError(f"Expected f.size == {self.N}, got {f.size}")
            fp = np.empty_like(f)

            # boundary derivatives
            fp[:self.K]  = self.D_top @ f
            fp[-self.K:] = self.D_bot @ f

            # interior via DST-I
            rhs = self._assemble_rhs(f)
            y = idst(dst(rhs, type=1, norm='ortho') / self._lam, type=1, norm='ortho')
            fp[self.i0:self.i1 + 1] = y
            return fp

        elif f.ndim == 2 and f.shape[0] == self.N:
            # simple per-column apply; vectorized enough for clarity
            out = np.empty_like(f)
            for j in range(f.shape[1]):
                out[:, j] = self(f[:, j])
            return out

        else:
            raise ValueError("f must be shape (N,) or (N, nfields)")

