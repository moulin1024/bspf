"""
Grid mapping and domain decomposition utilities for BSPF.

This module provides functions for creating adaptive grid mappings using
sigmoid functions, allowing for domain decomposition and grid refinement.
"""

import numpy as np
import sympy as sp


def logistic(x):
    """σ(x) = 1 / (1 + exp(-x)) (symbolic)."""
    return 1 / (1 + sp.exp(-x))


def build_multi_sigmoid_expr(t, centers, sharpness, heights=None, baseline=None, normalize=True, domain=None):
    """
    Build a multi-sigmoid mapping expression.
    
    φ_raw(t) = baseline*t + Σ h_i * σ(k_i*(t - c_i))
    If normalize: φ(t) = (φ_raw(t) - φ_raw(a)) / (φ_raw(b) - φ_raw(a))
    
    Parameters
    ----------
    t : sympy.Symbol
        Symbolic variable
    centers : list
        List of sigmoid centers
    sharpness : list
        List of sigmoid sharpness values
    heights : list, optional
        List of sigmoid heights (default: all 1)
    baseline : sympy.Basic or float, optional
        Baseline slope (default: symbolic 'm')
    normalize : bool, optional
        If True, maps domain to domain (default: True)
    domain : tuple, optional
        Tuple (a, b) defining the domain interval. If None, uses [0, 1]
    
    Returns
    -------
    sympy.Basic
        Symbolic expression for the mapping φ(t)
    """
    if heights is None:
        heights = [sp.Integer(1)] * len(centers)
    if baseline is None:
        baseline = sp.symbols('m', positive=True)
    if domain is None:
        domain = (0, 1)
    
    a, b = domain
    a_sym = sp.Float(a) if isinstance(a, (int, float)) else a
    b_sym = sp.Float(b) if isinstance(b, (int, float)) else b

    phi_raw = baseline * t
    for c, k, h in zip(centers, sharpness, heights):
        phi_raw += h * logistic(k * (t - c))

    if not normalize:
        return phi_raw

    phi_a = phi_raw.subs(t, a_sym)
    phi_b = phi_raw.subs(t, b_sym)
    denom = phi_b - phi_a
    return (phi_raw - phi_a) / denom * (b_sym - a_sym) + a_sym


def transform_to_unit_interval(x, domain):
    """Transform from [a,b] to [0,1]"""
    a, b = domain
    return (x - a) / (b - a)


def transform_from_unit_interval(s, domain):
    """Transform from [0,1] to [a,b]"""
    a, b = domain
    return s * (b - a) + a


def validate_domain(domain):
    """
    Validate domain parameter.
    
    Parameters
    ----------
    domain : tuple, list, or None
        Domain interval (a, b)
    
    Returns
    -------
    tuple
        Validated domain tuple
    
    Raises
    ------
    ValueError
        If domain is invalid
    """
    if domain is None:
        return (0, 1)
    
    if not isinstance(domain, (list, tuple)) or len(domain) != 2:
        raise ValueError("domain must be a tuple/list of length 2: (a, b)")
    
    a, b = domain
    if not isinstance(a, (int, float, sp.Basic)) or not isinstance(b, (int, float, sp.Basic)):
        raise ValueError("domain endpoints must be numeric or symbolic")
    
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a >= b:
        raise ValueError("domain must satisfy a < b")
    
    return tuple(domain)


def build_expr_via_connections_with_values(p_vals, k_vals, h_vals, m_val, normalize=True, domain=None):
    """
    Numeric-parameter builder for arbitrary domain intervals.
    
    Parameters
    ----------
    p_vals : list
        List of length K-1 of internal connection points in (0,1), strictly increasing
    k_vals : list
        List of length K (>0) sharpness values
    h_vals : list
        List of length K (>=0) heights
    m_val : float
        Scalar (>0) baseline slope
    normalize : bool, optional
        If True, maps exactly domain → domain (default: True)
    domain : tuple, optional
        Tuple (a, b) defining the domain interval. If None, uses [0, 1]
    
    Returns
    -------
    expr_v : sympy.Basic
        SymPy expression φ(t) depending only on t
    dexpr_v : sympy.Basic
        SymPy expression dφ/dt depending only on t
    centers : list
        List of SymPy midpoints [c1, ..., cK] computed from p_vals in domain coordinates
    
    Raises
    ------
    ValueError
        If input parameters are invalid
    """
    domain = validate_domain(domain)
    a, b = domain
    
    K = len(k_vals)
    if len(h_vals) != K:
        raise ValueError("len(h_vals) must equal len(k_vals) (= K).")
    if len(p_vals) != max(0, K - 1):
        raise ValueError("len(p_vals) must be K-1.")
    if K >= 2:
        if not all(0.0 < p < 1.0 for p in p_vals):
            raise ValueError("All p_vals must lie strictly inside (0,1).")
        if not all(p_vals[i] < p_vals[i+1] for i in range(K - 2)):
            raise ValueError("p_vals must be strictly increasing.")

    # Transform connection points to domain coordinates
    # Boundaries in unit interval, then transform to domain
    b_unit = [0.0] + list(p_vals) + [1.0]
    b_domain = [transform_from_unit_interval(p, domain) for p in b_unit]
    
    # Midpoint centers in domain coordinates
    centers = [(b_domain[i] + b_domain[i+1]) / 2 for i in range(K)]

    # Build φ using the existing builder (keep numerics as Floats; no simplify)
    t = sp.Symbol('t')
    expr_v = build_multi_sigmoid_expr(
        t,
        centers=centers,
        sharpness=[sp.Float(k) for k in k_vals],
        heights=[sp.Float(h) for h in h_vals],
        baseline=sp.Float(m_val),
        normalize=normalize,
        domain=domain,
    )
    dexpr_v = sp.diff(expr_v, t)
    return expr_v, dexpr_v, centers


def create_adaptive_mapping(domain_source, domain_target, n_segments=2, sharpness_range=(10, 20), 
                          height_range=(0.1, 0.5), baseline_slope=0.1, normalize=True):
    """
    Create an adaptive mapping from source domain to target domain.
    
    Parameters
    ----------
    domain_source : tuple
        Tuple (a, b) - source domain interval
    domain_target : tuple
        Tuple (c, d) - target domain interval  
    n_segments : int, optional
        Number of sigmoid segments (default: 2)
    sharpness_range : tuple, optional
        Range for sigmoid sharpness values (default: (10, 20))
    height_range : tuple, optional
        Range for sigmoid heights (default: (0.1, 0.5))
    baseline_slope : float, optional
        Baseline slope parameter (default: 0.1)
    normalize : bool, optional
        Whether to normalize the mapping (default: True)
    
    Returns
    -------
    expr_v : sympy.Basic
        SymPy expression for the mapping φ(t)
    dexpr_v : sympy.Basic
        SymPy expression for dφ/dt
    centers : list
        List of sigmoid centers
    """
    # Validate domains
    domain_source = validate_domain(domain_source)
    domain_target = validate_domain(domain_target)
    
    # Generate connection points for n_segments
    if n_segments <= 1:
        p_vals = []
        k_vals = [np.random.uniform(*sharpness_range)]
        h_vals = [np.random.uniform(*height_range)]
    else:
        # Evenly spaced connection points in unit interval
        p_vals = [i / n_segments for i in range(1, n_segments)]
        k_vals = [np.random.uniform(*sharpness_range) for _ in range(n_segments)]
        h_vals = [np.random.uniform(*height_range) for _ in range(n_segments)]
    
    # Build mapping from source domain to unit interval, then to target domain
    t = sp.Symbol('t')
    
    # First, create mapping from source to unit interval
    t_unit = (t - domain_source[0]) / (domain_source[1] - domain_source[0])
    
    # Create mapping from unit interval using sigmoid functions
    expr_unit, dexpr_unit, centers = build_expr_via_connections_with_values(
        p_vals, k_vals, h_vals, baseline_slope, normalize=True, domain=(0, 1)
    )
    
    # Transform to target domain
    expr_final = expr_unit * (domain_target[1] - domain_target[0]) + domain_target[0]
    
    # Apply chain rule for derivative
    dexpr_final = sp.diff(expr_final, t)
    
    # Substitute the unit interval transformation
    expr_final = expr_final.subs(t, t_unit)
    dexpr_final = dexpr_final.subs(t, t_unit)
    
    return expr_final, dexpr_final, centers


def create_simple_mapping(domain_source, domain_target=None, p_vals=None, k_vals=None, h_vals=None, m_val=0.1):
    """
    Simple interface to create mappings between arbitrary intervals.
    
    Parameters
    ----------
    domain_source : tuple
        Tuple (a, b) - source domain interval
    domain_target : tuple, optional
        Tuple (c, d) - target domain interval. If None, uses source domain
    p_vals : list, optional
        Connection points in (0,1). If None, uses [0.5]
    k_vals : list, optional
        Sharpness values. If None, uses [15.0, 15.0]
    h_vals : list, optional
        Height values. If None, uses [0.25, 0.25]
    m_val : float, optional
        Baseline slope (default: 0.1)
    
    Returns
    -------
    expr_v : sympy.Basic
        SymPy expression for the mapping φ(t)
    dexpr_v : sympy.Basic
        SymPy expression for dφ/dt
    centers : list
        List of sigmoid centers
    """
    # Set defaults
    if domain_target is None:
        domain_target = domain_source
    if p_vals is None:
        p_vals = [0.5]
    if k_vals is None:
        k_vals = [15.0, 15.0]
    if h_vals is None:
        h_vals = [0.25, 0.25]
    
    # Validate domains
    domain_source = validate_domain(domain_source)
    domain_target = validate_domain(domain_target)
    
    t = sp.Symbol('t')
    
    # If source and target domains are the same, use direct mapping
    if domain_source == domain_target:
        return build_expr_via_connections_with_values(
            p_vals, k_vals, h_vals, m_val, normalize=True, domain=domain_source
        )
    
    # Otherwise, map through unit interval
    # First normalize to [0,1]
    t_norm = (t - domain_source[0]) / (domain_source[1] - domain_source[0])
    
    # Create mapping on unit interval
    expr_unit, _, centers_unit = build_expr_via_connections_with_values(
        p_vals, k_vals, h_vals, m_val, normalize=True, domain=(0, 1)
    )
    
    # Scale to target domain
    expr_target = expr_unit * (domain_target[1] - domain_target[0]) + domain_target[0]
    
    # Substitute normalized variable
    expr_final = expr_target.subs(t, t_norm)
    dexpr_final = sp.diff(expr_final, t)
    
    # Transform centers to target domain
    centers_target = [c * (domain_target[1] - domain_target[0]) + domain_target[0] 
                     for c in centers_unit]
    
    return expr_final, dexpr_final, centers_target







