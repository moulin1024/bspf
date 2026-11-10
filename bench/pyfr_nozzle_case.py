"""PyFR case for the sextic Laval nozzle (L = 4, A = 1.5), Euler, matched to examples/pde/laval_nozzle_2d.py.

    python make_case.py NX NY ORDER OUT_PREFIX [--steady|--explicit]
"""
import sys
import numpy as np
import gmsh
from scipy.optimize import brentq

L, A, G = 4.0, 1.5, 1.4
h = lambda x: 1 + (A - 1) * (1 - (1 - (x / L) ** 2) ** 3)
nx, ny, order, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
mode = sys.argv[5] if len(sys.argv) > 5 else "--steady"

# ---- curved high-order quad mesh
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("nozzle")
# exact walls: h is a degree-6 polynomial in x, x linear in the curve parameter -> a degree-6 Bezier curve
from math import comb
deg = 6
sn = 0.5 - 0.5 * np.cos(np.pi * np.arange(deg + 1) / deg)
Bm = np.array([[comb(deg, k) * si**k * (1 - si)**(deg - k) for k in range(deg + 1)] for si in sn])
cx = np.linalg.solve(Bm, -L + 2 * L * sn)
cy = np.linalg.solve(Bm, h(-L + 2 * L * sn))
pb = [gmsh.model.geo.addPoint(cx[k], -cy[k], 0) for k in range(deg + 1)]
pt = [gmsh.model.geo.addPoint(cx[k], cy[k], 0) for k in range(deg + 1)]
bottom, top = gmsh.model.geo.addBezier(pb), gmsh.model.geo.addBezier(pt)
left, right = gmsh.model.geo.addLine(pb[0], pt[0]), gmsh.model.geo.addLine(pb[-1], pt[-1])
surf = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([bottom, right, -top, -left])])
for c, n in ((bottom, nx), (top, nx), (left, ny), (right, ny)):
    gmsh.model.geo.mesh.setTransfiniteCurve(c, n + 1)
gmsh.model.geo.mesh.setTransfiniteSurface(surf)
gmsh.model.geo.mesh.setRecombine(2, surf)
gmsh.model.geo.synchronize()
gmsh.model.addPhysicalGroup(1, [left], name="inlet")
gmsh.model.addPhysicalGroup(1, [right], name="outlet")
gmsh.model.addPhysicalGroup(1, [bottom, top], name="wall")
gmsh.model.addPhysicalGroup(2, [surf], name="fluid")
gmsh.model.mesh.generate(2)
gmsh.model.mesh.setOrder(max(order, 2))
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write(out + ".msh")
gmsh.finalize()

# ---- quasi-1D initial guess as an expression: Mach(x) fitted by a polynomial in t = x / L
def area_ratio(M):
    return (1 / M) * ((2 / (G + 1)) * (1 + 0.5 * (G - 1) * M * M)) ** ((G + 1) / (2 * (G - 1)))
def mach(x):
    a = h(x)
    if a <= 1 + 1e-14:
        return 1.0
    return brentq(lambda M: area_ratio(M) - a, 1 + 1e-12, 50) if x > 0 else brentq(lambda M: area_ratio(M) - a, 1e-8, 1 - 1e-12)
xf = np.linspace(-L, L, 401)
coef = np.polynomial.chebyshev.chebfit(xf / L, [mach(x) for x in xf], 14)
poly = np.polynomial.chebyshev.cheb2poly(coef)
Mexpr = "(" + " + ".join(f"({c:.15e})*pow(x/{L}, {k})" for k, c in enumerate(poly)) + ")"
Texpr = f"(1/(1 + 0.2*{Mexpr}*{Mexpr}))"
rho = f"pow({Texpr}, 2.5)"
p = f"(pow({Texpr}, 3.5)/{G})"
u = f"({Mexpr}*sqrt({Texpr}))"
hx = f"(1 + {A - 1}*(1 - pow(1 - pow(x/{L}, 2), 3)))"
dh = f"({A - 1}*6*(x/{L})*pow(1 - pow(x/{L}, 2), 2)/{L})"
v = f"({u}*y*{dh}/{hx})"
Min = mach(-L)
Tin = 1 / (1 + 0.2 * Min ** 2)
rin, pin, uin = Tin ** 2.5, Tin ** 3.5 / G, Min * np.sqrt(Tin)

if mode == "--steady":
    integ = f"""[solver-time-integrator]
formulation = dual
scheme = backward-euler
pseudo-scheme = rk45
controller = none
pseudo-controller = local-pi
tstart = 0.0
tend = 1.0e8
dt = 1.0e8
pseudo-dt = 0.001
pseudo-niters-min = 1
pseudo-niters-max = {int(sys.argv[6]) if len(sys.argv) > 6 else 4000}
pseudo-resid-norm = l2
pseudo-resid-tol = 1e-10
atol = 1e-4
pseudo-dt-max-mult = 2.5

[solver-dual-time-integrator-multip]
pseudo-dt-fact = 1.75
cycle = {[(k, 1) for k in range(order, 0, -1)] + [(0, 2)] + [(k, 1) for k in range(1, order + 1)]}

[soln-plugin-pseudostats]
flushsteps = 1
file = {out}-residual.csv
header = true
"""
else:
    integ = f"""[solver-time-integrator]
formulation = std
scheme = rk4
controller = none
tstart = 0.0
tend = {sys.argv[6] if len(sys.argv) > 6 else 0.1}
dt = {sys.argv[7] if len(sys.argv) > 7 else 1e-4}
"""
open(out + ".ini", "w").write(f"""[backend]
precision = double
rank-allocator = linear

[backend-cuda]
device-id = 0
mpi-type = standard

[constants]
gamma = {G}

[solver]
system = euler
order = {order}

{integ}
[solver-interfaces]
riemann-solver = rusanov

[solver-interfaces-line]
flux-pts = gauss-legendre

[solver-elements-quad]
soln-pts = gauss-legendre

[soln-ics]
rho = {rho}
u = {u}
v = {v}
p = {p}

[soln-bcs-inlet]
type = char-riem-inv
rho = {rin:.15e}
u = {uin:.15e}
v = 0.0
p = {pin:.15e}

[soln-bcs-outlet]
type = sup-out-fn

[soln-bcs-wall]
type = slp-adia-wall

[soln-plugin-integrate]
nsteps = 1000000
file = {out}-integral.csv
header = true
quad-deg = 12
int-rhou = rho*u
int-entropy2 = pow(p/pow(rho, {G})*{G} - 1, 2)
""")
print("wrote", out + ".msh", out + ".ini", "elements", nx * ny, "DOF", nx * ny * (order + 1) ** 2)
