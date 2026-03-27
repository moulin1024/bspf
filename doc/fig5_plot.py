import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load solution snapshots
solution_file = 'data/solution_nx1400_u.npy'
time_file = 'data/solution_nx1400_t.npy'
x_file = 'data/solution_nx1400_x.npy'

# Load exact solution
exact_file = 'data/solution_nx1400_exact_u.npy'

# Load data first to get correct dimensions
U = np.load(solution_file)  # Shape: (n_snapshots, nx)
U_exact = np.load(exact_file)  # Shape: (n_snapshots, nx)
t = np.load(time_file)       # Shape: (n_snapshots,)
x = np.load(x_file)          # Shape: (nx,)

# Get actual dimensions from loaded data
nt, nx = U.shape

# Verify exact solution has same dimensions
if U_exact.shape != U.shape:
    raise ValueError(f"Exact solution shape {U_exact.shape} does not match numerical solution shape {U.shape}")

# Subsample data to reduce memory usage for plotting
# Take every nth time step and spatial point
time_stride =  max(1, nt // 100)  # Limit to ~100 time points
space_stride = max(1, nx // 100)  # Limit to ~100 spatial points

U_sub = U[::time_stride, ::space_stride]
U_exact_sub = U_exact[::time_stride, ::space_stride]
t_sub = t[::time_stride]
x_sub = x[::space_stride]

print(f"Original shape: ({nt}, {nx})")
print(f"Subsampled shape: ({len(t_sub)}, {len(x_sub)})")

# Compute error using loaded exact solution
err_sub = np.abs(U_sub - U_exact_sub)/3.0

# Create meshgrid for 3D plotting
XX, TT = np.meshgrid(x_sub, t_sub)

# Create subplot figure (1 row, 2 columns)
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "surface"}, {"type": "surface"}]],
    subplot_titles=["<b>(a)</b>", "<b>(b)</b>"],
)

# Update the subplot title font size
for ann in fig['layout']['annotations']:
    ann['font'] = dict(size=32, color="black")
    ann['xanchor'] = "left"
    ann['x'] -= 0.2
    ann['yanchor'] = "top"
    ann['align'] = "left"

# ---- First surface: U ----
fig.add_trace(
    go.Surface(
        x=XX, y=TT, z=U_sub,
        colorscale="Blues",
        reversescale=True,
        cmin=0,
        cmax=U_sub.max(),
        colorbar=dict(
            title=dict(text="u", font=dict(size=24)),
            tickfont=dict(size=24),
            orientation="v",
            x=0.45, xanchor="left",
            y=0.5, len=1
        ),
        showscale=True
    ),
    row=1, col=1
)

# ---- Second surface: Error ----
fig.add_trace(
    go.Surface(
        x=XX, y=TT, z=err_sub,
        colorscale="Reds",
        reversescale=True,
        cmin=0,
        cmax=np.max(err_sub),
        colorbar=dict(
            title=dict(text="|Error|", font=dict(size=24)),
            tickfont=dict(size=24),
            orientation="v",
            x=1, xanchor="left",
            y=0.5, len=1,
            exponentformat="power",
            showexponent="all"
        ),
        showscale=True
    ),
    row=1, col=2
)

# Axis labels
fig.update_scenes(
    xaxis_title="x",
    yaxis_title="t",
    zaxis_title="u",
    xaxis=dict(title=dict(font=dict(size=28)), tickfont=dict(size=16)),
    yaxis=dict(title=dict(font=dict(size=28)), tickfont=dict(size=16)),
    zaxis=dict(title=dict(font=dict(size=28)), tickfont=dict(size=16)),
    row=1, col=1
)
fig.update_scenes(
    xaxis_title="x",
    yaxis_title="t",
    zaxis_title="|Error|",
    xaxis=dict(title=dict(font=dict(size=28)), tickfont=dict(size=16)),
    yaxis=dict(title=dict(font=dict(size=28)), tickfont=dict(size=16)),
    zaxis=dict(title=dict(font=dict(size=28)), tickfont=dict(size=16), exponentformat="power", showexponent="last"),
    row=1, col=2
)

# Layout tweaks
fig.update_layout(
    height=500, width=1300,
    margin=dict(l=2, r=2, t=5, b=5),
    autosize=True
)
fig.update_xaxes(automargin=True)
fig.update_yaxes(automargin=True)

# Try to save as PDF (may fail if data is still too large)
try:
    fig.write_image("figs/fig5.pdf", width=1700, height=600, scale=2)  # Reduced scale from 4 to 2
    print("Saved Plotly surface plot to figs/fig5.pdf")
except Exception as e:
    print(f"Could not save as PDF: {e}")
    print("Saving as HTML instead...")
    fig.write_html("figs/fig5.html")
    print("Saved Plotly surface plot to figs/fig5.html")
