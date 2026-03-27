import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load data first to get correct dimensions
U = np.load('./data/U_nu0.01.npy')
Ue = np.load('./data/u_exact_nu0.01.npy')

# Get actual dimensions from loaded data
nt, nx = U.shape
L = 2.0 * np.pi
T = 2.0

# Downsample to 101x101 for plotting
target_nt = 201
target_nx = 201

# Calculate stride for downsampling
time_stride = max(1, nt // target_nt)
space_stride = max(1, nx // target_nx)

# Downsample data
U_sub = U[::time_stride, ::space_stride]
Ue_sub = Ue[::time_stride, ::space_stride]

# Get actual dimensions after downsampling
nt_sub, nx_sub = U_sub.shape

# Create grids matching the downsampled data dimensions
x = np.linspace(0, L, nx_sub)
t = np.linspace(0, T, nt_sub)

# Create meshgrid for 3D plotting
XX, TT = np.meshgrid(x, t)
err = np.abs(U_sub - Ue_sub)
# Create subplot figure (1 row, 2 columns)
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "surface"}, {"type": "surface"}]],
    subplot_titles=["<b>(a)</b>", "<b>(b)</b>"],
)

# Update the subplot title font size
for ann in fig['layout']['annotations']:
    ann['font'] = dict(size=28, color="black")  # change 24 to any size you like
    ann['xanchor'] = "left"   # anchor text to its left edge
    ann['x'] -= 0.2           # shift further left (tweak the value)
    ann['yanchor'] = "top"    # keep it at the top
    ann['align'] = "left"     # align text left

# ---- First surface: U ----
fig.add_trace(
    go.Surface(
        x=XX, y=TT, z=U_sub,
        colorscale="Blues",
        reversescale=True,        # same as Blues_r
        cmin=0.2,
        cmax=1,
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
        x=XX, y=TT, z=err,
        colorscale="Reds",
        reversescale=True,
        cmin=0,
        cmax=3e-11,
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
    zaxis=dict(title=dict(font=dict(size=28)), tickfont=dict(size=16),exponentformat="power",showexponent="last"),
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
fig.write_image("figs/fig3.pdf", width=1700, height=600, scale=2)
