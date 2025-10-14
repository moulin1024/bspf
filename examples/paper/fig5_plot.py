import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 2*np.pi, 1000)
t = np.linspace(0, 2.0, 1000)
U = np.load('./data/U_nu0.01.npy')
Ue = np.load('./data/u_exact_nu0.01.npy')

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
XX,TT = np.meshgrid(x,t)
err = np.abs(U-Ue)
# Create subplot figure (1 row, 2 columns)
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "surface"}, {"type": "surface"}]],
    subplot_titles=["<b>(a)</b>", "<b>(b)</b>"],
)

# Update the subplot title font size
for ann in fig['layout']['annotations']:
    ann['font'] = dict(size=24, color="black")  # change 24 to any size you like
    ann['xanchor'] = "left"   # anchor text to its left edge
    ann['x'] -= 0.2           # shift further left (tweak the value)
    ann['yanchor'] = "top"    # keep it at the top
    ann['align'] = "left"     # align text left

# ---- First surface: U ----
fig.add_trace(
    go.Surface(
        x=XX, y=TT, z=U,
        colorscale="Blues",
        reversescale=True,        # same as Blues_r
        cmin=0.2,
        cmax=1,
        colorbar=dict(
            title=dict(text="U", font=dict(size=20)),  # title font
        tickfont=dict(size=16),
            orientation="v",
            x=0.45, xanchor="left",  # place under left subplot
            y=0.5, len=1           # shift and size
        ),
        showscale=True
    ),
    row=1, col=1
)

# ---- Second surface: Error ----
fig.add_trace(
    go.Surface(
        x=XX, y=TT, z=err,
        colorscale="Blues",
        reversescale=True,
        cmin=0,
        cmax=err.max(),
        colorbar=dict(
            title=dict(text="|Error|", font=dict(size=20)),  # title font
        tickfont=dict(size=16)  ,
            orientation="v",
            x=1, xanchor="left",  # place under right subplot
            y=0.5, len=1, exponentformat="power",   # use 10^x form
        showexponent="all"        # show exponent once on axis
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
    xaxis=dict(title=dict(font=dict(size=20)), tickfont=dict(size=16)),
    yaxis=dict(title=dict(font=dict(size=20)), tickfont=dict(size=16)),
    zaxis=dict(title=dict(font=dict(size=20)), tickfont=dict(size=16)),
    row=1, col=1
)
fig.update_scenes(
    xaxis_title="x",
    yaxis_title="t",
    zaxis_title="|Error|",
    xaxis=dict(title=dict(font=dict(size=20)), tickfont=dict(size=16)),
    yaxis=dict(title=dict(font=dict(size=20)), tickfont=dict(size=16)),
    zaxis=dict(title=dict(font=dict(size=20)), tickfont=dict(size=16),exponentformat="power",showexponent="last"),
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
fig.write_image("fig5.pdf", width=1600, height=600, scale=4)
