import matplotlib as mpl
import matplotlib.pyplot as plt

def apply_journal_style(
    base_font="Arial",    
    axis_label=9, tick=8, legend=8, title=10,
    line_width=1.0, marker_size=3.0,
    save_dpi=300
):
    # Fonts & Embedding (PDF/EPS not converted to curves)
    mpl.rcParams["font.family"] = base_font
    mpl.rcParams["mathtext.default"] = "regular"  # Mathematical text uses regular font, easy to match with Sans
    mpl.rcParams["pdf.fonttype"] = 42             # 42: TrueType, optional 42/3
    mpl.rcParams["ps.fonttype"]  = 42

    # Size
    mpl.rcParams["axes.titlesize"]   = title
    mpl.rcParams["axes.labelsize"]   = axis_label
    mpl.rcParams["xtick.labelsize"]  = tick
    mpl.rcParams["ytick.labelsize"]  = tick
    mpl.rcParams["legend.fontsize"]  = legend

    # Curve and marker
    mpl.rcParams["lines.linewidth"]  = line_width
    mpl.rcParams["lines.markersize"] = marker_size
    mpl.rcParams["grid.linewidth"]   = 0.5

    # Axis and ticks
    mpl.rcParams["axes.linewidth"]   = 0.8
    mpl.rcParams["xtick.direction"]  = "out"
    mpl.rcParams["ytick.direction"]  = "out"
    mpl.rcParams["xtick.major.size"] = 3
    mpl.rcParams["ytick.major.size"] = 3

    # Color blindness friendly color scheme (blue orange + extension)
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=[
        "#377eb8", "#ff7f00", "#4daf4a", "#984ea3",
        "#e41a1c", "#a65628", "#f781bf", "#999999"
    ])

    # Save
    mpl.rcParams["savefig.dpi"]      = save_dpi
    mpl.rcParams["savefig.bbox"]     = "tight"
    mpl.rcParams["savefig.pad_inches"]= 0.02
    mpl.rcParams["figure.dpi"]       = save_dpi

