import textwrap

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator, StrMethodFormatter

SHOW_TITLES = False

SINGLE_COLUMN = 3.5
DOUBLE_COLUMN = 7.16
LINE_HEIGHT = 2.8

WIDE_MARGINS = {"left": 0.62, "right": 0.62, "bottom": 0.42, "top": 0.16}
TITLE_SPACE = 0.3

BAR_THICKNESS = 0.62
BAR_ROW_HEIGHT = 0.3

INK = "#1f1f1f"
MUTED = "#5c5c5c"
GRID = "#e5e5e5"

SOURCE_STYLE = {
    "steam_tracker": {"label": "Steam Tracker", "color": "#2a78d6", "marker": "o"},
    "delisted_games": {"label": "Delisted Games", "color": "#eb6834", "marker": "s"},
    "dead_game_list": {"label": "Dead Game List", "color": "#199e70", "marker": "^"},
    "doesitplay": {"label": "DoesItPlay", "color": "#4a3aa7", "marker": "D"},
}

CAUSE_STYLE = {
    "licensing": {"label": "Licensing", "color": "#2a78d6", "marker": "o"},
    "ip_change": {"label": "IP change", "color": "#eb6834", "marker": "s"},
    "server": {"label": "Service shut down", "color": "#1baf7a", "marker": "^"},
    "store_closure": {"label": "Store closure", "color": "#eda100", "marker": "D"},
    "publisher_exit": {"label": "Publisher exit", "color": "#e87ba4", "marker": "v"},
    "other": {"label": "Other reason", "color": "#8a8a8a", "marker": "x"},
    "not_stated": {"label": "Not stated", "color": "#b5b5b5", "marker": "+"},
}

TOTAL_STYLE = {"color": INK, "marker": "o"}
IGDB_STYLE = {"color": "#8a8a8a", "marker": "s", "linestyle": (0, (4, 2))}


def apply_style():
    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.labelsize": 8.5,
        "axes.labelpad": 4,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.major.size": 3,
        "xtick.major.width": 0.6,
        "xtick.minor.size": 1.5,
        "xtick.minor.width": 0.5,
        "ytick.major.size": 0,
        "ytick.major.pad": 4,
        "lines.linewidth": 1.4,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "legend.handlelength": 2.2,
        "legend.borderaxespad": 0.2,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": None,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def source_label(source: str, wrap: int = None) -> str:
    label = SOURCE_STYLE.get(source, {}).get("label", source)
    return "\n".join(textwrap.wrap(label, wrap)) if wrap else label


def set_title(ax, title: str):
    if not SHOW_TITLES:
        return
    width = int(ax.figure.get_figwidth() * 13)
    ax.set_title("\n".join(textwrap.wrap(title, width)))


def wide_figure():
    fig = plt.figure(figsize=(DOUBLE_COLUMN, LINE_HEIGHT))
    m = WIDE_MARGINS
    top = m["top"] + (TITLE_SPACE if SHOW_TITLES else 0)
    w, h = DOUBLE_COLUMN, LINE_HEIGHT
    ax = fig.add_axes([
        m["left"] / w,
        m["bottom"] / h,
        1 - (m["left"] + m["right"]) / w,
        1 - (m["bottom"] + top) / h,
    ])
    return fig, ax


def style_count_axis(ax):
    locator = MaxNLocator(nbins=5, integer=True)
    top = ax.dataLim.y1
    ax.set_ylim(0, max(t for t in locator.tick_values(0, top) if t >= top))
    ax.yaxis.set_major_locator(locator)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))


def style_year_axis(ax, years):
    ax.set_xlim(min(years) - 0.5, max(years) + 0.5)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))


def plot_bars(labels, values, colors, title, value_label=None, value_format="{:,}".format):
    rows = sorted(
        zip(labels, values, colors if isinstance(colors, list) else [colors] * len(labels)),
        key=lambda row: row[1],
    )
    labels, values, colors = zip(*rows)

    height = 0.55 + BAR_ROW_HEIGHT * len(labels) + (TITLE_SPACE if SHOW_TITLES else 0)
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN, height))
    ax.barh(labels, values, height=BAR_THICKNESS, color=colors, zorder=2)

    largest = max(values)
    for i, value in enumerate(values):
        ax.text(
            value + largest * 0.015, i, value_format(value),
            va="center", ha="left",
            fontsize=plt.rcParams["xtick.labelsize"], color=INK,
        )

    ax.set_xlim(0, largest * 1.14)
    ax.set_xticks([])
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.tick_params(axis="y", length=0, pad=5)
    if value_label:
        ax.set_xlabel(value_label)
    set_title(ax, title)
    fig.tight_layout(pad=0.3)
    return fig, ax


def plot_source_lines(pivot, title, xlabel, ylabel, year_range):
    fig, ax = wide_figure()
    years = pivot.index.astype(int)
    ordered = [s for s in SOURCE_STYLE if s in pivot.columns]
    ordered += [s for s in pivot.columns if s not in SOURCE_STYLE]
    for source in ordered:
        style = SOURCE_STYLE.get(source, {})
        ax.plot(
            years,
            pivot[source],
            color=style.get("color"),
            marker=style.get("marker", "o"),
            markeredgecolor="white",
            markeredgewidth=0.4,
            label=source_label(source),
            clip_on=False,
            zorder=3,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    style_year_axis(ax, year_range)
    style_count_axis(ax)
    ax.legend(loc="upper left")
    set_title(ax, title)
    return fig, ax
