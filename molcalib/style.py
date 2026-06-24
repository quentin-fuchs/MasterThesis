"""Publication figure style for MProject notebooks.

Provides the Okabe-Ito color palette, font-size constants, and a setup()
helper that applies the 'thesis' matplotlib style sheet.

Usage
-----
    import matplotlib.pyplot as plt
    from molcalib.style import setup, C, FS

    setup()   # call once per notebook, before any plotting

    ax.bar(x, y, color=C['green'])
    ax.set_ylabel('...', fontsize=FS['label'])

Colors (Okabe-Ito, colorblind-safe)
------------------------------------
    C['green']   #009E73  — Cat A / generative success / DiffDock-L Native
    C['orange']  #E69F00  — Cat B / intermediate / DiffDock-L re-ranking
    C['blue']    #0072B2  — Cat C / fundamental failure
    C['purple']  #CC79A7  — SigmaDock / distinct architecture
    C['black']   #000000  — annotations, axes, oracle lines
    C['grey']    #888888  — secondary oracle lines / muted elements
    C['sky']     #56B4E9  — sky blue (extra slot)
    C['yellow']  #F0E442  — yellow (extra slot)

Font sizes
----------
    FS['label']   12   axis labels (xlabel / ylabel)
    FS['tick']    11   tick labels
    FS['ann']     10   in-panel annotations
    FS['letter']  14   subfigure letters (A), (B)
    FS['legend']   9.5 legend text
"""

import matplotlib.pyplot as plt

# ── Okabe-Ito palette ─────────────────────────────────────────────────────────
C = {
    'green':  '#009E73',
    'orange': '#E69F00',
    'blue':   '#0072B2',
    'purple': '#CC79A7',
    'black':  '#000000',
    'grey':   '#888888',
    'sky':    '#56B4E9',
    'yellow': '#F0E442',
}

# ── Font sizes ────────────────────────────────────────────────────────────────
FS = {
    'label':  12,
    'tick':   11,
    'ann':    10,
    'letter': 14,
    'legend': 9.5,
}


def setup() -> None:
    """Apply the 'thesis' matplotlib style sheet and patch the color cycle."""
    plt.style.use('thesis')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(
        color=[C['green'], C['orange'], C['blue'], C['purple'],
               C['sky'], C['yellow'], C['grey'], C['black']]
    )


def subfig_label(ax, letter: str, *, x: float = 0.03, y: float = 0.98,
                 ha: str = 'left') -> None:
    """Add a bold subfigure letter to *ax* in axes coordinates.

    Use ha='right' and x=0.97 when a legend occupies the upper-left corner.
    """
    ax.text(x, y, f'({letter})', transform=ax.transAxes,
            fontsize=FS['letter'], fontweight='bold',
            va='top', ha=ha, color=C['black'])
