from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from PIL import Image

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

__all__ = ('visualize_neighbors',)


def add_text(text: str, *, ax: Axes) -> Axes:
    ax.text(.5, .1, text,
            c='w',
            ha='center',
            va='center',
            transform=ax.transAxes,
            path_effects=[
                pe.withStroke(
                    linewidth=4,
                    foreground='k'
                )
            ]
            )
    return ax


def visualize_neighbors(
        query: str,
        neighbors: list[str],
        *,
        labels: list[str] = None,
        ncols: int = 5
) -> Figure:
    # Resolves the image labeling.
    labels = labels or [f'{index:d}' + ('st' if index == 1 else
                                        'nd' if index == 2 else
                                        'rd' if index == 3 else 'th')
                        for index, _ in enumerate(neighbors, start=1)]

    # Prepares to plot the images.
    num_items = len(neighbors)
    ncols = min(ncols, num_items)
    num_rests = num_items % ncols
    nrows = num_items // ncols + 1 + (num_rests > 0)

    # Prepares the canvas.
    fig = plt.figure(
        figsize=(2 * ncols, 2 * nrows),
        dpi=128,
        tight_layout=True
    )
    grid = GridSpec(
        nrows=nrows,
        ncols=ncols * 2,
        figure=fig
    )

    # Plots the query image.
    idx = 2 * ncols // 2 - 1
    ax = fig.add_subplot(grid[0, idx:idx + 2])
    img = Image.open(query)
    ax.imshow(img)
    ax.axis('off')
    add_text(text='Query', ax=ax)

    # Plots the nearest neighbors.
    for idx, (label, path) in enumerate(zip(labels, neighbors)):
        # Builds the grid-spec index.
        icol = idx % ncols * 2
        irow = idx // ncols + 1
        # Adjusts for possibly unfilled last row.
        if num_rests != 0 and irow == nrows - 1:
            icol += ncols - num_rests

        # Loads the image.
        img = Image.open(path)
        # Plots the image.
        ax = fig.add_subplot(grid[irow, icol:icol + 2])
        ax.imshow(img)
        ax.axis('off')
        # Annotates the image.
        add_text(text=label, ax=ax)

    # Returns the finalized figure.
    return fig
