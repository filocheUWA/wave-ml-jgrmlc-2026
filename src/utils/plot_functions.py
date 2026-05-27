"""General plotting helpers for exploratory spectral diagnostics."""

import numpy as np
import matplotlib.pyplot as plt

def plot_2Dspec_imshow(
    X, freqs, dirs=(180/np.pi)*np.arange(5, 361, 10), show=False, n_yticks=6
):
    """Display a directional spectrum on frequency-direction axes."""

    X = X.numpy() if hasattr(X, "numpy") else np.asarray(X)
    dirs = np.asarray(dirs)
    freqs = np.asarray(freqs)

    # Wrap direction for circular continuity
    X = np.vstack([X, X[0:1]])
    dirs = np.append(dirs, dirs[0] + 360 if dirs[0] < dirs[-1] else dirs[0] - 360)

    # Frequency bin edges define the log-scaled image extent.
    log_edges = np.zeros(len(freqs) + 1)
    log_edges[1:-1] = np.sqrt(freqs[:-1] * freqs[1:])
    log_edges[0] = freqs[0] - (log_edges[1] - freqs[0])
    log_edges[-1] = freqs[-1] + (freqs[-1] - log_edges[-2])

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(
        X.T, aspect='auto', origin='lower',
        extent=[dirs[0], dirs[-1], log_edges[0], log_edges[-1]],
        cmap='jet'
    )

    ax.set_yscale('log')

    # Choose a small number of ticks, always including first and last
    tick_idxs = np.linspace(0, len(freqs) - 1, n_yticks, dtype=int)
    yticks = freqs[tick_idxs]
    ax.set_yticks(yticks)
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.tick_params(axis='y', which='major', length=6, labelsize=12)
    ax.tick_params(axis='y', which='minor', length=3)

    # Direction ticks are shown in degrees using the provided convention.
    ax.set_xticks(np.linspace(dirs[0], dirs[-1], num=7))
    ax.set_xlabel('Direction (Degrees)', fontsize=14)
    ax.set_ylabel('Frequency (Hz)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=12)

    if show:
        plt.show()
    else:
        return fig, ax


def plot_spectra(freqs, spectra_list, labels, ax=None, 
                 vertical_lines=True, desired_xticks=None, 
                 colors=None, xlabel='Frequency (Hz)', ylabel='Energy Density', 
                 title='Energy density $E(f)$', ylim=(0, 7)):
    """
    Plot multiple 1D spectra on the same frequency grid.
    
    Parameters
    ----------
    freqs : 1D array
        Frequency grid (Hz)
    spectra_list : list of 1D arrays
        Each array is the spectrum to plot, same shape as freqs
    labels : list of str
        Legend labels for each spectrum
    ax : matplotlib axis, optional
        If None, a new figure and axis are created
    vertical_lines : bool, default True
        Draw vertical lines at frequency grid points.
    desired_xticks : list, optional
        Custom xtick positions (log scale recommended)
    colors : list, optional
        List of colors (default: matplotlib cycle)
    xlabel, ylabel, title : str
        Plot labels
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, spectrum in enumerate(spectra_list):
        ax.plot(freqs, spectrum, label=labels[i], color=colors[i % len(colors)])
    
    if vertical_lines:
        for fm in freqs:
            ax.axvline(fm, color='blue', linewidth=0.5, alpha=0.3, zorder=0)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale('log')
    if desired_xticks is not None:
        ax.set_xticks(desired_xticks)
        ax.set_xticklabels([str(x) for x in desired_xticks])
    ax.legend(loc='upper right')
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.set_ylim(*ylim)
    return ax
