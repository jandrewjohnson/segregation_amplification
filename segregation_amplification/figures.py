"""Scripted, seeded versions of the paper's simulation figures (Figures 2-6).

Each function builds one figure, saves it as a PNG, and writes a JSON sidecar recording the seed and every parameter used.
"""
import json
import os
import platform

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import scipy

from segregation_amplification import __version__
from segregation_amplification.aspatial import masking_share_grid
from segregation_amplification.model import SegregationMaskingModel

DEFAULT_SEED = 1

# Output file stem for each paper figure.
FIGURE_NAMES = {
    '2': 'fig2_aspatial_masking',
    '3': 'fig3_segregation_over_time',
    '4': 'fig4_masking_over_time',
    '5': 'fig5_segregation_threshold',
    '6': 'fig6_spatial_adjacency',
}

# Political types (types_map values): 1 = Blue, 2 = Red. Masking (masking_choice values): 1 = mask, -1 = no mask. 0 = vacant in both.
VACANT_COLOR = '#ffffff'
TYPE_COLORS = {0: VACANT_COLOR, 1: '#2a78d6', 2: '#e34948'}
TYPE_LABELS = {1: 'Blue type', 0: 'Vacant', 2: 'Red type'}
MASK_COLORS = {-1: 'purple', 0: VACANT_COLOR, 1: 'green'}
MASK_LABELS = {1: 'Mask', 0: 'Vacant', -1: 'No mask'}
ASPATIAL_CMAP = 'Greens'


def _categorical_imshow(ax, array, colors):
    values = sorted(colors)
    cmap = ListedColormap([colors[v] for v in values])
    bounds = [v - .5 for v in values] + [values[-1] + .5]
    ax.imshow(array, cmap=cmap, norm=BoundaryNorm(bounds, cmap.N), interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#c3c2b7')


def _legend_handles(colors, labels):
    return [Patch(facecolor=colors[v], edgecolor='#52514e', linewidth=.5, label=labels[v]) for v in labels]


def _color_record(colors, labels):
    return {labels[v]: colors[v] for v in labels}


TYPE_COLOR_RECORD = _color_record(TYPE_COLORS, TYPE_LABELS)
MASK_COLOR_RECORD = _color_record(MASK_COLORS, MASK_LABELS)


def _iteration_label(ax, n):
    t = ax.text(.04, .96, str(n), transform=ax.transAxes, fontsize=7, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=.85, linewidth=0, pad=1.5))


def _save(fig, output_dir, name, metadata):
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, name + '.png')
    fig.savefig(png_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    metadata = dict(metadata)
    metadata['figure'] = name
    metadata['environment'] = {
        'segregation_amplification': __version__,
        'python': platform.python_version(),
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'matplotlib': matplotlib.__version__,
    }
    with open(os.path.join(output_dir, name + '.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    return png_path


def _model_metadata(model, **extra):
    metadata = {
        'seed': model.seed,
        'spatial_shape': list(model.spatial_shape),
        'proportion_filled': model.proportion_filled,
        'game_type': model.game_type,
        'params': model.params,
    }
    metadata.update(extra)
    return metadata


def figure_2_aspatial(output_dir, seed=DEFAULT_SEED):
    """Masking share in the aspatial game over a grid of mean direct utility and mean reciprocal response."""
    n_sample_points = 20
    n_agents = 50
    grid, d_means, r_means = masking_share_grid(seed=seed, n_sample_points=n_sample_points, n_agents=n_agents)

    fig, ax = plt.subplots(figsize=(5, 4.2))
    d_step = d_means[1] - d_means[0]
    r_step = r_means[1] - r_means[0]
    # Reciprocal response increases downward, matching the layout described in the paper (full masking in the lower right).
    extent = [d_means[0] - d_step / 2, d_means[-1] + d_step / 2, r_means[-1] + r_step / 2, r_means[0] - r_step / 2]
    im = ax.imshow(grid, cmap=ASPATIAL_CMAP, vmin=0, vmax=100, extent=extent, aspect='auto', interpolation='nearest')
    ax.set_xlabel(r'Mean direct utility from masking, $\mu_d$')
    ax.set_ylabel(r'Mean reciprocal response, $\mu_r$')
    ax.set_title('Masking compliance in the aspatial reciprocal externality game', fontsize=10)
    fig.colorbar(im, ax=ax, label='Percent masking')

    metadata = {
        'seed': seed,
        'n_sample_points': n_sample_points,
        'n_agents': n_agents,
        'd_means': d_means.tolist(),
        'r_means': r_means.tolist(),
        'std': {'d': .25, 'r': .25, 's': .1},
        's_mean': .5,
        'percent_masking': grid.tolist(),
        'colors': {'colormap': ASPATIAL_CMAP},
    }
    return _save(fig, output_dir, FIGURE_NAMES['2'], metadata)


def _time_series_figure(model, iterations_to_plot, array_name, colors, labels, n_rows=4, n_cols=5):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5 + 1.2, n_rows * 1.5))
    last_iteration = 0
    for ax, iteration in zip(axes.flat, iterations_to_plot):
        model.update(iteration - last_iteration)
        last_iteration = iteration
        _categorical_imshow(ax, np.copy(getattr(model, array_name)), colors)
        _iteration_label(ax, iteration)
    fig.subplots_adjust(left=0, right=.84, bottom=0, top=1, wspace=.03, hspace=.03)
    fig.legend(handles=_legend_handles(colors, labels), loc='center left', bbox_to_anchor=(.85, .5), frameon=False, fontsize=8)
    return fig


def figure_3_segregation_over_time(output_dir, seed=DEFAULT_SEED, spatial_shape=(150, 150)):
    """Agents sorting into segregated neighborhoods over iterations."""
    iterations_to_plot = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 19, 22, 25, 28, 35, 40, 50, 100]
    model = SegregationMaskingModel(spatial_shape, seed=seed)
    fig = _time_series_figure(model, iterations_to_plot, 'types_map', TYPE_COLORS, TYPE_LABELS)
    return _save(fig, output_dir, FIGURE_NAMES['3'], _model_metadata(model, iterations_plotted=iterations_to_plot, colors=TYPE_COLOR_RECORD))


def figure_4_masking_over_time(output_dir, seed=DEFAULT_SEED, spatial_shape=(150, 150)):
    """Masking choices over iterations as segregation emerges."""
    iterations_to_plot = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 30, 50, 75, 100]
    model = SegregationMaskingModel(spatial_shape, seed=seed)
    fig = _time_series_figure(model, iterations_to_plot, 'masking_choice', MASK_COLORS, MASK_LABELS)
    return _save(fig, output_dir, FIGURE_NAMES['4'], _model_metadata(model, iterations_plotted=iterations_to_plot, colors=MASK_COLOR_RECORD))


def figure_5_segregation_threshold(output_dir, seed=DEFAULT_SEED, spatial_shape=(100, 100), thresholds=(.11, .25, .46), n_iterations=100):
    """Equilibrium masking and political type for several segregation thresholds (tau)."""
    fig, axes = plt.subplots(2, len(thresholds), figsize=(len(thresholds) * 2.2 + 1.4, 5))
    masking_rates = {}
    for col, threshold in enumerate(thresholds):
        model = SegregationMaskingModel(spatial_shape, seed=seed, params={'segregation_threshold': threshold})
        # Advance one iteration per call, as in the interactive viewer.
        for _ in range(n_iterations):
            model.update(1)
        masking_rates[threshold] = model.get_masking_rate()

        _categorical_imshow(axes[0, col], model.masking_choice, MASK_COLORS)
        _categorical_imshow(axes[1, col], model.types_map, TYPE_COLORS)
        axes[0, col].set_title(r'$\tau$ = %s' % threshold, fontsize=10)
        axes[1, col].set_xlabel('Masking rate: %.1f%%' % (masking_rates[threshold] * 100), fontsize=9, style='italic')

    axes[0, 0].set_ylabel('Masking choice', fontsize=10)
    axes[1, 0].set_ylabel('Political type', fontsize=10)
    fig.subplots_adjust(right=.8, wspace=.05, hspace=.12)
    fig.legend(handles=_legend_handles(MASK_COLORS, MASK_LABELS), loc='center left', bbox_to_anchor=(.81, .7), frameon=False, fontsize=8)
    fig.legend(handles=_legend_handles(TYPE_COLORS, TYPE_LABELS), loc='center left', bbox_to_anchor=(.81, .3), frameon=False, fontsize=8)

    metadata = _model_metadata(model, thresholds=list(thresholds), n_iterations=n_iterations, iterations_per_update=1,
                               masking_rates={str(k): v for k, v in masking_rates.items()},
                               colors={'masking_choice': MASK_COLOR_RECORD, 'political_type': TYPE_COLOR_RECORD})
    metadata['params'] = {k: v for k, v in metadata['params'].items() if k != 'segregation_threshold'}
    return _save(fig, output_dir, FIGURE_NAMES['5'], metadata)


def figure_6_spatial_adjacency(output_dir, seed=DEFAULT_SEED, spatial_shape=(150, 150), n_iterations=40):
    """Masking choice beside political type for one run, showing behavior at the borders between Red and Blue zones."""
    model = SegregationMaskingModel(spatial_shape, seed=seed)
    model.update(n_iterations)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.3))
    _categorical_imshow(axes[0], model.masking_choice, MASK_COLORS)
    _categorical_imshow(axes[1], model.types_map, TYPE_COLORS)
    axes[0].set_title('Masking choice', fontsize=10)
    axes[1].set_title('Political type', fontsize=10)
    axes[0].legend(handles=_legend_handles(MASK_COLORS, MASK_LABELS), loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
    axes[1].legend(handles=_legend_handles(TYPE_COLORS, TYPE_LABELS), loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
    fig.subplots_adjust(wspace=.45)

    return _save(fig, output_dir, FIGURE_NAMES['6'], _model_metadata(model, n_iterations=n_iterations, masking_rate=model.get_masking_rate(),
                                                                              colors={'masking_choice': MASK_COLOR_RECORD, 'political_type': TYPE_COLOR_RECORD}))


FIGURES = {
    '2': figure_2_aspatial,
    '3': figure_3_segregation_over_time,
    '4': figure_4_masking_over_time,
    '5': figure_5_segregation_threshold,
    '6': figure_6_spatial_adjacency,
}
