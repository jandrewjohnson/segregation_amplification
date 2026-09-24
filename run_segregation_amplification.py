"""Optional ProjectFlow entry point: build the paper figures, then open the interactive simulation.

This is an alternative to `python make_figures.py` and `python -m segregation_amplification.interactive` for readers who use
hazelbean's ProjectFlow. It requires hazelbean >= 2.0, which is not a dependency of the model itself:

    pip install -e ".[projectflow]"
    python run_segregation_amplification.py

Outputs go to a project directory beside (never inside) this repo: <repo_parent>/projects/segregation_amplification/.

    Configuration: level 2 of 4 -- a task tree, constants still inline
    Code layout:   library package -- tasks call into segregation_amplification
    Ownership:     self-contained -- you own every file this run touches
"""
import os

import hazelbean as hb

from segregation_amplification import figures
from segregation_amplification.interactive import InteractiveView
from segregation_amplification.model import SegregationMaskingModel


def build_task_tree(p):
    p.paper_figures_task = p.add_task(paper_figures)
    p.interactive_simulation_task = p.add_task(interactive_simulation)


def paper_figures(p):
    """Figures 2-6 of the paper, each as a PNG plus a JSON record of its seed and parameters."""
    p.figure_paths = {number: os.path.join(p.cur_dir, name + '.png') for number, name in figures.FIGURE_NAMES.items()}

    if p.run_this:
        for number, path in p.figure_paths.items():
            if not hb.path_exists(path):
                figures.FIGURES[number](p.cur_dir, seed=p.seed)
                hb.log('Wrote ' + path)


def interactive_simulation(p):
    """Open the live viewer on a fresh model. Blocks until the window is closed."""
    if p.run_this:
        model = SegregationMaskingModel(p.world_shape, seed=p.seed)
        InteractiveView(model)


def run_project(p):
    """Execute the pipeline against the ProjectFlow the caller configured.

    Reads p.world_shape, p.seed, and optionally p.tasks_to_skip. Returns p.
    """
    build_task_tree(p)
    p.skip_tasks(p.tasks_to_skip)

    p.L = hb.get_logger(p.project_name)
    hb.log('Created ProjectFlow object at ' + p.project_dir)

    p.execute()

    return p


if __name__ == '__main__':
    # run_mode: 'check' resumes in place (existing figures are kept) | 'full' timestamps a new dir.
    p = hb.ProjectFlow(project_name='segregation_amplification', run_mode='check')

    p.world_shape = (150, 150)  # Grid for the interactive simulation.
    p.seed = figures.DEFAULT_SEED  # Seeds both the figures and the interactive model; None gives a new run each time.

    # To open only the interactive simulation: p.tasks_to_skip = ['paper_figures']

    run_project(p)
