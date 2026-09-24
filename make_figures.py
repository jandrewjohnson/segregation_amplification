"""Regenerate the paper's simulation figures (Figures 2-6).

Usage:
    python make_figures.py                   # all figures into figures/
    python make_figures.py --only 3 4        # a subset
    python make_figures.py --seed 7 --output-dir figures_seed7
"""
import argparse
import time

import matplotlib
matplotlib.use('Agg')  # Write files only; never open windows.

from segregation_amplification.figures import DEFAULT_SEED, FIGURES


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', default='figures', help='Where to write PNGs and their JSON parameter records (default: figures/).')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Random seed (default: %(default)s).')
    parser.add_argument('--only', nargs='+', choices=sorted(FIGURES), help='Figure numbers to build (default: all).')
    args = parser.parse_args()

    for number in args.only or sorted(FIGURES):
        start = time.time()
        path = FIGURES[number](args.output_dir, seed=args.seed)
        print('Figure %s -> %s (%.1fs)' % (number, path, time.time() - start))


if __name__ == '__main__':
    main()
