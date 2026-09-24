import numpy as np
import pytest

from segregation_amplification.aspatial import masking_share, masking_share_grid
from segregation_amplification.model import SegregationMaskingModel

SHAPE = (40, 40)


def run(seed, n_iterations=10, **kwargs):
    model = SegregationMaskingModel(SHAPE, seed=seed, **kwargs)
    model.update(n_iterations)
    return model


def same_type_share(model):
    """Mean share of same-type agents among each agent's occupied 8-neighbors (torus)."""
    types = model.types_map
    same = np.zeros(types.shape)
    occupied = np.zeros(types.shape)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            neighbor = np.roll(np.roll(types, dr, axis=0), dc, axis=1)
            same += (neighbor == types) & (neighbor > 0)
            occupied += neighbor > 0
    agents = (types > 0) & (occupied > 0)
    return float(np.mean(same[agents] / occupied[agents]))


def test_same_seed_reproduces_run():
    a, b = run(seed=3), run(seed=3)
    for name in ['types_map', 'masking_choice', 'infection_status', 'agent_locations']:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_different_seeds_differ():
    assert not np.array_equal(run(seed=3).types_map, run(seed=4).types_map)


def test_agents_are_conserved():
    model = SegregationMaskingModel(SHAPE, seed=0)
    n_blue, n_red = np.sum(model.types_map == 1), np.sum(model.types_map == 2)
    model.update(20)
    assert np.sum(model.types_map > 0) == model.n_agents
    assert np.sum(model.types_map == 1) == n_blue
    assert np.sum(model.types_map == 2) == n_red
    # Every occupied cell has a masking choice and every vacant cell has none.
    assert set(np.unique(model.masking_choice[model.types_map > 0])) <= {-1., 1.}
    assert np.all(model.masking_choice[model.types_map == 0] == 0)


def test_segregation_emerges():
    model = SegregationMaskingModel(SHAPE, seed=0)
    initial = same_type_share(model)
    model.update(50)
    assert initial == pytest.approx(.5, abs=.05)
    assert same_type_share(model) > initial + .2


def test_type_bias_splits_masking_by_type():
    # With the default parameters (r > 0, b < 0), segregated Blue agents mask and Red agents do not.
    model = run(seed=0, n_iterations=50)
    blue_rate = np.mean(model.masking_choice[model.types_map == 1] == 1)
    red_rate = np.mean(model.masking_choice[model.types_map == 2] == 1)
    assert blue_rate > .9
    assert red_rate < .5


@pytest.mark.parametrize('d_mean, r_mean, expected', [(.5, 2., 100.), (-1.5, -1., 0.), (-1., .5, 0.), (-.2, 1., 100.)])
def test_aspatial_game_matches_analytic_threshold(d_mean, r_mean, expected):
    # With negligible heterogeneity every agent masks iff d + s * r > 0 (s = .5).
    rng = np.random.default_rng(0)
    share = masking_share(rng, d_mean, r_mean, d_std=1e-6, r_std=1e-6, s_std=1e-6)
    assert share == expected


def test_aspatial_grid_is_monotone_in_both_means():
    grid, _, _ = masking_share_grid(seed=0, n_sample_points=8, d_std=1e-6, r_std=1e-6, s_std=1e-6)
    assert np.all(np.diff(grid, axis=0) >= 0)
    assert np.all(np.diff(grid, axis=1) >= 0)
