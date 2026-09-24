"""Aspatial reciprocal externality game: every agent is in every other agent's neighborhood, so space plays no role."""
import numpy as np
from scipy import stats


def truncated_normal(rng, mean, std, low, high, size):
    return stats.truncnorm.rvs((low - mean) / std, (high - mean) / std, loc=mean, scale=std, size=size, random_state=rng)


def masking_share(rng, d_mean, r_mean, n_agents=50, d_std=.25, r_std=.25, s_mean=.5, s_std=.1):
    """Percent of agents who mask for one draw of agents with the given parameter means.

    Each agent masks if the utility from masking, d_i + s_i * mean(r_j for j != i), exceeds the utility from not masking (0).
    """
    d = truncated_normal(rng, d_mean, d_std, -10, 10, n_agents)
    s = truncated_normal(rng, s_mean, s_std, 0, 1, n_agents)
    r = truncated_normal(rng, r_mean, r_std, -10, 10, n_agents)

    average_reciprocal_response = (r.sum() - r) / (n_agents - 1)
    utility_from_cooperating = d + s * average_reciprocal_response
    utility_from_defecting = 0.
    return float(np.mean(utility_from_cooperating > utility_from_defecting) * 100)


def masking_share_grid(seed=None, n_sample_points=20, d_range=(-1.5, .5), r_range=(-1., 2.), **kwargs):
    """Percent masking over a grid of mean direct utility (d) and mean reciprocal response (r).

    Returns (grid, d_means, r_means) where grid[i, j] is the masking share at r_means[i], d_means[j].
    """
    rng = np.random.default_rng(seed)
    d_means = np.linspace(*d_range, n_sample_points)
    r_means = np.linspace(*r_range, n_sample_points)
    grid = np.zeros((n_sample_points, n_sample_points))
    for i, r_mean in enumerate(r_means):
        for j, d_mean in enumerate(d_means):
            grid[i, j] = masking_share(rng, d_mean, r_mean, **kwargs)
    return grid, d_means, r_means
