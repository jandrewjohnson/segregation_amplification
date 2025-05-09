import os
import sys
import random
import numpy as np
from numpy.core import multiarray
from numpy.core.defchararray import upper
import scipy as sp
from scipy.stats import truncnorm
import hazelbean as hb
import time

import tribal_masking_computational_core
import collections

class TribalMaskingModel(object):

    def __init__(self, spatial_shape=None):
        if spatial_shape is None:
            self.spatial_shape = (75, 150)
        else:
            self.spatial_shape = spatial_shape

        self.debug_level = 5
        self.proportion_filled = 0.75
        self.game_type = 'move_if_too_many_dissimilar'
        # self.game_type = 'move_if_not_enough_similar'


        self.initial_params = {
            'segregation_threshold': 0.55, # tau
            'neighborhood_radius': 3, #G
            'infection_probability': 0.005,
            'masking_efficacy': 0.55,
            'infection_duration': 10,
            'immunity_decay': .99,
            'd': -0.25,
            'dd': 0.0000001,
            's': 1.0,
            'sd': 0.0000001,
            'r': 3.0,
            'rd': 0.0000001,
            'b': -0.1,
            'bd': .0000001,
        }

        # amplitudes = {
        #                  Status.Susceptible: 10,
        #                  Status.Recovered_Immune: 10,
        #                  Status.Infected: 10
        #              },

        # Epidemiological
        critical_limit = 0.01,
        contagion_rate = .9,
        incubation_time = 5,
        contagion_time = 10,
        recovering_time = 20,

        # Uncertainty turned out not to matter much, so simplifying it away from the UI is clean. It will still use the default values however, just not plot changes to them.
        self.show_uncertainty_sliders = False

        # self.initial_params = {'d': [-10., 10., -.25, .0000001], 's': [0., 10., 1.0, .0000001], 'r': [-10., 10., 3.0, .0000001], 'b': [-10., 10., -.1, .0000001]}

        self.params = self.initial_params.copy()

        self.initialize_agents()

        self.initialize_choice_variables()

        self.initialize_reporting_variables()

    def initialize_agents(self):

        self.proportion_filled = self.proportion_filled

        self.n_rows = self.spatial_shape[0]
        self.n_cols = self.spatial_shape[1]

        # Save row and col ids as two new parameters so that we can quickly look up paraeters my spatial location OR agent id.
        self.row_ids = np.arange(self.n_rows)
        self.col_ids = np.arange(self.n_cols)

        # Number of agents is a function then of the size of the space and the proportion filled, rounded down and inted.
        self.n_cells = self.n_rows * self.n_cols
        self.n_agents = np.int32(np.floor(self.n_cells * self.proportion_filled))

        # Keep track of all the agents via IDs
        self.agent_ids = np.arange(0, self.n_agents).astype(np.int32)

        # positional_indices records the 2-length r, c of each agent grid-cell based on their r, c.
        self.positional_indices = np.empty((self.n_rows, self.n_cols, 2), dtype=np.int32)
        self.positional_indices[:, :, 0] = self.row_ids[:, None]
        self.positional_indices[:, :, 1] = self.col_ids

        # Positional indices are just the r, c coordinates ordered. Create a shuffled version to represent initial placement.
        # CONVENTIONS NOTE: Get means it doesn't cahnge the model state (and instead takes parameters and returns something.
        # This would enable non-static usage from outside the class)
        self.shuffled_positional_indices = self.get_shuffled_poistional_indices(self.positional_indices)

        # CONVENTIONS NOTE: Initilize means it changes the model state (and thus doesn't take args and returns nothing)
        self.initialize_agent_locations()

        self.initialize_agent_types()

        self.initialize_random_parameters()




    def get_shuffled_poistional_indices(self, positional_indices):
        # Somewhat convoluted way of randomizing the initial postitions.
        shuffled_positional_indices = np.copy(positional_indices)  # Operate on a copy to keep original

        # Shuffle requires a flattened array, so reshape it from 3d to 2d, keeping the 2d r-c indices.
        # Note also that shuffled_positional_indices is ALL LOCATIONS which is a superset of agent_locations, defined below.
        new_shape = (shuffled_positional_indices.shape[0] * shuffled_positional_indices.shape[1], shuffled_positional_indices.shape[2])
        shuffled_positional_indices = shuffled_positional_indices.reshape(new_shape)

        # Randomize the order by creating a randomized list of r-c indices for future fast iterating.
        # LEARNING POINT, this was the most efficient approach I could think of for shuffling in the first 2 of three directions while keeping the third intact.
        rng = np.random.default_rng()
        rng.shuffle(shuffled_positional_indices, axis=0)

        return shuffled_positional_indices

    def initialize_agent_locations(self):

        # First initialize all agents_list on a random location according to the initial shuffled queue.
        self.agent_locations = self.shuffled_positional_indices[:self.n_agents]

        # Also initialize the unoccupied locations. When agents move, the grab a new spot from this list and move their previous slot into this array.
        self.unoccupied_locations = self.shuffled_positional_indices[self.n_agents:]

        # DESIGN CHOICE: agent_locations and agent_types are 1_dim arrays, but agent_types is then made 2d into types_map. Is this redundant? Do I need only
        # 1 1-dim for performance and save everything as 2dim?

        # agent ids and types are initialized as maps.
        self.agent_ids_map = np.zeros((self.n_rows, self.n_cols), dtype=np.int32)
        for i in range(self.n_agents): # THESE LOOPS might become a performance chokepoint for cythonization.
            self.agent_ids_map[self.shuffled_positional_indices[i, 0], self.shuffled_positional_indices[i, 1]] = i

    def initialize_agent_types(self):

        # Agent types are 1 = liberal, 2 = conservative. 0 is left blank to reflect absence of agents.
        self.agent_types = np.random.randint(1, 3, size=self.n_agents).astype(np.int32)
        self.types_map = np.zeros((self.n_rows, self.n_cols), dtype=np.int32)
        for i in range(self.n_agents):
            self.types_map[self.shuffled_positional_indices[i, 0], self.shuffled_positional_indices[i, 1]] = self.agent_types[i]

    def initialize_random_parameters(self):

        if not isinstance(self.agent_ids_map, np.ndarray):
            self.agent_ids_map = np.asarray(self.agent_ids_map)

        self.params_with_random_map = ['d', 's', 'r', 'b']
        self.parameter_maps = {}
        for k in self.params_with_random_map:
            min_p = -10.
            max_p = 10.

            # A bit awkward, becasue each parameter map requires two parameters to define the random initialization, mean and std, I grab both
            # by simply having the convention that the std is the label of mean concatted with 'd'
            mean_p = self.params[k]
            std_p = self.params[k + 'd']

            randomized_parameter_map = self.get_truncated_normal_parameter_map(min_p, max_p, mean_p, std_p, self.spatial_shape, self.agent_ids_map)
            self.parameter_maps[k] = randomized_parameter_map

        # One non-parallelism is that type bias was initialized as a normal trunk, but it should be zero for type 1.
        self.parameter_maps['b'] = np.where(self.types_map == 1, 1.0, self.parameter_maps['b'])
        # Other non-parallelism is that type bias in turn defines R for type 2.
        self.parameter_maps['r'] = np.where(self.types_map == 2, self.parameter_maps['r'] * self.parameter_maps['b'], self.parameter_maps['r'])

        self.parameter_maps['average_reciprocal_response_from_masking'] = np.zeros(self.spatial_shape, dtype=np.float32)
        # self.parameter_maps['immunity_status'] = np.zeros(self.spatial_shape, dtype=np.float32)

    def initialize_choice_variables(self):
        self.masking_choice = np.zeros(self.spatial_shape, dtype=np.float32)
        self.infection_status = np.zeros(self.spatial_shape, dtype=np.float32)
        self.immunity_status = np.zeros(self.spatial_shape, dtype=np.float32)
        
        initialize_with_random_infections = 1
        if initialize_with_random_infections:
            
            # Infect the first few agents
            n_to_infect = 25
            for i in range(n_to_infect):
                self.infection_status[self.shuffled_positional_indices[i, 0], self.shuffled_positional_indices[i, 1]] = 1
                
        self.immunity_efficacy = np.zeros(self.spatial_shape, dtype=np.float32)

    def initialize_reporting_variables(self):
        self.infection_status_last_100 = [0.0] * 100
        self.masking_choice_last_100 = [0.0] * 100

    def get_truncated_normal_parameter_map(self, min, max, mean, std, spatial_shape, agent_ids_map):
        a = sp.stats.truncnorm.rvs((min - mean) / std, (max - mean) / std, loc=mean, scale=std, size=spatial_shape[0] * spatial_shape[1])
        a = np.where(agent_ids_map.flatten() > 0, a, 0.)
        return a.reshape(spatial_shape).astype(np.float32)

    def update(self, n_iterations):
        # Send the model and the n iterations to the cython core. This will update the model arrays for n_iteration number of steps and is very fast.
        start = time.time()
        tribal_masking_computational_core.update_model_arrays(self, np.int32(n_iterations))
        # compute_time_per_iteration = (time.time() - start) / n_iterations
        # compute_time_per_agent = compute_time_per_iteration / self.n_agents

        # print('Compute time per iteration: ' + str(compute_time_per_iteration))



if __name__=='__main__':
    print ('You probably meant to run tribal_masking_main.py')