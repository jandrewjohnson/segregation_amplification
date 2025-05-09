import os
import sys
import random
import numpy as np
from numpy.core import multiarray
from numpy.core.defchararray import upper
import scipy as sp
from scipy.stats import truncnorm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import hazelbean as hb

# Recompile cython file if needed.
recompile_cython = True
if recompile_cython == True:
    cython_command = "python compile_cython_functions.py build_ext -i clean"  #
    returned = os.system(cython_command)
    if returned:
        raise NameError('Cythonization failed.')

import computational_core
import functions

def aspatial_externality_game(p):
    """Generates a two-dimensional plot analyzing how varying the mean of Direct Utilty, D, compared to varying mean of Response
    R variable. Effectively, this is the base model but where EVERYONE is in each agents' neighborhood, rendering space meaningless.
    """
    p.two_dim_analysis_array_path = os.path.join(p.cur_dir, 'two_dim_analysis_array.npy')

    def calc_utility_from_choice(agent, other_agents, choice):  
        d = agent['parameters']['d']
        s = agent['parameters']['s']
        average_reciprocal_response = sum(choice * [i['parameters']['r'] for i in other_agents]) / len(other_agents)
        utility = choice * d +  s * average_reciprocal_response

        return utility

   
    if p.run_this:
        ## Test specific set of parameters
        
        # Initialize agents
        agents = {}
        n_agents = 100
        init = {'d': [-10, 10, -.5, 1], 's': [0, 1, .5, .1], 'r': [-10, 10, 1.0, 1]}
        for i in range(n_agents):
            agents[i] = {}
            agents[i]['parameters'] = {'d': sp.stats.truncnorm.rvs((init['d'][0] - init['d'][2]) / init['d'][3], (init['d'][1] - init['d'][2]) / init['d'][3], loc=init['d'][2], scale=init['d'][3])[0], 
                                       's': sp.stats.truncnorm.rvs((init['s'][0] - init['s'][2]) / init['s'][3], (init['s'][1] - init['s'][2]) / init['s'][3], loc=init['s'][2], scale=init['s'][3])[0],
                                       'r': sp.stats.truncnorm.rvs((init['r'][0] - init['r'][2]) / init['r'][3], (init['r'][1] - init['r'][2]) / init['r'][3], loc=init['r'][2], scale=init['r'][3])[0]}

        # Make choice
        for i in range(n_agents):
            other_agents = [agents[k] for k, v in agents.items() if k != i]

            utility_from_cooperating = calc_utility_from_choice(agents[i], other_agents, 1)  
            utility_from_defecting = calc_utility_from_choice(agents[i], other_agents, -1)  
            if utility_from_cooperating > utility_from_defecting:
                agents[i]['choice'] = 1
                agents[i]['utility'] = utility_from_cooperating
                
            else:
                agents[i]['choice'] = -1
                agents[i]['utility'] = utility_from_defecting

        # Report results
        debug_individual_choices = 1
        if debug_individual_choices:
            for i in range(n_agents):
                print ('Agent ' + str(i) + ' chose ' + str(agents[i]['choice']) + ' and got utility ' + str(agents[i]['utility']) + ' when they had params ' + str(agents[i]['parameters']))

        percent_masked = sum([i['choice'] for i in agents.values() if i['choice'] > 0] ) / len(agents) * 100
        print (str(percent_masked) + '% wore masks for initialization' + str(init))

        # Rerun the full game for different points in D-R space.
        n_sample_points = 20
        if not hb.path_exists(p.two_dim_analysis_array_path) or True:
            # Initialize agents
            to_plot_matrix = np.zeros((n_sample_points, n_sample_points))
            x_label_list = []
            y_label_list = []
            for ji, j in enumerate(np.linspace(-1, 1, n_sample_points)):
                x_label_list.append(-.5+j)
                for ki, k in enumerate(np.linspace(-2, 1, n_sample_points)):
                    y_label_list.append(1.0+k)
                    agents = {}
                    n_agents = 50
                    init = {'d': [-10, 10, -.5 + j, .25], 's': [0, 1, .5, .1], 'r': [-10, 10, 1.0 + k, .25]}
                    for i in range(n_agents):
                        agents[i] = {}
                        agents[i]['parameters'] = {'d': sp.stats.truncnorm.rvs((init['d'][0] - init['d'][2]) / init['d'][3], (init['d'][1] - init['d'][2]) / init['d'][3], loc=init['d'][2], scale=init['d'][3])[0], 
                                                's': sp.stats.truncnorm.rvs((init['s'][0] - init['s'][2]) / init['s'][3], (init['s'][1] - init['s'][2]) / init['s'][3], loc=init['s'][2], scale=init['s'][3])[0],
                                                'r': sp.stats.truncnorm.rvs((init['r'][0] - init['r'][2]) / init['r'][3], (init['r'][1] - init['r'][2]) / init['r'][3], loc=init['r'][2], scale=init['r'][3])[0]}

                    # Make choice for this point
                    for i in range(n_agents):
                        other_agents = [agents[k] for k, v in agents.items() if k != i]

                        utility_from_cooperating = calc_utility_from_choice(agents[i], other_agents, 1)  
                        utility_from_defecting = calc_utility_from_choice(agents[i], other_agents, 0)  
                        if utility_from_cooperating > utility_from_defecting:
                            agents[i]['choice'] = 1
                            agents[i]['utility'] = utility_from_cooperating
                            
                        else:
                            agents[i]['choice'] = 0
                            agents[i]['utility'] = utility_from_defecting

                    # Report results for 
                    debug_individual_choices = 0
                    if debug_individual_choices:
                        for i in range(n_agents):
                            print ('Agent ' + str(i) + ' chose ' + str(agents[i]['choice']) + ' and got utility ' + str(agents[i]['utility']) + ' when they had params ' + str(agents[i]['parameters']))

                    percent_masked = sum([i['choice'] for i in agents.values() if i['choice'] > 0] ) / len(agents) * 100
                    print (str(percent_masked) + '% wore masks for permutation ' + str(j))

                    to_plot_matrix[ji, ki] = percent_masked

            hb.save_array_as_npy(to_plot_matrix, p.two_dim_analysis_array_path)
        else:
            to_plot_matrix = hb.load_npy_as_array(p.two_dim_analysis_array_path)

        x_label_list_pruned = [x_label_list[0], x_label_list[-1]] 
        y_label_list_pruned = [y_label_list[0], y_label_list[-1]] 

        fig, ax = plt.subplots()
        ax.set_title('Masking compliance in aspatial, reciprocal externality game')

        ax.set_xlabel('Mean direct utility from masking')
        ax.set_xticks([0, n_sample_points-1])
        ax.set_xticklabels(x_label_list_pruned)
        
        ax.set_ylabel('Mean reciprocal response')
        ax.set_yticks([0, n_sample_points-1])
        ax.set_yticklabels(y_label_list_pruned)

        im = ax.imshow(to_plot_matrix, cmap='Greens')
        
        cbar = fig.colorbar(im, label='Percent masking')
        fig.savefig(os.path.join(p.cur_dir, 'aspatial_reciprocal.png'))

def spatial_externality_game(p):
    """Minimal spatial externality game solves in space, but for a random distribution. VERY UNINTERESTING
    because the random location means there are very-hard to see spillovers, but here it is."""

    if p.run_this:
        
        shape = (10, 10)
        n_agents = shape[0] * shape[1]

        init = {'d': [-10, 10, -.5, 1], 's': [0, 1, .5, .1], 'r': [-10, 10, 1.0, 1]}


        a = sp.stats.truncnorm.rvs((init['d'][0] - init['d'][2]) / init['d'][3], (init['d'][1] - init['d'][2]) / init['d'][3], loc=init['d'][2], scale=init['d'][3], size=n_agents)
        b_initial = a.reshape(shape).astype(np.float32)

        a = sp.stats.truncnorm.rvs((init['d'][0] - init['d'][2]) / init['d'][3], (init['d'][1] - init['d'][2]) / init['d'][3], loc=init['d'][2], scale=init['d'][3], size=n_agents)
        s_initial = a.reshape(shape).astype(np.float32)

        a = sp.stats.truncnorm.rvs((init['d'][0] - init['d'][2]) / init['d'][3], (init['d'][1] - init['d'][2]) / init['d'][3], loc=init['d'][2], scale=init['d'][3], size=n_agents)
        r_initial = a.reshape(shape).astype(np.float32)


        spatial_result = computational_core.spatial_externality_game(b_initial, s_initial, r_initial)
        spatial_result = np.asarray(spatial_result)


        hb.show(spatial_result, output_path=os.path.join(p.cur_dir, 'spatial_externality_game.png'))


def spatial_segregation_game(p):
    """Combine the models and generate static outputs and/or animations."""

    p.spatial_segregation_game_animation_paths = {}
    p.spatial_segregation_game_animation_paths['segregation_game_no_convergence'] = os.path.join(p.cur_dir, 'segregation_game_no_convergence.gif')
    p.spatial_segregation_game_animation_paths['segregation_game_leaving_dissimilar'] = os.path.join(p.cur_dir, 'segregation_game_leaving_dissimilar.gif')
    p.spatial_segregation_game_animation_paths['segregation_game_insufficient_similar'] = os.path.join(p.cur_dir, 'segregation_game_insufficient_similar.gif')

    p.spatial_segregation_game_parameters = {}
    p.spatial_segregation_game_parameters['segregation_game_no_convergence'] = {'threshold': .7, 'game_type': 'move_if_too_many_dissimilar'}
    p.spatial_segregation_game_parameters['segregation_game_leaving_dissimilar'] = {'threshold': .3, 'game_type': 'move_if_too_many_dissimilar'}
    p.spatial_segregation_game_parameters['segregation_game_insufficient_similar'] = {'threshold': .3, 'game_type': 'move_if_not_enough_similar'}


    if p.run_this:


        # Define the size of the societal space
        spatial_shape = (100, 200)

        # In order to allow agents to be able to move, the segregation game has some portion of the initial "houses" unoccupied. Set the proportion filled here.
        proportion_filled = .75

        # Number of agents is a function then of the size of the space and the proportion filled, rounded down and inted.
        n_agents = int(np.floor(spatial_shape[0] * spatial_shape[1] * proportion_filled))

        # For convenience, we define input parameters here via a dictionary.
        init = {'d': [-10, 10, -.5, 1], 's': [0, 1, .5, .1], 'r': [-10, 10, 1.0, 1]}

        # Keep track of all the agents via IDs
        agent_ids = np.arange(0, n_agents).astype(np.int64)


        for name, animation_path in p.spatial_segregation_game_animation_paths.items():


            ### INITIALIZATION
            # Save row and col ids as two new parameters so that we can quickly look up paraeters my spatial location OR agent id.
            row_ids = np.arange(spatial_shape[0])
            col_ids = np.arange(spatial_shape[1])

            # positional_indices records the 2-length r, c of each agent grid-cell based on their r, c.
            positional_indices = np.empty((spatial_shape[0], spatial_shape[1], 2), dtype=int)
            positional_indices[:, :, 0] = row_ids[:, None]
            positional_indices[:, :, 1] = col_ids

            # Somewhat convoluted way of randomizing the initial postitions.
            shuffled_positional_indices = np.copy(positional_indices) # Operate on a copy to keep original

            # Shuffle requires a flattened array, so reshape it from 3d to 2d, keeping the 2d r-c indices.
            new_shape = (shuffled_positional_indices.shape[0] * shuffled_positional_indices.shape[1], shuffled_positional_indices.shape[2])
            shuffled_positional_indices = shuffled_positional_indices.reshape(new_shape)

            # Randomize the order by creating a randomized list of r-c indices for future fast iterating.
            # LEARNING POINT, this was the most efficient approach I could think of for shuffling in the first 2 of three directions while keeping the third intact.
            rng = np.random.default_rng()
            rng.shuffle(shuffled_positional_indices, axis=0)

            # First initialize all agents_list on a random location according to the initial shuffled queue.
            agent_locations = shuffled_positional_indices[:n_agents]

            # Also initialize the unoccupied locations. When agents move, the grab a new spot from this list and move their previous slot into this array.
            unoccupied_locations = shuffled_positional_indices[n_agents:]

            # Agent types are 1 = liberal, 2 = conservative. 0 is left blank to reflect absence of agents.
            agent_types = np.random.randint(1, 3, size=n_agents).astype(np.int64)

            # agent ids and types are initialized as maps.
            agent_ids_map = np.zeros((spatial_shape[0], spatial_shape[1]), dtype=np.int64)
            types_map = np.zeros((spatial_shape[0], spatial_shape[1]), dtype=np.int64)
            for i in range(n_agents):
                agent_ids_map[shuffled_positional_indices[i, 0], shuffled_positional_indices[i, 1]] = i
                types_map[shuffled_positional_indices[i, 0], shuffled_positional_indices[i, 1]] = agent_types[i]

            do_animation = 1
            types_map_plot_list = []


            threshold = p.spatial_segregation_game_parameters[name]['threshold']
            game_type = p.spatial_segregation_game_parameters[name]['game_type']

            # hb.show(types_map)
            hb.timer('starting cython')
            computational_core.spatial_segregation_game(agent_ids, agent_locations, unoccupied_locations, agent_types, agent_ids_map,
                                                                       types_map, threshold=threshold, game_type=game_type, reporting_threshold=1)
            hb.timer('Initial run of spatial_segregation_game took:')
            if do_animation:
                types_map_animation_array = np.copy(types_map)
                types_map_plot_list.append(types_map_animation_array)

            previous_n_changed = 0
            for i in range(333):
                n_changed = computational_core.spatial_segregation_game(agent_ids, agent_locations, unoccupied_locations, agent_types, agent_ids_map,
                                                                                       types_map, threshold=threshold, game_type=game_type, reporting_threshold=1)
                if n_changed <= 0:
                    break
                if do_animation:
                    if n_changed != previous_n_changed:
                        previous_n_changed = n_changed
                        types_map_animation_array = np.copy(types_map)
                        types_map_plot_list.append(types_map_animation_array)

            hb.timer('Finished cython cython')
            # hb.show(types_map)

            if do_animation:
                figsize = (8, 4)
                fig, ax = plt.subplots(figsize=figsize)
                x = np.arange(0, types_map_plot_list[0].shape[1])
                y = np.arange(0, types_map_plot_list[0].shape[0])

                def animate(i):
                    pc = ax.pcolormesh(x, y, types_map_plot_list[i])
                    return pc, # WTF THIS REQUIRES THE commaa

                anim = FuncAnimation(fig, animate, frames=len(types_map_plot_list), interval=100, repeat_delay=1000, blit=True) #
                anim.save(animation_path)

def combined_game_noninteractive(p):
    """Produce animations for the combined game."""

    p.combined_game_animation_path = os.path.join(p.cur_dir, 'combined_game_animation.gif')

    if p.run_this:

        # Define the size of the societal space
        spatial_shape = (100, 200)

        # In order to allow agents to be able to move, the segregation game has some portion of the initial "houses" unoccupied. Set the proportion filled here.
        proportion_filled = .75

        # For convenience, we define input parameters here via a dictionary.
        params = {'d': [-10, 10, -.25, .05], 's': [0, 1, 0.9, .1], 'r': [-10, 10, 1.0, 1.95]}
        type_bias = -.1

        agent_ids, agent_locations, unoccupied_locations, agent_types, agent_ids_map, types_map, d, s, r = functions.initialize_agents(spatial_shape, proportion_filled, type_bias, params)

        do_animation = 1
        types_map_plot_list = []
        masking_choice_plot_list = []

        game_type = 'move_if_too_many_dissimilar'
        game_type = 'move_if_not_enough_similar'

        threshold = 0.5
        # hb.show(types_map)
        # hb.show(r)

        if do_animation:

            types_map_animation_array = np.copy(types_map)
            types_map_plot_list.append(types_map_animation_array)

            ## NOTE be careful about thinking which things already have an initialization and which do not. Masking does not as it's a result.
            # masking_choice_animation_array = np.copy(masking_choice)
            # masking_choice_plot_list.append(masking_choice_animation_array)

        # IMPORTANT, in order to enable interactive plotting below, I pass a bunch of trivially initialized variables here which get fully initialized in the cython function
        masking_choice = np.zeros(spatial_shape).astype(np.float32)
        average_reciprocal_response_from_masking = np.zeros(spatial_shape).astype(np.float32)
        average_reciprocal_response_from_not_masking = np.zeros(spatial_shape).astype(np.float32)

        previous_n_changed = 0
        n_iterations = 300
        neighborhood_radius = 3
        ims_to_animate = {}
        to_plot_names = ['b', 's', 'r', 'masking_choice', 'average_reciprocal_response_from_masking', 'average_reciprocal_response_from_not_masking']
        to_plot_dict = {}
        for c, j in enumerate(to_plot_names):
            to_plot_dict[j] = []

        mean_metric_map = np.zeros(spatial_shape).astype(np.float32)

        hb.timer('Starting iterations of combined_game_noninteractive.')
        for i in range(n_iterations):

            # CALCULATE THE ACTUAL GAME.
            n_changed = computational_core.spatial_segregation_externality_game(
                agent_ids, agent_locations, unoccupied_locations, agent_types, agent_ids_map,
                types_map, threshold, neighborhood_radius, game_type,
                d, s, r, masking_choice, average_reciprocal_response_from_masking, mean_metric_map,
                reporting_threshold=0
            )



            # End early if few agents move locations.
            min_changers_to_end = 10
            if n_changed <= min_changers_to_end:
                break

            # At each iteration where there was a successful move, copy the relevant arrays for plotting later.

            if do_animation:

                if n_changed != previous_n_changed:
                    to_plot_memviews = [d, s, r, masking_choice, average_reciprocal_response_from_masking, average_reciprocal_response_from_not_masking]
                    to_plot_arrays = [to_plot_dict[to_plot_names[i]].append(np.copy(to_plot_memviews[i])) for i in range(len(to_plot_names))]
                    previous_n_changed = n_changed

        hb.timer('Finished iterations of combined_game_noninteractive.')

        if do_animation:
            for name in to_plot_names:
                fig, ax = plt.subplots()
                ims = []
                for c in range(len(to_plot_dict[name])):
                    im = ax.imshow(to_plot_dict[name][c], interpolation='nearest')
                    ims.append([im])
                ani = matplotlib.animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1)
                ani.save(os.path.join(p.cur_dir, name + '_animation.mp4'), fps=6, writer='ffmpeg', dpi=150, bitrate=-1)

def combined_game_interactive_full_resolve(p):

    p.combined_game_animation_path = os.path.join(p.cur_dir, hb.ruri('combined_game_animation.gif'))

    if p.run_this:

        # Define the size of the societal space
        spatial_shape = (75, 75)

        # In order to allow agents to be able to move, the segregation game has some portion of the initial "houses" unoccupied. Set the proportion filled here.
        proportion_filled = .75

        n_agents = int(np.floor(spatial_shape[0] * spatial_shape[1] * proportion_filled))

        do_animation = 0
        types_map_plot_list = []
        masking_choice_plot_list = []

        game_type = 'move_if_too_many_dissimilar'
        game_type = 'move_if_not_enough_similar'

        threshold = 0.5

        # IMPORTANT, in order to enable interactive plotting below, I pass a bunch of trivially initialized variables here which get fully initialized in the cython function

        average_reciprocal_response_from_masking = np.zeros(spatial_shape).astype(np.float32)

        previous_n_changed = 0
        n_iterations = 300


        # After the model has converged, get the masking choice array and plot it as an imshow
        threshold_init = 0.25
        neighborhood_radius_init = 3
        type_bias_init = -0.1
        params_init = {'d': [-10, 10, -.25, .0000001], 's': [0, 10, 1.0, .0000001], 'r': [-10, 10, 3.0, .0000001]}
        d_init = params_init['d'][2]
        dd_init = params_init['d'][3]
        s_init = params_init['s'][2]
        sd_init = params_init['s'][3]
        r_init = params_init['r'][2]
        rd_init = params_init['r'][3]
        
        a1, a2, a3, a4 = functions.get_masking_choice_full_resolve(spatial_shape, proportion_filled, game_type, threshold_init, neighborhood_radius_init, type_bias_init, d_init, dd_init, s_init, sd_init, r_init, rd_init)
        a2 = np.where(a2 > 0, a2, np.nan)

        functions.plot_combined_game_interactive_full_resolve(a1, a2, a3, a4, spatial_shape, proportion_filled, game_type, n_agents, threshold_init, neighborhood_radius_init, type_bias_init, d_init, dd_init, s_init, sd_init, r_init, rd_init)


def combined_game_interactive_time_variant(p):

    if p.run_this:

        # Define the size of the societal space
        spatial_shape = (200, 400)

        # In order to allow agents to be able to move, the segregation game has some portion of the initial "houses" unoccupied. Set the proportion filled here.
        proportion_filled = .75

        n_agents = int(np.floor(spatial_shape[0] * spatial_shape[1] * proportion_filled))

        # There are two different versions of the game
        game_type = 'move_if_too_many_dissimilar'
        game_type = 'move_if_not_enough_similar'

        # Initialize a few runtime variables
        previous_n_changed = 0
        n_iterations = 900

        # IMPORTANT, in order to enable interactive plotting below, I pass a bunch of trivially initialized variables here which get fully initialized in the cython function
        average_reciprocal_response_from_masking = np.zeros(spatial_shape).astype(np.float32)



        def get_initial_masking_choice(threshold, neighborhood_radius, type_bias, d_param, dd_param, s_param, sd_param, r_param, rd_param):
            # For convenience, we define input parameters here via a dictionary.
            params = {'d': [-10, 10, d_param, dd_param], 's': [0, 1, s_param, sd_param], 'r': [-10, 10, r_param, rd_param]}
            # params = {'d': [-10, 10, -.25, .05], 's': [0, 1, 0.9, .1], 'r': [-10, 10, 1.0, 1.95]}
            masking_choice = np.zeros(spatial_shape).astype(np.float32)
            mean_metric_map = np.zeros(spatial_shape).astype(np.float32)

            agent_ids, agent_locations, unoccupied_locations, agent_types, agent_ids_map, types_map, d, s, r = functions.initialize_agents(spatial_shape, proportion_filled, type_bias, params)
            neighborhood_radius = np.int64(neighborhood_radius)

            for i in range(n_iterations):
                n_changed = computational_core.spatial_segregation_externality_game(
                    agent_ids, agent_locations, unoccupied_locations, agent_types, agent_ids_map,
                    types_map, threshold, neighborhood_radius, game_type,
                    d, s, r, masking_choice, average_reciprocal_response_from_masking, mean_metric_map,
                    reporting_threshold=0)
                # End early if few agents move locations.
                min_changers_to_end = 1
                if n_changed <= min_changers_to_end:
                    break

            return np.copy(np.asarray(masking_choice)), np.copy(np.asarray(types_map)), np.copy(np.asarray(mean_metric_map)), np.copy(np.asarray(average_reciprocal_response_from_masking)), agent_ids, agent_locations, unoccupied_locations, agent_types, agent_ids_map, types_map, d, s, r, masking_choice, mean_metric_map
            # return np.copy(np.asarray(masking_choice)), np.copy(np.asarray(types_map))

        def get_masking_choice(threshold, neighborhood_radius, type_bias, d_param, dd_param, s_param, sd_param, r_param, rd_param):
            # For convenience, we define input parameters here via a dictionary.
            params = {'d': [-10, 10, d_param, dd_param], 's': [0, 1, s_param, sd_param], 'r': [-10, 10, r_param, rd_param]}
            # params = {'d': [-10, 10, -.25, .05], 's': [0, 1, 0.9, .1], 'r': [-10, 10, 1.0, 1.95]}
            masking_choice = np.zeros(spatial_shape).astype(np.float32)
            mean_metric_map = np.zeros(spatial_shape).astype(np.float32)
            agent_ids, agent_locations, unoccupied_locations, agent_types, agent_ids_map, types_map, d, s, r = initialize_agents(spatial_shape, proportion_filled, type_bias, params)
            neighborhood_radius = np.int64(neighborhood_radius)
            n_iterations = 150
            for i in range(n_iterations):

                n_changed = computational_core.spatial_segregation_externality_game(
                    agent_ids, agent_locations, unoccupied_locations, agent_types, agent_ids_map,
                    types_map, threshold, neighborhood_radius, game_type,
                    d, s, r, masking_choice, average_reciprocal_response_from_masking, mean_metric_map,
                    reporting_threshold=0)
                # End early if few agents move locations.
                min_changers_to_end = 1
                if n_changed <= min_changers_to_end:
                    break

            return np.copy(np.asarray(masking_choice)), np.copy(np.asarray(types_map)), np.copy(np.asarray(mean_metric_map)), np.copy(np.asarray(average_reciprocal_response_from_masking))
            # return np.copy(np.asarray(masking_choice)), np.copy(np.asarray(types_map))

        def sliders_on_changed_full_resolve(val):
            a, a2, a3, a4  = get_masking_choice(threshold_slider.val, neighborhood_radius_slider.val, type_bias_slider.val, d_slider.val, dd_slider.val, s_slider.val, sd_slider.val, r_slider.val, rd_slider.val)
            a2 = np.where(a2 > 0, a2, np.nan)
            im_ul.set_data(a)
            im_ur.set_data(a2)
            im_ll.set_data(a3)
            im_lr.set_data(a4)

            percent_masked = np.sum(np.where(a == 1, 1, 0)) / n_agents
            percent_masked_annotation.set_text('Percent masked: ' + str(hb.round_significant_n(percent_masked, 4)))

            fig.canvas.draw_idle()




        def update_masking_choice_without_reposition(threshold, neighborhood_radius, type_bias, d_param, dd_param, s_param, sd_param, r_param, rd_param,
                                                     agent_ids, agent_locations, unoccupied_locations, agent_types, agent_ids_map, types_map, d, s, r):


            # TODOO I've gotten very confused about which variables need to be passed to which function. For instgance, type bias here is not used because only the ARRAY was.

            n_changed = computational_core.spatial_segregation_externality_game_update_masking_params(
                agent_ids, agent_locations, unoccupied_locations, agent_types, agent_ids_map,
                types_map, threshold, neighborhood_radius, game_type,
                d, s, r, masking_choice, average_reciprocal_response_from_masking, mean_metric_map,
                reporting_threshold=0)


            return np.copy(np.asarray(masking_choice)), np.copy(np.asarray(types_map)), np.copy(np.asarray(mean_metric_map)), np.copy(np.asarray(average_reciprocal_response_from_masking))
            # return np.copy(np.asarray(masking_choice)), np.copy(np.asarray(types_map))

        # After the model has converged, get the masking choice array and plot it as an imshow
        threshold_init = 0.45
        neighborhood_radius_init = 3
        type_bias_init = -0.1
        params_init = {'d': [-10, 10, -.25, .0000001], 's': [0, 10, 1.0, .0000001], 'r': [-10, 10, 3.0, .0000001]}
        d_init = params_init['d'][2]
        dd_init = params_init['d'][3]
        s_init = params_init['s'][2]
        sd_init = params_init['s'][3]
        r_init = params_init['r'][2]
        rd_init = params_init['r'][3]

        agent_ids=None

        a1, a2, a3, a4, agent_ids, agent_locations, unoccupied_locations, agent_types, agent_ids_map, types_map, d, s, r, masking_choice, mean_metric_map \
            = functions.get_initial_masking_choice_time_variant(threshold_init, neighborhood_radius_init, type_bias_init, d_init, dd_init, s_init, sd_init, r_init, rd_init)


        a2 = np.where(a2 > 0, a2, np.nan)


        functions.plot_combined_game_interactive_time_variant(a1, a2, a3, a4, spatial_shape, proportion_filled, game_type, n_agents, threshold_init, neighborhood_radius_init, type_bias_init, d_init, dd_init, s_init, sd_init, r_init, rd_init)



        # fmt = None
        # col_dict = {-1: "purple",
        #             0: "white",
        #             1: "green", }
        # labels = np.array(["No mask", "Vacant", "Mask"])
        # cm = matplotlib.colors.ListedColormap([col_dict[x] for x in col_dict.keys()])
        # len_lab = len(labels)
        # norm_bins = np.sort([*col_dict.keys()]) + 0.5
        # norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
        # norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
        # # try:
        # #     fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])
        # # except:
        # #     fmt = None
        # diff = norm_bins[1:] - norm_bins[:-1]
        # tickz = norm_bins[:-1] + diff / 2
        #
        # im_ul = ax_ul.imshow(a, cmap=cm, norm=norm, interpolation='nearest')
        # cb_ul = fig.colorbar(im_ul, ax=ax_ul, format=fmt, ticks=tickz)
        #
        # col_dict = {
        #     0: "white",
        #     1: "blue",
        #     2: "red"
        # }
        # labels = np.array(["Vacant", "Tribe 1", "Tribe 2"])
        # cm = matplotlib.colors.ListedColormap([col_dict[x] for x in col_dict.keys()])
        # len_lab = len(labels)
        # norm_bins = np.sort([*col_dict.keys()]) + 0.5
        # norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
        # norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
        # # try:
        # #     fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])
        # # except:
        # #     fmt = None
        # diff = norm_bins[1:] - norm_bins[:-1]
        # tickz = norm_bins[:-1] + diff / 2
        #
        # im_ur = ax_ur.imshow(a2, cmap=cm, norm=norm, interpolation='nearest')
        # cb_ur = fig.colorbar(im_ur, ax=ax_ur, format=fmt, ticks=tickz)
        #
        # # im_ur = ax_ur.imshow(a2, cmap='Set1_r', interpolation='nearest', vmin=-7, vmax=3)
        # im_ll = ax_ll.imshow(a3, cmap='YlGn', interpolation='nearest', vmin=0, vmax=1)
        # im_lr = ax_lr.imshow(a4, cmap='BrBG', interpolation='nearest', vmin=-2, vmax=2)
        #
        # # im_ul = ax_ul.imshow(a, cmap='PRGn', interpolation='nearest', vmin=-1.5, vmax=1.5)
        # # im_ur = ax_ur.imshow(a2, cmap='Set1_r', interpolation='nearest', vmin=-7, vmax=3)
        # # im_ll = ax_ll.imshow(a3, cmap='YlGn', interpolation='nearest', vmin=0, vmax=1)
        # # im_lr = ax_lr.imshow(a4, cmap='BrBG', interpolation='nearest', vmin=-2, vmax=2)
        # #
        # percent_masked = np.sum(np.where(a == 1, 1, 0)) / n_agents
        # percent_masked_annotation = ax_ul.annotate('Percent masked: ' + str(hb.round_significant_n(percent_masked, 4)), xy=(0.01, 0.95), xycoords='figure fraction')
        #
        # # diff = norm_bins[1:] - norm_bins[:-1]
        # # tickz = norm_bins[:-1] + diff / 2
        # # cb_ul = fig.colorbar(im_ul, ax=ax_ul, format=fmt, ticks=tickz)
        # # cb_ul = fig.colorbar(im_ul, ax=ax_ul, ticks=[-1, 1], shrink=.8 )
        # # cb_ul.ax.set_yticklabels(['no mask', 'mask'])
        #
        # # fig.colorbar(im_ur, ax=ax_ur, shrink=.8)
        #
        # fig.colorbar(im_ll, ax=ax_ll, shrink=.8)
        # fig.colorbar(im_lr, ax=ax_lr, shrink=.8, extend='both')
        #
        # ax_ul.set_title('Masking choice')
        # ax_ur.set_title('Political type')
        # ax_ll.set_title('Neighborhood similarity')
        # ax_lr.set_title('Neighborhood r')
        #
        # ax_ul.axis('off')
        # ax_ur.axis('off')
        # ax_ll.axis('off')
        # ax_lr.axis('off')



        # START HERE: Combined two games but got lost on variable scope and now agent_locations doesn't update when you drag a non-full-update variable. This probably
        # suggests I need a full on data model.

        # # Define an action for modifying the line when any slider's value changes
        # def sliders_on_changed_just_masking_updates(val):
        #
        #     params = {'d': [-10, 10, d_slider.val, dd_slider.val], 's': [0, 1, s_slider.val, sd_slider.val], 'r': [-10, 10, r_slider.val, rd_slider.val]}
        #     # params_init = {'d': [-10, 10, -.25, .0000001], 's': [0, 10, 1.0, .0000001], 'r': [-10, 10, 3.0, .0000001]}
        #     # d_init = params_init['d'][2]
        #     # dd_init = params_init['d'][3]
        #     # s_init = params_init['s'][2]
        #     # sd_init = params_init['s'][3]
        #     # r_init = params_init['r'][2]
        #     # rd_init = params_init['r'][3]
        #
        #     row_ids = np.arange(spatial_shape[0])
        #     col_ids = np.arange(spatial_shape[1])
        #
        #     # Number of agents is a function then of the size of the space and the proportion filled, rounded down and inted.
        #     n_agents = int(np.floor(spatial_shape[0] * spatial_shape[1] * proportion_filled))
        #     n_cells = spatial_shape[0] * spatial_shape[1]
        #
        #     a = sp.stats.truncnorm.rvs((params['d'][0] - params['d'][2]) / params['d'][3], (params['d'][1] - params['d'][2]) / params['d'][3], loc=params['d'][2], scale=params['d'][3], size=n_cells)
        #     a = np.where(agent_ids_map.flatten() > 0, a, 0.)
        #     d_paramsial = a.reshape(spatial_shape).astype(np.float32)
        #     d = np.copy(d_paramsial)
        #
        #     a = sp.stats.truncnorm.rvs((params['s'][0] - params['s'][2]) / params['s'][3], (params['s'][1] - params['s'][2]) / params['s'][3], loc=params['s'][2], scale=params['s'][3], size=n_cells)
        #     a = np.where(agent_ids_map.flatten() > 0, a, 0.)
        #     s_paramsial = a.reshape(spatial_shape).astype(np.float32)
        #     s = np.copy(s_paramsial)
        #
        #     a = sp.stats.truncnorm.rvs((params['r'][0] - params['r'][2]) / params['r'][3], (params['r'][1] - params['r'][2]) / params['r'][3], loc=params['r'][2], scale=params['r'][3], size=n_cells)
        #     a = np.where(agent_ids_map.flatten() > 0, a, 0.)
        #     a = np.where(types_map.flatten() == 2, a * type_bias_slider.val, a)
        #     r_initial = a.reshape(spatial_shape).astype(np.float32)
        #     r = np.copy(r_initial)
        #
        #
        #
        #     a, a2, a3, a4 = update_masking_choice_without_reposition(threshold_slider.val, neighborhood_radius_slider.val, type_bias_slider.val, d_slider.val, dd_slider.val, s_slider.val, sd_slider.val, r_slider.val, rd_slider.val,
        #                                                              agent_ids, agent_locations, unoccupied_locations, agent_types, agent_ids_map, types_map, d, s, r)
        #     a2 = np.where(a2 > 0, a2, np.nan)
        #     im_ul.set_data(a)
        #     im_ur.set_data(a2)
        #     im_ll.set_data(a3)
        #     im_lr.set_data(a4)
        #
        #     percent_masked = np.sum(np.where(a == 1, 1, 0)) / n_agents
        #     percent_masked_annotation.set_text('Percent masked: ' + str(hb.round_significant_n(percent_masked, 4)))
        #
        #     fig.canvas.draw_idle()

        # # Turn off hover text (was breaking the window size)
        # ax_ul.format_coord = lambda x, y: ""
        # ax_ur.format_coord = lambda x, y: ""
        # ax_ll.format_coord = lambda x, y: ""
        # ax_lr.format_coord = lambda x, y: ""
        #
        # # Add Sliders
        # # Define an axes area and draw a slider in it
        # threshold_slider_ax = fig.add_axes([.3, 0.27, 0.4, 0.03], facecolor=axis_color)
        # threshold_slider = Slider(threshold_slider_ax, 'threshold', 0.0, 1.0, valinit=threshold_init)
        # threshold_slider.on_changed(sliders_on_changed_full_resolve)
        #
        # neighborhood_radius_slider_ax = fig.add_axes([.3, 0.24, 0.4, 0.03], facecolor=axis_color)
        # neighborhood_radius_slider = Slider(neighborhood_radius_slider_ax, 'neighborhood_radius', 1., 6., valinit=neighborhood_radius_init)
        # neighborhood_radius_slider.on_changed(sliders_on_changed_just_masking_updates)
        #
        # type_bias_slider_ax = fig.add_axes([.3, 0.21, 0.4, 0.03], facecolor=axis_color)
        # type_bias_slider = Slider(type_bias_slider_ax, 'type_bias', -2., 2., valinit=type_bias_init)
        # type_bias_slider.on_changed(sliders_on_changed_just_masking_updates)
        #
        # d_slider_ax = fig.add_axes([0.3, 0.18, 0.4, 0.03], facecolor=axis_color)
        # d_slider = Slider(d_slider_ax, 'd', -5., 5., valinit=d_init)
        # d_slider.on_changed(sliders_on_changed_just_masking_updates)
        #
        # dd_slider_ax = fig.add_axes([0.3, 0.15, 0.4, 0.03], facecolor=axis_color)
        # dd_slider = Slider(dd_slider_ax, 'dd', 0.00001, 5., valinit=dd_init)
        # dd_slider.on_changed(sliders_on_changed_just_masking_updates)
        #
        # s_slider_ax = fig.add_axes([0.3, 0.12, 0.4, 0.03], facecolor=axis_color)
        # s_slider = Slider(s_slider_ax, 's', .0000001, 4., valinit=s_init)
        # s_slider.on_changed(sliders_on_changed_just_masking_updates)
        #
        # sd_slider_ax = fig.add_axes([0.3, 0.09, 0.4, 0.03], facecolor=axis_color)
        # sd_slider = Slider(sd_slider_ax, 'sd', 0.0000001, 1., valinit=sd_init)
        # sd_slider.on_changed(sliders_on_changed_just_masking_updates)
        #
        # r_slider_ax = fig.add_axes([0.3, 0.06, 0.4, 0.03], facecolor=axis_color)
        # r_slider = Slider(r_slider_ax, 'r', -5, 5, valinit=r_init)
        # r_slider.on_changed(sliders_on_changed_just_masking_updates)
        #
        # rd_slider_ax = fig.add_axes([0.3, .03, 0.4, 0.03], facecolor=axis_color)
        # rd_slider = Slider(rd_slider_ax, 'rd', 0.0000001, 1., valinit=rd_init)
        # rd_slider.on_changed(sliders_on_changed_just_masking_updates)
        #
        # plt.show()
        #
        #
