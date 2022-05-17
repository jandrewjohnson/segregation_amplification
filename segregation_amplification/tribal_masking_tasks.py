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
from hazelbean import visualization

# Recompile cython file if needed.
recompile_cython = True
if recompile_cython == True:
    cython_command = "python compile_cython_functions.py build_ext -i clean"  #
    returned = os.system(cython_command)
    if returned:
        raise NameError('Cythonization failed.')

import tribal_masking_functions
import tribal_masking_computational_core
import tribal_masking_tasks
import tribal_masking_visualization

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
            agents[i]['parameters'] = {'d': sp.stats.truncnorm.rvs((init['d'][0] - init['d'][2]) / init['d'][3], (init['d'][1] - init['d'][2]) / init['d'][3], loc=init['d'][2], scale=init['d'][3]),
                                       's': sp.stats.truncnorm.rvs((init['s'][0] - init['s'][2]) / init['s'][3], (init['s'][1] - init['s'][2]) / init['s'][3], loc=init['s'][2], scale=init['s'][3]),
                                       'r': sp.stats.truncnorm.rvs((init['r'][0] - init['r'][2]) / init['r'][3], (init['r'][1] - init['r'][2]) / init['r'][3], loc=init['r'][2], scale=init['r'][3])}

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
                        agents[i]['parameters'] = {'d': sp.stats.truncnorm.rvs((init['d'][0] - init['d'][2]) / init['d'][3], (init['d'][1] - init['d'][2]) / init['d'][3], loc=init['d'][2], scale=init['d'][3]),
                                                's': sp.stats.truncnorm.rvs((init['s'][0] - init['s'][2]) / init['s'][3], (init['s'][1] - init['s'][2]) / init['s'][3], loc=init['s'][2], scale=init['s'][3]),
                                                'r': sp.stats.truncnorm.rvs((init['r'][0] - init['r'][2]) / init['r'][3], (init['r'][1] - init['r'][2]) / init['r'][3], loc=init['r'][2], scale=init['r'][3])}

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


        spatial_result = tribal_masking_computational_core.spatial_externality_game(b_initial, s_initial, r_initial)
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
        agent_ids = np.arange(0, n_agents).astype(np.int)


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
            agent_types = np.random.randint(1, 3, size=n_agents).astype(np.int)

            # agent ids and types are initialized as maps.
            agent_ids_map = np.zeros((spatial_shape[0], spatial_shape[1]), dtype=np.int)
            types_map = np.zeros((spatial_shape[0], spatial_shape[1]), dtype=np.int)
            for i in range(n_agents):
                agent_ids_map[shuffled_positional_indices[i, 0], shuffled_positional_indices[i, 1]] = i
                types_map[shuffled_positional_indices[i, 0], shuffled_positional_indices[i, 1]] = agent_types[i]

            do_animation = 1
            types_map_plot_list = []


            threshold = p.spatial_segregation_game_parameters[name]['threshold']
            game_type = p.spatial_segregation_game_parameters[name]['game_type']

            # hb.show(types_map)
            hb.timer('starting cython')
            tribal_masking_computational_core.spatial_segregation_game(agent_ids, agent_locations, unoccupied_locations, agent_types, agent_ids_map,
                                                                       types_map, threshold=threshold, game_type=game_type, reporting_threshold=1)
            hb.timer('Initial run of spatial_segregation_game took:')
            if do_animation:
                types_map_animation_array = np.copy(types_map)
                types_map_plot_list.append(types_map_animation_array)

            previous_n_changed = 0
            for i in range(333):
                n_changed = tribal_masking_computational_core.spatial_segregation_game(agent_ids, agent_locations, unoccupied_locations, agent_types, agent_ids_map,
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

        agent_ids, agent_locations, unoccupied_locations, agent_types, agent_ids_map, types_map, d, s, r = tribal_masking_functions.initialize_agents(spatial_shape, proportion_filled, type_bias, params)

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
            n_changed = tribal_masking_computational_core.spatial_segregation_externality_game(
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
                ani.save(os.path.join(p.cur_dir, name + '_animation.gif'), fps=6, dpi=150, bitrate=-1)

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
        
        a1, a2, a3, a4 = tribal_masking_functions.get_masking_choice_full_resolve(spatial_shape, proportion_filled, game_type, threshold_init, neighborhood_radius_init, type_bias_init, d_init, dd_init, s_init, sd_init, r_init, rd_init)
        a2 = np.where(a2 > 0, a2, np.nan)

        tribal_masking_visualization.plot_combined_game_interactive_full_resolve(a1, a2, a3, a4, spatial_shape, proportion_filled, game_type, n_agents, threshold_init, neighborhood_radius_init, type_bias_init, d_init, dd_init, s_init, sd_init, r_init, rd_init)


def combined_game_interactive_time_variant(p):
    """This was an aborted attempt to make it solve with time-steps, but I decided it was too complex and needed
    to go object oriented"""
    if p.run_this:

        # Define the size of the societal space
        spatial_shape = (50, 100)

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
            = tribal_masking_functions.get_initial_masking_choice_time_variant(threshold_init, spatial_shape, proportion_filled, n_iterations, game_type, neighborhood_radius_init, type_bias_init, d_init, dd_init, s_init, sd_init, r_init, rd_init)


        a2 = np.where(a2 > 0, a2, np.nan)


        tribal_masking_visualization.plot_combined_game_interactive_time_variant(a1, a2, a3, a4, spatial_shape, proportion_filled, game_type, n_agents, threshold_init, neighborhood_radius_init, type_bias_init, d_init, dd_init, s_init, sd_init, r_init, rd_init)



def test_different_data_models_performance(p):

    if p.run_this:

        # Define the size of the societal space
        spatial_shape = (50, 100)

        # In order to allow agents to be able to move, the segregation game has some portion of the initial "houses" unoccupied. Set the proportion filled here.
        proportion_filled = .75

        n_agents = int(np.floor(spatial_shape[0] * spatial_shape[1] * proportion_filled))

        # There are two different versions of the game
        game_type = 'move_if_too_many_dissimilar'
        game_type = 'move_if_not_enough_similar'

        # Initialize a few runtime variables
        previous_n_changed = 0
        n_iterations = 111

        # IMPORTANT, in order to enable interactive plotting below, I pass a bunch of trivially initialized variables here which get fully initialized in the cython function
        average_reciprocal_response_from_masking = np.zeros(spatial_shape).astype(np.float32)




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

        do_performance_tests = 1
        if do_performance_tests:
            n_tests = 10
            hb.timer()
            model = tribal_masking_functions.tribal_masking_model(spatial_shape)

            hb.timer('Initialize model')

            for i in range(n_tests):
                tribal_masking_computational_core.spatial_segregation_only(model, n_iterations)

            hb.timer('memview')


            model = tribal_masking_functions.tribal_masking_model(spatial_shape)

            hb.timer('Initialize model')
            for i in range(n_tests):
                tribal_masking_computational_core.spatial_segregation_only_non_memview(model, n_iterations)

            hb.timer('array')

        model = tribal_masking_functions.tribal_masking_model(spatial_shape)

        tribal_masking_visualization.plot_combined_game_interactive_sliders(model)
        # hb.visualization.show(np.asarray(model.agent_ids_map))


def combined_game_with_policies(p):

    if p.run_this:

        model = tribal_masking_functions.tribal_masking_model((50, 100))

        tribal_masking_visualization.plot_combined_game_interactive_sliders(model)

def combined_game_with_policies_and_infection(p):

    if p.run_this:

        model = tribal_masking_functions.tribal_masking_model((50, 100))

        tribal_masking_visualization.plot_combined_game_interactive_infections(model)


