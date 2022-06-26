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

import tribal_masking_model_classes
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


def combined_game_with_policies_and_infection(p):

    if p.run_this:
        # START HERE: Consider having additional plotting functions that incorporate line graphs over time, export to CSV, export to gif, and
        # run just a single step.
        model = tribal_masking_model_classes.tribal_masking_model(p.world_shape)

        view = tribal_masking_visualization.tribal_masking_view(model)
        # tribal_masking_visualization.plot_combined_game_interactive_infections(model)


