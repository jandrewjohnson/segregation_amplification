import os
import sys
import random
import pandas as pd
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
recompile_cython = False
if recompile_cython == True:
    old_cwd = os.getcwd()
    b = os.listdir(old_cwd)
    conda_executable_path = "C://Miniconda3//envs//research2022d//python.exe"
    cython_command = conda_executable_path + " compile_cython_functions.py build_ext -i clean"  #
    print('Running cythonization command: ' + cython_command)
    returned = os.system(cython_command)
    # if returned:
        # raise NameError('Cythonization failed.')

import tribal_masking_model_classes
# import tribal_masking_computational_core
import tribal_masking_tasks
import tribal_masking_view_classes

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

def manual_model_call(p):
    
    if p.run_this:     
        
        # Initialize parameters
        n_iterations_per_step = 1
        n_runs = 20
        n_steps = 400
        
        # Check to see if we need to save simulation outputs to CSV if they don't exist.
        n_infections_path = os.path.join(p.cur_dir, 'n_infections.csv')
        if not hb.path_exists(n_infections_path):
            
            # Store the results of each run in a column of a numpy array (for later use in Pandas or plotting)
            n_infections_by_run_array = np.zeros((n_steps, n_runs), dtype=np.int64)
            
            # Iterate through each run
            for run_id in range(n_runs):

                # (re)-create the model. If you don't call this on each run_id, it will just pickup where the model object last was.
                model = tribal_masking_model_classes.TribalMaskingModel(p.world_shape)
                
                # For this plot, we are going to just set the infection manually. But the commented out line shows how you could modify it.
                model.params['infection_probability'] = 0.009  # Default is .005
                # model.params['infection_probability'] = 0.0025 * (run_id + 1) # Default is .005
                
                # Run the model for n_steps           
                for i in range(n_steps):
                    
                    # On each step, update the model through n_iterations
                    model.update(n_iterations_per_step)
                    
                    # Count how many are infected at this stage
                    n_infected = float(np.sum(np.where(model.infection_status > 0, 1.0, 0.0)))
                    
                    # Write the result to the output array. If needed, this could be optimized through writing a cython multi-model loop call.
                    n_infections_by_run_array[i, run_id] = n_infected
                    
                    print('n_infected: ' + str(n_infected) + ' on step ' + str(i))

            # Convert array to a df
            df = pd.DataFrame(data=n_infections_by_run_array, columns=list(range(n_runs)))

            # Write the df to a CSV
            df.to_csv(n_infections_path, index=False)  
             
        else: # Results do exist so just load them.
            df = pd.read_csv(n_infections_path)
            n_infections_by_run_array = df.values
        
        # Check if n_infections png exists. Skip if it does.
        n_infections_plot_path = os.path.join(p.cur_dir, 'n_infections.png')    
        if not hb.path_exists(n_infections_plot_path):
            
            # iterate through each n_runs to plot to a scatter array as a different color.
            for i in range(n_runs):
                to_plot = n_infections_by_run_array[:, i]
                plt.scatter(list(range(n_steps)), to_plot, s=25, c='blue', alpha=.05, edgecolors='none')
                
            plt.savefig(n_infections_plot_path, bbox_inches='tight')


def testing_different_infection_probabilities(p):
    """THIS IS BROKEN because I didn't figure out how to create a scatters' legend."""
    if p.run_this:     
        
        # Initialize parameters
        n_iterations_per_step = 1
        n_runs = 10
        n_steps = 400
        
        # Check to see if we need to save simulation outputs to CSV if they don't exist.
        n_infections_path = os.path.join(p.cur_dir, 'n_infections_by_infectiousness.csv')
        if not hb.path_exists(n_infections_path):
            
            # Store the results of each run in a column of a numpy array (for later use in Pandas or plotting)
            n_infections_by_run_array = np.zeros((n_steps, n_runs), dtype=np.int64)
            infection_rates = []
            
            # Iterate through each run
            for run_id in range(n_runs):

                # (re)-create the model. If you don't call this on each run_id, it will just pickup where the model object last was.
                model = tribal_masking_model_classes.TribalMaskingModel(p.world_shape)
                
                # For this plot, we are going to just set the infection manually. But the commented out line shows how you could modify it.
                model.params['infection_probability'] = 0.01 * ((run_id + 1)/5) # Default is .005
                
                # Save the parameters to a list for later plotting
                infection_rates.append(model.params['infection_probability'])
                
                # Run the model for n_steps           
                for i in range(n_steps):
                    
                    # On each step, update the model through n_iterations
                    model.update(n_iterations_per_step)
                    
                    # Count how many are infected at this stage
                    n_infected = float(np.sum(np.where(model.infection_status > 0, 1.0, 0.0)))
                    
                    # Write the result to the output array. If needed, this could be optimized through writing a cython multi-model loop call.
                    n_infections_by_run_array[i, run_id] = n_infected
                    
                    print('n_infected: ' + str(n_infected) + ' on step ' + str(i))


           
            # Convert array to a df
            df = pd.DataFrame(data=n_infections_by_run_array, columns=list(range(n_runs)))
                        
            # Write the df to a CSV
            df.to_csv(n_infections_path, index=False)  
             
        else: # Results do exist so just load them.
            df = pd.read_csv(n_infections_path)
            n_infections_by_run_array = df.values
            infection_rates = [0.0025 * (int(i) + 1) for i in list(df.columns)]
            
        # Check if n_infections png exists. Skip if it does.
        n_infections_plot_path = os.path.join(p.cur_dir, 'n_infections_by_infectiousness.png')    
        if not hb.path_exists(n_infections_plot_path) or 1:
            
            # iterate through each n_runs to plot to a scatter array as a different color.
            fig, ax = plt.subplots()
            scatters = []
            for i in range(n_runs):
                to_plot = n_infections_by_run_array[:, i] / 7500 * 100
                
                infectiousness = infection_rates[i]
                scatter = ax.plot(list(range(n_steps)), to_plot, alpha=.9, label=hb.round_significant_n(infectiousness, 2))                
                # scatter = ax.scatter(list(range(n_steps)), to_plot, s=10, alpha=.9, edgecolors='none', label=infectiousness)                
                scatters.append(scatter)
                
            # legend = ax.legend(loc="lower left", title="Classes")
            # ax.add_artist(legend)
        
            legend = ax.legend()
            legend.set_title("Infectiousness", prop = {'size':10})
            
            ax.set_xlabel('Days')
            ax.set_ylabel('Percent infected')
            
            # ax.axis('off')   
            # produce a legend with a cross section of sizes from the scatter
            # handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
            # legend = ax.legend(handles, labels, loc="upper right", title="Sizes")


            plt.savefig(n_infections_plot_path, bbox_inches='tight')

def effect_of_segregation_on_masking_behavior(p):
    
    if p.run_this:    

        fig_path = os.path.join(p.cur_dir, 'simple_masking_and_types_plot.png')
        if not hb.path_exists(fig_path):
        
            print('Creating fig at ' + os.path.abspath(fig_path))
            
            # Create the model
            hb.timer('Create model.')
            
            # world_shape = (1000, 1000)
            world_shape = (150, 150)
            
            model = tribal_masking_model_classes.TribalMaskingModel(world_shape)    
            
            n_iterations = 40
            model.update(n_iterations) 
            hb.timer('Model finished.')
            
            fig = tribal_masking_view_classes.generate_simple_masking_and_types_plot(model)
            fig.savefig(fig_path, bbox_inches='tight')

        fig_path = os.path.join(p.cur_dir, 'time_plot_of_segregation_convergence.png')
        if not hb.path_exists(fig_path):        
            print('Creating fig at ' + os.path.abspath(fig_path))
            
            # Create the model
            world_shape = (150, 150)            
            model = tribal_masking_model_classes.TribalMaskingModel(world_shape)    
                            
            fig = tribal_masking_view_classes.generate_time_plot_of_segregation_convergence_plot(model)
            fig.show()
            fig.savefig(fig_path, bbox_inches='tight')

        fig_path = os.path.join(p.cur_dir, 'time_plot_of_masking_convergence.png')
        if not hb.path_exists(fig_path):        
            print('Creating fig at ' + os.path.abspath(fig_path))
            
            # Create the model
            world_shape = (150, 150)            
            model = tribal_masking_model_classes.TribalMaskingModel(world_shape)    
                            
            fig = tribal_masking_view_classes.generate_time_plot_of_masking_convergence_plot(model)
            # fig.show()
            fig.savefig(fig_path, bbox_inches='tight')

        print('model', model)
            
def combined_game_with_policies_and_infection(p):

    if p.run_this:

        # Create the model
        model = tribal_masking_model_classes.TribalMaskingModel(p.world_shape)

        # Assign and launch a viewer of the model.
        view = tribal_masking_view_classes.TribalMaskingView(model)


