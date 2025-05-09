import os, sys
import hazelbean as hb

recompile_cython = 1
if recompile_cython:
    cython_command = sys.executable + " compile_cython_functions.py --quiet build_ext -i clean"  #
    print('cython_command', cython_command)
    # cython_command = "python " + python_file_uri + " --quiet build_ext -i clean"  #
    # cython_command = "python " + python_file_uri + " --verbose build_ext -i clean"  #
    returned = os.system(cython_command)
    if returned:
        print('Cythonization failed.')
        
import tribal_masking_tasks

project_name = 'fourth_attempt'
project_dir = os.path.join('../../projects', project_name)

p = hb.ProjectFlow(project_dir)

p.L = hb.get_logger('tribal_masking')
p.world_shape = (100, 100)



aspatial_externality_game_task = p.add_task(tribal_masking_tasks.aspatial_externality_game,                                           run=0)
manual_model_call_task = p.add_task(tribal_masking_tasks.manual_model_call,                                                           run=1)
testing_different_infection_probabilities_task = p.add_task(tribal_masking_tasks.testing_different_infection_probabilities,           run=1)
effect_of_segregation_on_masking_behavior_task = p.add_task(tribal_masking_tasks.effect_of_segregation_on_masking_behavior,           run=1)
combined_game_with_policies_and_infection_task = p.add_task(tribal_masking_tasks.combined_game_with_policies_and_infection,           run=1)

p.execute()

