import os
import hazelbean as hb
import tribal_masking_tasks

project_name = 'fourth_attempt'
project_dir = os.path.join('../../projects', project_name)


p = hb.ProjectFlow(project_dir)
p.conda_executable_path = "C:\Miniconda3\envs\research2022d\python.exe"

p.L = hb.get_logger('tribal_masking')
p.world_shape = (100, 100)

aspatial_externality_game_task = p.add_task(tribal_masking_tasks.aspatial_externality_game,                                           run=0)
manual_model_call_task = p.add_task(tribal_masking_tasks.manual_model_call,                                                           run=0)
testing_different_infection_probabilities_task = p.add_task(tribal_masking_tasks.testing_different_infection_probabilities,           run=1)
effect_of_segregation_on_masking_behavior_task = p.add_task(tribal_masking_tasks.effect_of_segregation_on_masking_behavior,           run=0)
combined_game_with_policies_and_infection_task = p.add_task(tribal_masking_tasks.combined_game_with_policies_and_infection,           run=0)

p.execute()

