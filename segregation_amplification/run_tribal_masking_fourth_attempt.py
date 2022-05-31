import os
import hazelbean as hb
import tribal_masking_tasks

project_name = 'fourth_attempt'
project_dir = os.path.join('../../projects', project_name)
p = hb.ProjectFlow(project_dir)

p.L = hb.get_logger('tribal_masking')

aspatial_externality_game_task = p.add_task(tribal_masking_tasks.aspatial_externality_game)
combined_game_with_policies_and_infection_task = p.add_task(tribal_masking_tasks.combined_game_with_policies_and_infection)

aspatial_externality_game_task.run = 1
combined_game_with_policies_and_infection_task.run = 1

p.execute()

