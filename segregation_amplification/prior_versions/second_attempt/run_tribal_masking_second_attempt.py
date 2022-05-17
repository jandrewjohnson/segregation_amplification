import os
import hazelbean as hb
import tasks

project_name = 'second_attempt'
project_dir = os.path.join('../../projects', project_name)
p = hb.ProjectFlow(project_dir)

L = hb.get_logger('tribal_masking')

aspatial_externality_game_task = p.add_task(tasks.aspatial_externality_game)
spatial_externality_game_task = p.add_task(tasks.spatial_externality_game)
spatial_segregation_game_task = p.add_task(tasks.spatial_segregation_game)
combined_game_noninteractive_task = p.add_task(tasks.combined_game_noninteractive)
combined_game_interactive_full_resolve_task = p.add_task(tasks.combined_game_interactive_full_resolve)
combined_game_interactive_time_variant_task = p.add_task(tasks.combined_game_interactive_time_variant)

aspatial_externality_game_task.run = 1
spatial_externality_game_task.run = 1
spatial_segregation_game_task.run = 1
combined_game_noninteractive_task.run = 1
combined_game_interactive_full_resolve_task.run = 1
combined_game_interactive_time_variant_task.run = 1

aspatial_externality_game_task.skip_existing = 1
spatial_externality_game_task.skip_existing = 1
spatial_segregation_game_task.skip_existing = 1
combined_game_noninteractive_task.skip_existing = 1
combined_game_interactive_full_resolve_task.skip_existing = 1
combined_game_interactive_time_variant_task.skip_existing = 0

p.execute()

