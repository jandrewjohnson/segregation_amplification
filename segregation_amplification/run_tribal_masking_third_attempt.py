import os
import hazelbean as hb
# from hazelbean.project_flow import ProjectFlow
import tribal_masking_tasks

project_name = 'third_attempt'
project_dir = os.path.join('../../projects', project_name)
p = hb.ProjectFlow(project_dir)

p.L = hb.get_logger('tribal_masking')

aspatial_externality_game_task = p.add_task(tribal_masking_tasks.aspatial_externality_game)
spatial_externality_game_task = p.add_task(tribal_masking_tasks.spatial_externality_game)
spatial_segregation_game_task = p.add_task(tribal_masking_tasks.spatial_segregation_game)
combined_game_noninteractive_task = p.add_task(tribal_masking_tasks.combined_game_noninteractive)
combined_game_interactive_full_resolve_task = p.add_task(tribal_masking_tasks.combined_game_interactive_full_resolve)
combined_game_interactive_time_variant_task = p.add_task(tribal_masking_tasks.combined_game_interactive_time_variant)
test_different_data_models_performance_task = p.add_task(tribal_masking_tasks.test_different_data_models_performance)
combined_game_with_policies_task = p.add_task(tribal_masking_tasks.combined_game_with_policies)
combined_game_with_policies_and_infection_task = p.add_task(tribal_masking_tasks.combined_game_with_policies_and_infection)

aspatial_externality_game_task.run = 0
spatial_externality_game_task.run = 0
spatial_segregation_game_task.run = 0
combined_game_noninteractive_task.run = 0
combined_game_interactive_full_resolve_task.run = 0
combined_game_interactive_time_variant_task.run = 0
test_different_data_models_performance_task.run = 1
combined_game_with_policies_task.run = 1
combined_game_with_policies_and_infection_task.run = 0

aspatial_externality_game_task.skip_existing = 0
spatial_externality_game_task.skip_existing = 0
spatial_segregation_game_task.skip_existing = 0
combined_game_noninteractive_task.skip_existing = 0
combined_game_interactive_full_resolve_task.skip_existing = 0
combined_game_interactive_time_variant_task.skip_existing = 0
test_different_data_models_performance_task.skip_existing = 0
combined_game_with_policies_task.skip_existing = 0
combined_game_with_policies_and_infection_task.skip_existing = 0

p.execute()

