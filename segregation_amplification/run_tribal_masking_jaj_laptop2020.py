import os
import hazelbean as hb
import tribal_masking_tasks
import sys
from PyQt5 import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


project_name = 'fifth_attempt'
project_dir = os.path.join('../../projects', project_name)
p = hb.ProjectFlow(project_dir)

p.L = hb.get_logger('tribal_masking')
p.world_shape = (150, 150)

aspatial_externality_game_task = p.add_task(tribal_masking_tasks.aspatial_externality_game,                                           run=0, skip_if_dir_exists=1)
combined_game_with_policies_and_infection_task = p.add_task(tribal_masking_tasks.combined_game_with_policies_and_infection,           run=1, skip_if_dir_exists=0)

p.execute()

if __name__ == "__main__":
    import sys
    # create a QApplication object
    
    
    app = QApplication(sys.argv)
    window = Start()
    app.exec_()