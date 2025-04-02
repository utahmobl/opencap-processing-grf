'''
    ---------------------------------------------------------------------------
    OpenCap processing: example_walking_opensimAD.py
    ---------------------------------------------------------------------------
    Copyright 2023 Stanford University and the Authors
    
    Author(s): Antoine Falisse
    
    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    
    This code makes use of CasADi, which is licensed under LGPL, Version 3.0;
    https://github.com/casadi/casadi/blob/master/LICENSE.txt.
    
    Install requirements:
        - Visit https://github.com/stanfordnmbl/opencap-processing for details.        
        - Third-party software packages:
            - Windows
                - Visual studio: https://visualstudio.microsoft.com/downloads/
                    - Make sure you install C++ support
                    - Code tested with community editions 2017-2019-2022
            - Linux
                - OpenBLAS libraries
                    - sudo apt-get install libopenblas-base
            
    Please contact us for any questions: https://www.opencap.ai/#contact

    This example shows how to run dynamic simulations of walking using
    data collected with OpenCap. The code uses OpenSimAD, a custom version
    of OpenSim that supports automatic differentiation (AD).

    This example is made of two sub-examples. The first one uses a torque-driven
    musculoskeletal model and the second one uses a muscle-driven
    musculoskeletal model. Please note that both examples are meant to
    demonstrate how to run dynamic simualtions and are not meant to be 
    biomechanically valid. We only made sure the simulations converged to
    solutions that were visually reasonable. You can find more examples of
    dynamic simulations using OpenSimAD in example_kinetics.py.
'''

# %% Select the example you want to run.





# % Directories, paths, and imports. You should not need to change anything.
import os
import sys
import shutil
current_file_path = os.path.abspath(__file__)
current_folder = os.path.dirname(current_file_path)
baseDir = os.path.dirname(current_folder)
sys.path.append(baseDir)
opensimADDir = os.path.join(baseDir, 'UtilsDynamicSimulations', 'OpenSimAD')
sys.path.append(opensimADDir)

from utilsOpenSimAD import processInputsOpenSimAD, plotResultsOpenSimAD, createOpenCapFolderStructure
from mainOpenSimAD import run_tracking
from utilsProcessing import segment_STS

# OpenCap authentication. Visit https://app.opencap.ai/login to create an
# account if you don't have one yet.
# get_token(saveEnvPath=os.getcwd())

# Outputs common to both examples.

# Insert your session ID   here. You can find the ID of all your sessions at
# https://app.opencap.ai/sessions.
# Visit https://app.opencap.ai/session/<session_id> to visualize the data of
# your session.
# Either opencap session id, or folder name with data
# session_id = "4d5c3eb1-1a59-4ea1-9178-d3634610561c"


path_to_data = 'Y:/Users/EMiller/SitToStand_Sims/demo2/'
ik_path = os.path.join(path_to_data, 'OpenSim', 'IK', 'shiftedIK', 'STS1_5_sync.mot') # Note this has to be a metadata file with the LaiUhlrich2022 model name
metadata_path =  os.path.join(path_to_data, 'OpenSim', 'sessionMetadata.yaml')# this doesn't exist in your file structure, will need to add it somewhere
model_path = os.path.join(path_to_data, 'OpenSim', 'Model', 'STS1_5', 'LaiUhlrich2022_scaled.osim')
grf_path = None

# segment sts and get times for analysis
STS_times = segment_STS(ik_path, pelvis_ty=None, timeVec=None, velSeated=0.3,
               velStanding=0.15, visualize=False, filter_pelvis_ty=True, 
               cutoff_frequency=4, delay=0.1)
# this outputs a bunch of times for STS, for now just going to try running the 
# first of them but should probably write a loop to run multiple of these here    
time_window = STS_times[2][0]



# edit this within loop to differentiate between sts trials
session_id = 'subject2' + 'sit2stand_demo' # probably want to make this more specific for each subject/ s2s trial etc...
trial_name = 'sit2stand_demo' # session id and trial name are all from the loop I had going originally to run the whole dataset, adapt as needed for your data
dataFolder = os.path.join(path_to_data,'OpenSim', 'Simulations')
if not os.path.exists(dataFolder):
    os.mkdir(dataFolder)
example_data_dir = os.path.join(dataFolder,session_id)
if not os.path.exists(example_data_dir):
    os.mkdir(example_data_dir)
shutil.copy(metadata_path, example_data_dir)


createOpenCapFolderStructure(example_data_dir,model_path,ik_path,metadata_path, trial_name, grf_path)

          
# Prepare inputs for dynamic simulation (this will be skipped if already done):
#   - Download data from OpenCap database
#   - Adjust wrapping surfaces
#   - Add foot-ground contacts
#   - Generate external function (OpenSimAD)
sim_setting_name = 'sit_to_stand' 
treadmill_speed = 0 # overground
settings = processInputsOpenSimAD(
    baseDir, dataFolder, session_id, trial_name, sim_setting_name, 
    time_window=time_window, treadmill_speed=treadmill_speed)

case = 'case2' # Insert a string to "name" your case. This is useful when running 
# multiple simulations on the same data but with different weights or conditions

# Run the dynamic simulation.
run_tracking(baseDir, dataFolder, session_id, settings, case=case)
# Plot some results.
plotResultsOpenSimAD(dataFolder, session_id, trial_name, settings, [case])
test = 1