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
runTorqueDrivenProblem = True
runMuscleDrivenProblem = False
runComparison = False

# %% Directories, paths, and imports. You should not need to change anything.
import os
import sys
baseDir = os.path.join(os.getcwd(), '..')
sys.path.append(baseDir)
opensimADDir = os.path.join(baseDir, 'UtilsDynamicSimulations', 'OpenSimAD')
sys.path.append(opensimADDir)

from utilsOpenSimAD import processInputsOpenSimAD, plotResultsOpenSimAD, createOpenCapFolderStructure
from mainOpenSimAD import run_tracking
from utilsAuthentication import get_token
from utilsProcessing import segment_gait
from utils import get_trial_id, download_trial
from utilsKineticsOpenSimAD import kineticsOpenSimAD
from utilsPlotting import plot_dataframe

# %% OpenCap authentication. Visit https://app.opencap.ai/login to create an
# account if you don't have one yet.
# get_token(saveEnvPath=os.getcwd())

# %% Inputs common to both examples.

# Insert your session ID here. You can find the ID of all your sessions at
# https://app.opencap.ai/sessions.
# Visit https://app.opencap.ai/session/<session_id> to visualize the data of
# your session.
# Either opencap session id, or folder name with data
# session_id = "4d5c3eb1-1a59-4ea1-9178-d3634610561c"
session_id = 'OC_val_mocap'

# Insert the name of the trial you want to simulate.
trial_name = 'walking1'

# Insert the type of activity you want to simulate. We have pre-defined settings
# for different activities (more details above). Visit 
# ./UtilsDynamicSimulations/OpenSimAD/settingsOpenSimAD.py to see all available
# activities and their settings. If your activity is not in the list, select
# 'other' to use generic settings or set your own settings.
sim_setting_name = 'walking_periodic_grf_tracking'

# GRF additions
# get directory of this file
this_file_dir = os.path.dirname(os.path.abspath(__file__))
dataFolder = os.path.join(this_file_dir,'..','Data','GRF_tracking_data')
example_data_dir = os.path.join(dataFolder,session_id)

model_path = os.path.join(example_data_dir,'LaiUhlrich2022_scaled.osim')
ik_path = os.path.join(example_data_dir,'walking1.mot') # needs to be trialname.mot
grf_path = os.path.join(example_data_dir,'walking1_forces.mot')
createOpenCapFolderStructure(example_data_dir,model_path,ik_path)

# time_window = [0.02, 1.09] # rHS to rHS - opencap
time_window = [.133, 1.23] # mocap 
treadmill_speed = 0 # overground


    
# %% Sub-example 1: walking simulation with torque-driven model.
# Insert a string to "name" you case.
case = 'case14'

# Prepare inputs for dynamic simulation (this will be skipped if already done):
#   - Download data from OpenCap database
#   - Adjust wrapping surfaces
#   - Add foot-ground contacts
#   - Generate external function (OpenSimAD)
settings = processInputsOpenSimAD(
    baseDir, dataFolder, session_id, trial_name, sim_setting_name, 
    time_window=time_window, treadmill_speed=treadmill_speed)

# Run the dynamic simulation.
run_tracking(baseDir, dataFolder, session_id, settings, case=case)
# Plot some results.
plotResultsOpenSimAD(dataFolder, session_id, trial_name, settings, [case])

test = 1