'''
    ---------------------------------------------------------------------------
    OpenCap processing: example_walking_opensimAD_GRF.py
    ---------------------------------------------------------------------------
    Copyright 2023 Stanford University and the Authors
    Original Author(s): Antoine Falisse
    Author of Updated GRF Tracking Algorithm: Emily Miller
    
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



# %% Directories, paths, and imports. You should not need to change anything.
import os
import sys
import numpy as np
baseDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(baseDir)
opensimADDir = os.path.join(baseDir, 'UtilsDynamicSimulations', 'OpenSimAD')
sys.path.append(opensimADDir)

from utilsOpenSimAD import processInputsOpenSimAD, plotResultsOpenSimAD
from mainOpenSimAD import run_tracking
from gait_analysis import process_gait_data
from utilsProcessing import map_stance_phase

# ------------------------ USER INPUTS ---------------------------------------
# Insert your session ID here. This should be automaically generated after 
# running the batchDownload.py script and should be located in the folder
# opencap-processing-grf/Data/
# If your data is somewhere else you can change the filepath in the dataFolder
#  variables below
# Insert the path to where you want the data has been downloaded.
dataFolder = os.path.join(baseDir, 'Data')

# Give the sessionID
session_id = "OpenCapData_6cc72044-d576-4073-865d-b0e2325655b5"

# Insert the name of the trial you want to simulate.
trial_name = 'walk_Optimized'

# Insert the type of activity you want to simulate
motion_type = 'walking_grf' # this will call the specific settings for GRF tracking

# Give the path to your GRF Data, the path below is automatically generated in the batchDownload_ForGRFTrack script
grf_path = os.path.join(dataFolder, session_id, 'ForceData', trial_name + '_forces.mot')

# Insert the time interval you want to simulate. It is recommended to simulate
# trials shorter than 2s (more details above). Set to [] to simulate full trial.
# You can use the process_gait_data function to automatically segment the 
# gait cycles first on the whole trial (will be stored in gait_events) and then 
# run again with inputs to trim the data based on gait_events output. 
# Also insert the speed of the treadmill # in m/s. A positive value 
# indicates that the subject is moving forward. You should set it to 0 
# if the trial was not measured on a treadmill
lowpass_cutoff_frequency = 6  # Apply a lowpass filter with cutoff frequency of 6 Hz
n_gait_cycles = 2  # Analyze the first 2 gait cycles, set to -1 for all gait cycles
gait_style = "overground"  # Specify the gait style, e.g., treadmill or overground
trimming_start = 0  # optionally trim data at beginning, if you already trimmed in the batchDownload script this is unneccesary
trimming_end = 0     # optionally trim data at end
treadmill_speed = 0
# ------------------------ END USER INPUTS -----------------------------------


# Make sure masks exactly match stance percentages
mask_ips, mask_cont, gait_events, foot_positions, time = process_gait_data(os.path.join(dataFolder, session_id),
        trial_name, 'l', lowpass_cutoff_frequency, n_gait_cycles, gait_style, trimming_start, trimming_end)
n_time = len(time)
stance_percent_l, stance_percent_r = map_stance_phase(gait_events, time_len=n_time)
CoP_left_mask  = np.isfinite(stance_percent_l).astype(int)
CoP_right_mask  = np.isfinite(stance_percent_r).astype(int)         

# %% Example 1: walking simulation with torque-driven model.
# Insert a string to "name" you case.
case = 'torque_driven_grf'

runTorqueDrivenProblem = True

# Prepare inputs for dynamic simulation (this will be skipped if already done):
#   - Download data from OpenCap database
#   - Adjust wrapping surfaces
#   - Add foot-ground contacts
#   - Generate external function (OpenSimAD)
time_window = [min(gait_events['time']), max(gait_events['time'])] 
settings = processInputsOpenSimAD(
    baseDir, dataFolder, session_id, trial_name, motion_type, 
    time_window=time_window, treadmill_speed=treadmill_speed)     

# Adjust settings for this example.
# Set the model to be torque-driven.
settings['torque_driven_model'] = runTorqueDrivenProblem
# Add inputs to the objective function
settings['CoP_mask']["right"] = CoP_right_mask
settings['CoP_mask']["left"] = CoP_left_mask
settings['Stance_Precentage']["right"] = stance_percent_r
settings['Stance_Precentage']["left"] = stance_percent_l
settings['input_times'] = gait_events['time'] 
            

# You can add periodic constraints to the problem. This will constrain initial and
# final states of the problem to be the same. This is useful for obtaining
# faster convergence. Please note that the walking trial we selected might not
# is not periodic. We here add periodic constraints to show how to do it.
# We add periodic constraints here for the coordinate values (coordinateValues)
# and coordinate speeds (coordinateSpeeds) of the lower-extremity joints
# (lowerLimbJoints). We also add periodic constraints for the activations of the
# ideal torque actuators at the lower-extremity (lowerLimbJointActivations) and
# lumbar (lumbarJointActivations) joints. 
# settings['periodicConstraints'] = {
#     'coordinateValues': ['lowerLimbJoints'],
#     'coordinateSpeeds': ['lowerLimbJoints'],
#     'lowerLimbJointActivations': ['all'],
#     'lumbarJointActivations': ['all']}

# Filter the data to be tracked. We here filter the coordinate values (Qs) with
# a 6 Hz (cutoff_freq_Qs) low-pass filter, the coordinate speeds (Qds) with a 6
# Hz (cutoff_freq_Qds) low-pass filter, and the coordinate accelerations (Qdds)
# with a 6 Hz (cutoff_freq_Qdds) low-pass filter. We also compute the coordinate
# accelerations by first splining the coordinate speeds (splineQds=True) and
# then taking the first time derivative (default is to spline the coordinate
# values and then take the second time derivative).
settings['filter_Qs_toTrack'] = True
settings['cutoff_freq_Qs'] = 6
settings['filter_Qds_toTrack'] = True
settings['cutoff_freq_Qds'] = 6
settings['filter_Qdds_toTrack'] = True
settings['cutoff_freq_Qdds'] = 6
settings['splineQds'] = True

# We set the mesh density to 50. We recommend using a mesh density of 100 by
# default, but we here use a lower value to reduce the computation time.
settings['meshDensity'] = 50

# Run the dynamic simulation.
run_tracking(baseDir, dataFolder, session_id, settings, case=case)
# Plot some results.
# Comment plotResultsOpenSimAD out if running a loop to batch process data
plotResultsOpenSimAD(dataFolder, session_id, trial_name, settings, [case])


