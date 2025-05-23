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
import numpy as np
current_file_path = os.path.abspath(__file__)
current_folder = os.path.dirname(current_file_path)
baseDir = os.path.dirname(current_folder)
sys.path.append(baseDir)
opensimADDir = os.path.join(baseDir, 'UtilsDynamicSimulations', 'OpenSimAD')
sys.path.append(opensimADDir)

from utilsOpenSimAD import processInputsOpenSimAD, plotResultsOpenSimAD, createOpenCapFolderStructure
from mainOpenSimAD import run_tracking
from utilsAuthentication import get_token
from utilsProcessing import segment_gait, getCOP_masks, map_stance_phase
from utils import get_trial_id, download_trial
from utilsKineticsOpenSimAD import kineticsOpenSimAD
from utilsPlotting import plot_dataframe

os.chdir("C:\\Users\\MoBL3\\Documents\\GRF_Project\\Optimize_Feet")
from gait_analysis import process_gait_data

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

runTorqueDrivenProblem = True
runMuscleDrivenProblem = False
runComparison = False

trials = ['1', '2', '3']
subjects = ['2', '3' ,'4', '5', '6', '7', '8', '9', '10', '11']

trials = ['1']
subjects = ['2']


for subject in subjects:
    for walking_trial in trials:

        try:

            
                if subject == '11':
                    walking_trial = str(int(walking_trial) + 1)
                
                if subject == '4' and walking_trial == '3':  # Adjuste d index since Python is 0-based
                    walking_trial = '4'
                
                
                path_to_original_data = 'Y:\\Users\\EMiller\\Abstracts\\ASB2025\\Data'
                treadmill_speed = 0 # overground
                
                
                path_to_data = path_to_original_data + '\\subject' + subject + '\\walking' + walking_trial
               #ik_path = os.path.join(path_to_data,  'walking' + walking_trial + '_ik.mot')
                ik_path = f"C:\\Users\\MoBL3\\Documents\\GRF_Project\\LabValidation_withVideos\\subject{subject}\\OpenSimData\\Mocap\\IK\\walking{walking_trial}.mot"
                
                #grf_path = os.path.join(path_to_data,  'Predicted_GRF.mot')
                grf_path = f"C:\\Users\\MoBL3\\Documents\\GRF_Project\\LabValidation_withVideos\\subject{subject}\\ForceData\\walking{walking_trial}_forces.mot"
                #grf_path = ('C:\\Users\\MoBL3\\Documents\\GRF_Project\\opencap-processing-grf\\Data\\GRF_tracking_data\\' + 'subject' + subject + 'walking' + walking_trial + '\\ForceData\\MocapCOPx_MLPredictedGRFs.mot')
                metadata_path = os.path.join(path_to_data, 'original_files', 'sessionMetadata.yaml')
                model_path = os.path.join(path_to_data,  'LaiUhlrich2022_scaled.osim')
                
                session_id = 'subject' + subject + 'walking' + walking_trial
                
                # Insert the name of the trial you want to simulate.
                trial_name = 'walking' + walking_trial
                
                # Insert the type of activity you want to simulate. We have pre-defined settings
                # for different activities (more details above). Visit 
                # ./UtilsDynamicSimulations/OpenSimAD/settingsOpenSimAD.py to see all available
                # activities and their settings. If your activity is not in the list, select
                # 'other' to use generic settings or set your own settings.
                sim_setting_name = 'walking_nonperiodic_cop_tracking_MD'
                
                # GRF additions
                # get directory of this file
                # this_file_dir = os.path.dirname(os.path.abspath(__file__))
                dataFolder = 'C:\\Users\\MoBL3\\Documents\\GRF_Project\\opencap-processing-grf\\Data\\GRF_tracking_data'
                example_data_dir = os.path.join(dataFolder,session_id)
                if not os.path.exists(example_data_dir):
                    os.mkdir(example_data_dir)
                shutil.copy(metadata_path, example_data_dir)
                
                createOpenCapFolderStructure(example_data_dir,model_path,ik_path,metadata_path, trial_name, grf_path)
                
                session_dir = "C:/Users/MoBL3/Documents/GRF_Project/LabValidation_withVideos/subject" + subject # Path to session data directory
                lowpass_cutoff_frequency = 6  # Apply a lowpass filter with cutoff frequency of 6 Hz
                n_gait_cycles = 1  # Analyze the first 5 gait cycles
                gait_style = "overground"  # Specify the gait style, e.g., treadmill or overground
                trimming_start = 0  # Trim the first 2 seconds of data
                trimming_end = 0  # Trim the last 1 second of data
                name_trial = 'walking' + walking_trial
                

                mask_ips, mask_cont, gait_events, foot_positions, time = process_gait_data(session_dir, name_trial, 'l', lowpass_cutoff_frequency, n_gait_cycles, gait_style, trimming_start, trimming_end)
                
                time = gait_events['time']
                n_time = len(time)
                stance_percent_l, stance_percent_r = map_stance_phase(gait_events, time_len=n_time)

                
                time_window = [min(gait_events['time']), max(gait_events['time'])] # change this for periodic to go from rHs to rHs
                CoP_left_mask =  mask_ips
                CoP_right_mask =  mask_cont
                
                           
                # Sub-example 1: walking simulation with torque-driven model.
                
                # Prepare inputs for dynamic simulation (this will be skipped if already done):
                #   - Download data from OpenCap database
                #   - Adjust wrapping surfaces
                #   - Add foot-ground contacts
                #   - Generate external function (OpenSimAD)
        
                stiffness = 1000000 # defualt sphere stiffness
        
        
                settings = processInputsOpenSimAD(
                    baseDir, dataFolder, session_id, trial_name, sim_setting_name, 
                    time_window=time_window, stiffness=stiffness, treadmill_speed=treadmill_speed)
                
                #CoP_right_mask, CoP_left_mask = getCOP_masks(grf_path)
                
                settings['CoP_mask']["right"] = CoP_right_mask
                settings['CoP_mask']["left"] = CoP_left_mask
                settings['Stance_Precentage']["right"] = stance_percent_r
                settings['Stance_Precentage']["left"] = stance_percent_l
                settings['input_times'] = gait_events['time']
        


                case = 'Mocap IK newFT IDWeights cop10 pos1000 CE0 pelvistrack10' # Name case based on stiffness value
                settings['weights']['copMonotonicTerm'] = 0
                settings['weights']['copAccelerationTerm'] = 0
                settings['weights']['contrainCOPX_to_footMarkers'] = 0;
                settings['weights']['copTrackingTerm'] = 10
                
                settings['weights']['grfTrackingTerm'] = 0.001             
                settings['weights']['positionTrackingTerm'] = 1000
                settings['torque_driven_model']  =  True

                
                # Edits to try to make this close to ID
                settings['weights']['accelerationTrackingTerm'] = 0
                settings['weights']['pelvisResidualsTerm'] = 0.0000001
                settings[ 'allowPelvisResiduals'] = True
                settings['weights']['activationTerm'] = 0.00001
                settings['weights']['coordinateExcitationTerm'] = 0
                settings['weights']['jointAccelerationTerm'] =0
                settings['coordinates_toTrack']['lumbar_extension']['weight'] = 5
                settings['coordinates_toTrack']['lumbar_bending']['weight'] = 5
                settings['coordinates_toTrack']['lumbar_rotation']['weight'] = 5
                
                settings['coordinates_toTrack']['pelvis_tx']['weight'] = 5
                settings['coordinates_toTrack']['pelvis_ty']['weight'] = 5
                settings['coordinates_toTrack']['pelvis_tz']['weight'] = 5

            
                # % Run the dynamic simulation.
                run_tracking(baseDir, dataFolder, session_id, settings, foot_positions, case=case)
                
                # Plot results.
                plotResultsOpenSimAD(dataFolder, session_id, trial_name, settings, [case])
        except:
            print('Errored')
                



            