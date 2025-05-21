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
import itertools
from concurrent.futures import ProcessPoolExecutor

# === Required imports for simulation setup ===

current_file_path = os.path.abspath(__file__)
current_folder = os.path.dirname(current_file_path)
baseDir = os.path.dirname(current_folder)
sys.path.append(baseDir)
opensimADDir = os.path.join(baseDir, 'UtilsDynamicSimulations', 'OpenSimAD')
sys.path.append(opensimADDir)
sys.path.append("C:/Users/MoBL3/Documents/GRF_Project/Optimize_Feet")

from gait_analysis import process_gait_data
from utilsOpenSimAD import processInputsOpenSimAD, plotResultsOpenSimAD, createOpenCapFolderStructure
from mainOpenSimAD import run_tracking
from utilsProcessing import map_stance_phase



# === Constants ===
baseDir = "C:/Users/MoBL3/Documents/GRF_Project/Optimize_Feet"
dataFolder = "C:/Users/MoBL3/Documents/GRF_Project/opencap-processing-grf/Data/GRF_tracking_data"
subjects = ['2']
trials = ['1']

# === Core function for a single run ===
def run_simulation(subject, walking_trial, pelvis_on, pelvis_weight,
                   foot_torque_on, foot_torque_weight,
                   cop_track_weight, cop_accel_weight):
    case_name = f"PR_{pelvis_on}_{str(pelvis_weight).replace('.', '_')}__FT_{foot_torque_on}_{str(foot_torque_weight).replace('.', '_')}__COPtrack_{str(cop_track_weight).replace('.', '_')}__COPaccel_{str(cop_accel_weight).replace('.', '_')}"
    try:
        session_id = f'subject{subject}walking{walking_trial}'
        trial_name = f'walking{walking_trial}'

        path_to_original_data = 'Y:/Users/EMiller/Abstracts/ASB2025/Data'
        path_to_data = os.path.join(path_to_original_data, f'subject{subject}', f'walking{walking_trial}')

        ik_path = f"C:/Users/MoBL3/Documents/GRF_Project/LabValidation_withVideos/subject{subject}/OpenSimData/Mocap/IK/walking{walking_trial}.mot"
        grf_path = f"C:/Users/MoBL3/Documents/GRF_Project/LabValidation_withVideos/subject{subject}/ForceData/walking{walking_trial}_forces.mot"
        metadata_path = os.path.join(path_to_data, 'original_files', 'sessionMetadata.yaml')
        model_path = os.path.join(path_to_data, 'LaiUhlrich2022_scaled.osim')

        example_data_dir = os.path.join(dataFolder, session_id)
        os.makedirs(example_data_dir, exist_ok=True)
        shutil.copy(metadata_path, example_data_dir)
        createOpenCapFolderStructure(example_data_dir, model_path, ik_path, metadata_path, trial_name, grf_path)

        session_dir = f"C:/Users/MoBL3/Documents/GRF_Project/LabValidation_withVideos/subject{subject}"
        gait_data = process_gait_data(session_dir, trial_name, 'l', 6, 1, "overground", 0, 0)
        gait_time, foot_positions, time = gait_data[:3]  # Updated unpacking

        gait_events = gait_data[0] if isinstance(gait_data, tuple) and isinstance(gait_data[0], dict) else {}
        n_time = len(gait_time)
        stance_percent_l, stance_percent_r = map_stance_phase(gait_data, time_len=n_time)
        time_window = [min(gait_time), max(gait_time)]
        mask_ips = gait_data[3] if len(gait_data) > 3 else None
        mask_cont = gait_data[4] if len(gait_data) > 4 else None

        stiffness = 1000000
        treadmill_speed = 0

        sim_setting_name = f"walk_PR_{pelvis_on}_FT_{foot_torque_on}_{str(foot_torque_weight).replace('.', '_')}_COP_{str(cop_track_weight).replace('.', '_')}_CA_{str(cop_accel_weight).replace('.', '_')}"

        settings = processInputsOpenSimAD(
            baseDir, dataFolder, session_id, trial_name, sim_setting_name,
            time_window=time_window, stiffness=stiffness, treadmill_speed=treadmill_speed)

        settings['CoP_mask']['right'] = mask_cont
        settings['CoP_mask']['left'] = mask_ips
        settings['Stance_Precentage']['right'] = stance_percent_r
        settings['Stance_Precentage']['left'] = stance_percent_l
        settings['input_times'] = gait_time

        # === Parameterized Settings ===
        settings['weights']['copMonotonicTerm'] = 0
        settings['weights']['copAccelerationTerm'] = cop_accel_weight
        settings['weights']['copTrackingTerm'] = cop_track_weight
        settings['weights']['grfTrackingTerm'] = 0.001
        settings['weights']['pelvisResidualsTerm'] = pelvis_weight
        settings['allowPelvisResiduals'] = pelvis_on
        settings['weights']['positionTrackingTerm'] = 1000
        settings['weights']['footTorqueTerm'] = foot_torque_weight
        settings['foot_torque_actuator'] = foot_torque_on
        settings['torque_driven_model'] = True
        settings['weights']['contrainCOPX_to_footMarkers'] = 0
        settings['coordinates_toTrack']['lumbar_extension']['weight'] = 5
        settings['coordinates_toTrack']['lumbar_rotation']['weight'] = 5

        print(f"Running simulation: {case_name}")
        run_tracking(baseDir, dataFolder, session_id, settings, foot_positions, case=case_name)
        plotResultsOpenSimAD(dataFolder, session_id, trial_name, settings, [case_name])
        print(f"✅ Finished: {case_name}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error in case {case_name}: {e}")

# === Extracted for multiprocessing ===
def submit_combo(args):
    pelvis, ft, cop_track, cop_accel = args
    for subject in subjects:
        for trial in trials:
            print(f"Submitting case: Subject {subject}, Trial {trial}, Params: {pelvis}, {ft}, COP track: {cop_track}, COP accel: {cop_accel}")
            run_simulation(subject, trial,
                           pelvis[0], pelvis[1],
                           ft[0], ft[1],
                           cop_track, cop_accel)

# === Parameter Grid and Parallel Runner ===
def run_all():
    pelvis_opts = [(True, 0.001), (False, 0)]
    foot_torque_opts = [(True, 0.0), (True, 0.001), (False, 0.0)]
    cop_track_opts = [0.001, 0.1, 100]
    cop_accel_opts = [0.001, 0.1, 0]

    parameter_combos = list(itertools.product(pelvis_opts, foot_torque_opts, cop_track_opts, cop_accel_opts))

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(submit_combo, args) for args in parameter_combos]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"❌ A simulation task failed: {e}")

if __name__ == "__main__":
    run_all()
