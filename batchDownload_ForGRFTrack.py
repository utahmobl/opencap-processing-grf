"""
---------------------------------------------------------------------------
OpenCap processing: batchDownload.py
---------------------------------------------------------------------------

Copyright 2022 Stanford University and the Authors

Author(s): Emily Miller, Antoine Falisse, Scott Uhlrich

Licensed under the Apache License, Version 2.0 (the "License"); you may not
use this file except in compliance with the License. You may obtain a copy
of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
import shutil
import glob
from pathlib import Path  # only used for HF output file
from gradio_client import Client, handle_file

# -------------------------------------------------------------------------
# Paths and imports
# -------------------------------------------------------------------------

# Base directory for this repository (opencap-processing-grf folder)
baseDir = os.path.dirname(os.path.abspath(__file__))

# Make repo root importable
sys.path.append(baseDir)

# Add OpenSim pipeline utilities
foot_opt_path = os.path.join(baseDir, "OpenSimPipeline", "ForGaitDynamics")
sys.path.append(foot_opt_path)

from utils import (  # noqa: E402
    download_session,
    make_long_ik,
    rename_grf_mot_columns,
    trim_mot_file,
    create_rajogopal_armless_model,
    create_LaiUhlrich_model,
    run_ik_for_gait_dynamics,
    reformat_ik_mot,
    harmonize_trc_markers_to_template,
    close_all_loggers,
    find_static_trial,
    convert_all_trc_in_folder_to_mm,
)
from FootContactOptimizer import refine_foot_kinematics_for_session  
from COP_Predictor import predict_cop_for_trial 

# ------------------------ USER INPUTS ---------------------------------------
# List of sessions to process.
# These correspond to OpenCap session IDs or subject folder names:
# "OpenCapData_<session_id>" inside the Data directory.
sessionList = ["6cc72044-d576-4073-865d-b0e2325655b5"]

# Gait style for foot kinematics refinement ("overground" or "treadmill")
gait_style = "overground"

# Treadmill speed in m/s. Set to zero for overground data.
treadmill_speed = 0.0

# Base directory for downloads and processed data
# Data will live in baseDir/Data/OpenCapData_<session_id>
downloadPath = os.path.join(baseDir, "Data")

# Only process trials whose marker files start with this prefix.
# Use "" to process all .trc files.
trial_prefix = "walk"
# ------------------------ END USER INPUTS -----------------------------------


for session_id in sessionList:
    print("\n" + "-" * 72)
    print(f"Processing session: {session_id}")
    print("-" * 72)
    
    # download your data from the OpenCap server
    # If only interested in marker and OpenSim data, downloadVideos=False will be faster.
    # Uncomment to let this script handle downloading:
    # If data is already downloaded or downloaded and trimmed can leave commented out    
    download_session(
        session_id,
        sessionBasePath=downloadPath,
        downloadVideos=False,
        trial_prefix=trial_prefix,
    )

    # Session folder with 'OpenCap_' export
    session_folder = os.path.join(downloadPath, f"OpenCapData_{session_id}")

    # Create scaled Rajogopal armless model for GaitDynamics input
    print("Creating scaled Armless Ragagopal model for GaitDynamics input")  
    convert_all_trc_in_folder_to_mm(os.path.join(session_folder, 'MarkerData'))     
    static_trc = find_static_trial(session_folder)  
 
    for_gait_dir = os.path.join(session_folder, "ForGaitDynamics\\")
    os.makedirs(for_gait_dir, exist_ok=True)
    scaled_RAmodel, height, mass = create_rajogopal_armless_model(
        generic_model_path=os.path.join(
            baseDir,
            "OpenSimPipeline",
            "ForGaitDynamics",
            "RajogopalArmless_Generic.osim",
        ),
        generic_scale_setup_xml=os.path.join(
            baseDir,
            "OpenSimPipeline",
            "ForGaitDynamics",
            "Setup_RajogopalArmless_Scaling_generic.xml",
        ),
        session_metadata_path=os.path.join(session_folder, "sessionMetadata.yaml"),
        static_trc_path=static_trc,
        output_dir=for_gait_dir,
    )
    
    # Create LaiUhlrich Model with correct markers
    scaled_laimodel, height, mass = create_LaiUhlrich_model(
        generic_model_path=os.path.join(
            baseDir,
            "OpenSimPipeline",
            "ForGaitDynamics",
            "LaiUhlrich2022_Generic.osim",
        ),
        generic_scale_setup_xml=os.path.join(
            baseDir,
            "OpenSimPipeline",
            "ForGaitDynamics",
            "Setup_LaiUhlrich_Scaling_generic.xml",
        ),
        session_metadata_path=os.path.join(session_folder, "sessionMetadata.yaml"),
        static_trc_path=static_trc,
        output_dir=os.path.join(session_folder, 'OpenSimData', 'Model')
    )
    
    
    # Run kinematics refinement for better foot contact
    #    (only for walking / level gait, not for all tasks)
    refine_foot_kinematics_for_session(
         session_folder=session_folder,
         trial_prefix=trial_prefix,          
         gait_style=gait_style, 
         trimming_start=0, # remove the first X seconds
         trimming_end=0, # remove the last X seconds
     )     

        # ---------------------------------------------------------------------
        # 4) Rerun inverse kinematics with armless Rajogopal model
        #    using optimized marker data
        # ---------------------------------------------------------------------
    print("Rerunning inverse kinematics with optimized marker data")
    # Rerun IK with laiuhlrich 2022 model and new marker data for better kinematics
    ik_files_lai = run_ik_for_gait_dynamics(
        session_folder=session_folder,
        scaled_model_path=scaled_laimodel,
        ik_setup_xml=os.path.join(
            baseDir,
            "OpenSimPipeline",
            "ForGaitDynamics",
            "Setup_IK_genericLaiUhlrich.xml",
        ),
        trial_prefix=trial_prefix,
        output_dir=for_gait_dir,
        model_type = "lai",
    )
    
    
    # Rerun IK with rajogopal armless model and new marker data for better kinematics for GaitDynamics input
    ik_files_rajogopal = run_ik_for_gait_dynamics(
        session_folder=session_folder,
        scaled_model_path=scaled_RAmodel,
        ik_setup_xml=os.path.join(
            baseDir,
            "OpenSimPipeline",
            "ForGaitDynamics",
            "Setup_IK_RajogopalArmless.xml",
        ),
        trial_prefix=trial_prefix,
        output_dir=for_gait_dir,
        model_type = "rajogopal",
    )
    
    # move optimized IK data so everything is where it needs to be for simulations
    kinematics_dir = os.path.join(session_folder, 'OpenSimData', 'Kinematics')
    for f in ik_files_lai:
        old_name = os.path.basename(f)
    
        # trial name is the part before the first "_"
        trial = old_name.split("_")[0]
    
        # new file name
        new_name = f"{trial}_Optimized.mot"
    
        # full path for destination
        new_path = os.path.join(kinematics_dir, new_name)
    
        # move and rename
        #shutil.copy(f, new_path)
        reformat_ik_mot(f, new_path)
        
    # move optimized TRC data so everything is where it needs to be for simulations
    marker_dir = os.path.join(session_folder, 'MarkerData')
    trc_files = glob.glob(os.path.join(for_gait_dir, "*optfeet*.trc"))    
    for f in trc_files:
        old_name = os.path.splitext(os.path.basename(f))[0]                
        trial = old_name.split("_")[2]                  
        new_name = f"{trial}_Optimized.trc"           
    
        new_path = os.path.join(marker_dir, new_name)
    
        shutil.copy(f, new_path)


    # ---------------------------------------------------------------------
    # 5) Run GaitDynamics GRF predictions
    # ---------------------------------------------------------------------
    client = Client("alanttan/GaitDynamics")

    for fname in os.listdir(for_gait_dir):
        if fname.endswith('forGaitDynamics.mot'):
            ik_path = os.path.join(for_gait_dir, fname)
            fname_no_ext = fname.replace(".mot", "")
    
            osim_path = os.path.join(for_gait_dir, "scaled_RagagopalArmless.osim")
            ik_long_path = os.path.join(
                for_gait_dir,
                f"{fname_no_ext}_long.mot",
            )

            # Make the long IK file if needed and also filter pelvis ty
            # IK file must be at least ~1.5 seconds long for GaitDynamics
            # Duplicate data is appended at the end and later trimmed
            added_rows, final_duration = make_long_ik(
                ik_path,
                ik_long_path,
                min_duration=1.51,
            )

    
            # Call the GaitDynamics Space on Hugging Face
            result = client.predict(
                mot_in=handle_file(ik_long_path),
                osim_in=handle_file(osim_path),
                height_in=height,
                weight_in=mass,
                speed_in=treadmill_speed,
                current_select="GRF Results",
                api_name="/enhanced_predict",
            )
    
            print("GRF " + result[0] + " files are saved in SessionDir/ForceData")
    
            # Extract GRF file from result
            output_file = Path(result[1]["value"]).resolve()
    
            # Make destination folder if needed
            force_dir = os.path.join(session_folder, "ForceData")
            os.makedirs(force_dir, exist_ok=True)

            # Copy and rename
            dest_path = os.path.join(force_dir, f"{fname_no_ext}_forces.mot")
            shutil.copy(output_file, dest_path)
    
            # Trim rows that were added for GaitDynamics compatibility
            # and Rename columns to opensim standard
            trim_mot_file(dest_path, added_rows)    
            rename_grf_mot_columns(dest_path)
    
            # # COP Predictor, overwriting COP in predicted GRFs
            code_loc = os.path.join(baseDir, "OpenSimPipeline", "ForGaitDynamics")
            trc_trial_name = fname_no_ext.replace("_IK_forGaitDynamics", "")
            refined_trc_opt = os.path.join(marker_dir,
                f"{trc_trial_name}_Optimized.trc",
            )

            predict_cop_for_trial(
                grf_path=dest_path,
                trial_name=trc_trial_name,
                trc_path=refined_trc_opt,
                ik_path=ik_path,
                artifact_path=os.path.join(code_loc, "cop_mlp_ar.pt"),
            )       
            
    close_all_loggers()
     
                

        

