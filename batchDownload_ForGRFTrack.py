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
    create_rajogopal_from_laiuhlrich,
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
sessionList = [
    "your-session-id-here",
]

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

# Seconds to trim from the start and end of each trial before processing.
# Increase these if gait event detection fails on your data.
trim_start = 0
trim_end = 0
# ------------------------ END USER INPUTS -----------------------------------


for session_id in sessionList:
    print("\n" + "-" * 72)
    print(f"Processing session: {session_id}")
    print("-" * 72)

    # Download data from the OpenCap server.
    # If only interested in marker and OpenSim data, downloadVideos=False is faster.
    # Comment out if data is already downloaded.
    download_session(
        session_id,
        sessionBasePath=downloadPath,
        downloadVideos=False,
        trial_prefix=trial_prefix,
    )

    # Session folder
    session_folder = os.path.join(downloadPath, f"OpenCapData_{session_id}")

    # -------------------------------------------------------------------------
    # Scale models for GaitDynamics and IK
    # -------------------------------------------------------------------------
    print("Creating scaled models for GaitDynamics input")
    convert_all_trc_in_folder_to_mm(os.path.join(session_folder, 'MarkerData'))

    try:
        static_trc = find_static_trial(session_folder)
        has_static = True
    except FileNotFoundError:
        print("  No static/neutral trial found; checking for existing LaiUhlrich model...")
        static_trc = None
        has_static = False

    for_gait_dir = os.path.join(session_folder, "ForGaitDynamics")
    os.makedirs(for_gait_dir, exist_ok=True)
    session_metadata_path = os.path.join(session_folder, "sessionMetadata.yaml")

    if has_static:
        # Normal path: scale both models from static TRC
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
            session_metadata_path=session_metadata_path,
            static_trc_path=static_trc,
            output_dir=for_gait_dir,
        )

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
            session_metadata_path=session_metadata_path,
            static_trc_path=static_trc,
            output_dir=os.path.join(session_folder, 'OpenSimData', 'Model'),
        )
    else:
        # Fallback: no static trial — reuse the LaiUhlrich model already scaled
        # by OpenCap (stored in OpenSimData/Model/<trial>/LaiUhlrich2022_scaled.osim)
        # and derive the Rajogopal armless scale factors from it directly.
        existing_lai = os.path.join(
            session_folder, 'OpenSimData', 'Model', trial_prefix, 'LaiUhlrich2022_scaled.osim'
        )
        if not os.path.exists(existing_lai):
            existing_lai = os.path.join(
                session_folder, 'OpenSimData', 'Model', 'LaiUhlrich2022_scaled.osim'
            )
        if not os.path.exists(existing_lai):
            # Search one level of subfolders under Model/
            model_base = os.path.join(session_folder, 'OpenSimData', 'Model')
            for sub in os.listdir(model_base):
                candidate = os.path.join(model_base, sub, 'LaiUhlrich2022_scaled.osim')
                if os.path.exists(candidate):
                    existing_lai = candidate
                    break
        if not os.path.exists(existing_lai):
            raise FileNotFoundError(
                f"No static trial and no existing LaiUhlrich model found for session "
                f"{session_id}. Cannot create scaled models without one of these."
            )

        print(f"  Found existing LaiUhlrich model: {existing_lai}")
        scaled_laimodel = existing_lai

        print("  Converting LaiUhlrich scale factors to Rajogopal armless model...")
        scaled_RAmodel, height, mass = create_rajogopal_from_laiuhlrich(
            generic_lai_path=os.path.join(
                baseDir,
                "OpenSimPipeline",
                "ForGaitDynamics",
                "LaiUhlrich2022_Generic.osim",
            ),
            scaled_lai_path=scaled_laimodel,
            generic_ra_path=os.path.join(
                baseDir,
                "OpenSimPipeline",
                "ForGaitDynamics",
                "RajogopalArmless_Generic.osim",
            ),
            generic_ra_scale_setup_xml=os.path.join(
                baseDir,
                "OpenSimPipeline",
                "ForGaitDynamics",
                "Setup_RajogopalArmless_Scaling_generic.xml",
            ),
            session_metadata_path=session_metadata_path,
            output_dir=for_gait_dir,
        )

    # -------------------------------------------------------------------------
    # Foot contact refinement
    # -------------------------------------------------------------------------
    refine_foot_kinematics_for_session(
        session_folder=session_folder,
        trial_prefix=trial_prefix,
        gait_style=gait_style,
        trimming_start=trim_start,
        trimming_end=trim_end,
        do_not_refine=False,
    )

    # -------------------------------------------------------------------------
    # Rerun IK with optimized marker data
    # -------------------------------------------------------------------------
    print("Rerunning inverse kinematics with optimized marker data")

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
        model_type="lai",
    )

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
        model_type="rajogopal",
    )

    # Copy optimized IK .mot files to Kinematics folder
    kinematics_dir = os.path.join(session_folder, 'OpenSimData', 'Kinematics')
    for f in ik_files_lai:
        old_name = os.path.basename(f)
        trial = old_name.split("_")[0]
        new_path = os.path.join(kinematics_dir, f"{trial}_Optimized.mot")
        reformat_ik_mot(f, new_path)

    # Copy optimized TRC files to MarkerData folder
    marker_dir = os.path.join(session_folder, 'MarkerData')
    trc_files = glob.glob(os.path.join(for_gait_dir, "*optfeet*.trc"))
    for f in trc_files:
        old_name = os.path.splitext(os.path.basename(f))[0]
        # filename is MarkerData_optfeet_<trial_name> — strip the fixed prefix
        trial = old_name.split("optfeet_", 1)[-1]
        new_path = os.path.join(marker_dir, f"{trial}_Optimized.trc")
        shutil.copy(f, new_path)

    # -------------------------------------------------------------------------
    # GaitDynamics GRF prediction
    # -------------------------------------------------------------------------
    client = Client("alanttan/GaitDynamics")

    for fname in os.listdir(for_gait_dir):
        if not fname.endswith('forGaitDynamics.mot'):
            continue

        ik_path = os.path.join(for_gait_dir, fname)
        fname_no_ext = fname.replace(".mot", "")
        osim_path = os.path.join(for_gait_dir, "scaled_RagagopalArmless.osim")
        ik_long_path = os.path.join(for_gait_dir, f"{fname_no_ext}_long.mot")

        # GaitDynamics requires ~1.5 s minimum; duplicate frames are added then trimmed
        added_rows, final_duration = make_long_ik(
            ik_path,
            ik_long_path,
            min_duration=1.51,
        )

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

        output_file = Path(result[1]["value"]).resolve()

        force_dir = os.path.join(session_folder, "ForceData")
        os.makedirs(force_dir, exist_ok=True)

        dest_path = os.path.join(force_dir, f"{fname_no_ext}_forces.mot")
        shutil.copy(output_file, dest_path)

        trim_mot_file(dest_path, added_rows)
        rename_grf_mot_columns(dest_path)

        # COP prediction, overwriting COP in predicted GRFs
        code_loc = os.path.join(baseDir, "OpenSimPipeline", "ForGaitDynamics")
        trc_trial_name = fname_no_ext.replace("_IK_forGaitDynamics", "")
        refined_trc_opt = os.path.join(marker_dir, f"{trc_trial_name}_Optimized.trc")

        predict_cop_for_trial(
            grf_path=dest_path,
            trial_name=trc_trial_name,
            trc_path=refined_trc_opt,
            ik_path=ik_path,
            artifact_path=os.path.join(code_loc, "cop_mlp_ar.pt"),
        )

    close_all_loggers()
