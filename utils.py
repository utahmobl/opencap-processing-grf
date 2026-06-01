# -*- coding: utf-8 -*-
"""
---------------------------------------------------------------------------
OpenCap processing: utils.py
---------------------------------------------------------------------------

Copyright 2022 Stanford University and the Authors

Author(s): Antoine Falisse, Scott Uhlrich

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
import re
import glob
import yaml
import shutil
import pickle
import zipfile
import platform
import logging
import subprocess
import urllib.request
from io import StringIO
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import opensim
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, gaussian

from utilsAPI import get_api_url
from utilsAuthentication import get_token


API_URL = get_api_url()
API_TOKEN = get_token()

# Silence OpenSim console logging (IK/Scale prints like "[info] Frame ...")
try:
    opensim.Logger.setLevelString("Off")   # OpenSim 4.x
except Exception:
    pass

# =============================================================================
# Basic server helpers
# =============================================================================
def download_file(url, file_name):
    with urllib.request.urlopen(url) as response, open(file_name, "wb") as out_file:
        shutil.copyfileobj(response, out_file)


def get_session_json(session_id):
    resp = requests.get(
        API_URL + "sessions/{}/".format(session_id),
        headers={"Authorization": "Token {}".format(API_TOKEN)},
    )

    if resp.status_code == 500:
        raise Exception("No server response. Likely not a valid session id.")

    sessionJson = resp.json()
    if "trials" not in sessionJson.keys():
        raise Exception(
            "This session is not in your username, nor is it public. You do not have access."
        )

    # Sort trials by time recorded.
    def get_created_at(trial):
        return trial["created_at"]

    sessionJson["trials"].sort(key=get_created_at)

    return sessionJson


def get_user_sessions():
    sessions = requests.get(
        API_URL + "sessions/valid/",
        headers={"Authorization": "Token {}".format(API_TOKEN)},
    ).json()
    return sessions


# TODO: this also contains public sessions of other users.
def get_user_sessions_all(user_token=API_TOKEN):
    sessions = requests.get(
        API_URL + "sessions/",
        headers={"Authorization": "Token {}".format(user_token)},
    ).json()
    return sessions


def get_user_subjects(user_token=API_TOKEN):
    subjects = requests.get(
        API_URL + "subjects/",
        headers={"Authorization": "Token {}".format(user_token)},
    ).json()
    return subjects


def get_subject_sessions(subject_id, user_token=API_TOKEN):
    sessions = requests.get(
        API_URL + "subjects/{}/".format(subject_id),
        headers={"Authorization": "Token {}".format(user_token)},
    ).json()["sessions"]
    return sessions


def get_trial_json(trial_id):
    trialJson = requests.get(
        API_URL + "trials/{}/".format(trial_id),
        headers={"Authorization": "Token {}".format(API_TOKEN)},
    ).json()
    return trialJson


def get_neutral_trial_id(session_id):
    session = get_session_json(session_id)
    neutral_ids = [t["id"] for t in session["trials"] if t["name"] == "neutral"]

    if len(neutral_ids) > 0:
        neutralID = neutral_ids[-1]
    elif session["meta"]["neutral_trial"]:
        neutralID = session["meta"]["neutral_trial"]["id"]
    else:
        raise Exception("No neutral trial in session.")

    return neutralID


def get_calibration_trial_id(session_id):
    session = get_session_json(session_id)

    calib_ids = [t["id"] for t in session["trials"] if t["name"] == "calibration"]

    if len(calib_ids) > 0:
        calibID = calib_ids[-1]
    elif session["meta"]["sessionWithCalibration"]:
        calibID = get_calibration_trial_id(session["meta"]["sessionWithCalibration"]["id"])
    else:
        raise Exception("No calibration trial in session.")

    return calibID


def get_camera_mapping(session_id, session_path):
    calibration_id = get_calibration_trial_id(session_id)
    trial = get_trial_json(calibration_id)
    resultTags = [res["tag"] for res in trial["results"]]

    mappingPath = os.path.join(session_path, "Videos", "mappingCamDevice.pickle")
    os.makedirs(os.path.join(session_path, "Videos"), exist_ok=True)
    if not os.path.exists(mappingPath):
        mappingURL = trial["results"][resultTags.index("camera_mapping")]["media"]
        download_file(mappingURL, mappingPath)


def get_model_and_metadata(session_id, session_path):
    neutral_id = get_neutral_trial_id(session_id)
    trial = get_trial_json(neutral_id)
    resultTags = [res["tag"] for res in trial["results"]]

    # Metadata.
    metadataPath = os.path.join(session_path, "sessionMetadata.yaml")
    if not os.path.exists(metadataPath):
        metadataURL = trial["results"][resultTags.index("session_metadata")]["media"]
        download_file(metadataURL, metadataPath)

    # Model.
    modelURL = trial["results"][resultTags.index("opensim_model")]["media"]
    modelName = modelURL[modelURL.rfind("-") + 1 : modelURL.rfind("?")]
    modelFolder = os.path.join(session_path, "OpenSimData", "Model")
    modelPath = os.path.join(modelFolder, modelName)
    if not os.path.exists(modelPath):
        os.makedirs(modelFolder, exist_ok=True)
        download_file(modelURL, modelPath)

    return modelName


def get_main_settings(session_folder, trial_name):
    settings_path = os.path.join(
        session_folder, "MarkerData", "Settings", "settings_" + trial_name + ".yaml"
    )
    main_settings = import_metadata(settings_path)
    return main_settings


def get_model_name_from_metadata(sessionFolder, appendText="_scaled"):
    metadataPath = os.path.join(sessionFolder, "sessionMetadata.yaml")

    if os.path.exists(metadataPath):
        metadata = import_metadata(os.path.join(sessionFolder, "sessionMetadata.yaml"))
        modelName = metadata["openSimModel"] + appendText + ".osim"
    else:
        raise Exception("Session metadata not found, could not identify OpenSim model.")

    return modelName


def get_motion_data(trial_id, session_path):
    trial = get_trial_json(trial_id)
    trial_name = trial["name"]
    resultTags = [res["tag"] for res in trial["results"]]

    # Marker data.
    if "marker_data" in resultTags:
        markerFolder = os.path.join(session_path, "MarkerData")
        markerPath = os.path.join(markerFolder, trial_name + ".trc")
        os.makedirs(markerFolder, exist_ok=True)
        if not os.path.exists(markerPath):
            markerURL = trial["results"][resultTags.index("marker_data")]["media"]
            download_file(markerURL, markerPath)

    # IK data.
    if "ik_results" in resultTags:
        ikFolder = os.path.join(session_path, "OpenSimData", "Kinematics")
        ikPath = os.path.join(ikFolder, trial_name + ".mot")
        os.makedirs(ikFolder, exist_ok=True)
        if not os.path.exists(ikPath):
            ikURL = trial["results"][resultTags.index("ik_results")]["media"]
            download_file(ikURL, ikPath)

    # Main settings
    if "main_settings" in resultTags:
        settingsFolder = os.path.join(session_path, "MarkerData", "Settings")
        settingsPath = os.path.join(settingsFolder, "settings_" + trial_name + ".yaml")
        os.makedirs(settingsFolder, exist_ok=True)
        if not os.path.exists(settingsPath):
            settingsURL = trial["results"][resultTags.index("main_settings")]["media"]
            download_file(settingsURL, settingsPath)


def get_geometries(session_path, modelName="LaiUhlrich2022_scaled"):
    geometryFolder = os.path.join(session_path, "OpenSimData", "Model", "Geometry")
    try:
        os.makedirs(geometryFolder, exist_ok=True)
        if "Lai" in modelName:
            modelType = "LaiArnold"
            vtpNames = [
                "capitate_lvs",
                "capitate_rvs",
                "hamate_lvs",
                "hamate_rvs",
                "hat_jaw",
                "hat_ribs_scap",
                "hat_skull",
                "hat_spine",
                "humerus_lv",
                "humerus_rv",
                "index_distal_lvs",
                "index_distal_rvs",
                "index_medial_lvs",
                "index_medial_rvs",
                "index_proximal_lvs",
                "index_proximal_rvs",
                "little_distal_lvs",
                "little_distal_rvs",
                "little_medial_lvs",
                "little_medial_rvs",
                "little_proximal_lvs",
                "little_proximal_rvs",
                "lunate_lvs",
                "lunate_rvs",
                "l_bofoot",
                "l_femur",
                "l_fibula",
                "l_foot",
                "l_patella",
                "l_pelvis",
                "l_talus",
                "l_tibia",
                "metacarpal1_lvs",
                "metacarpal1_rvs",
                "metacarpal2_lvs",
                "metacarpal2_rvs",
                "metacarpal3_lvs",
                "metacarpal3_rvs",
                "metacarpal4_lvs",
                "metacarpal4_rvs",
                "metacarpal5_lvs",
                "metacarpal5_rvs",
                "middle_distal_lvs",
                "middle_distal_rvs",
                "middle_medial_lvs",
                "middle_medial_rvs",
                "middle_proximal_lvs",
                "middle_proximal_rvs",
                "pisiform_lvs",
                "pisiform_rvs",
                "radius_lv",
                "radius_rv",
                "ring_distal_lvs",
                "ring_distal_rvs",
                "ring_medial_lvs",
                "ring_medial_rvs",
                "ring_proximal_lvs",
                "ring_proximal_rvs",
                "r_bofoot",
                "r_femur",
                "r_fibula",
                "r_foot",
                "r_patella",
                "r_pelvis",
                "r_talus",
                "r_tibia",
                "sacrum",
                "scaphoid_lvs",
                "scaphoid_rvs",
                "thumb_distal_lvs",
                "thumb_distal_rvs",
                "thumb_proximal_lvs",
                "thumb_proximal_rvs",
                "trapezium_lvs",
                "trapezium_rvs",
                "trapezoid_lvs",
                "trapezoid_rvs",
                "triquetrum_lvs",
                "triquetrum_rvs",
                "ulna_lv",
                "ulna_rv",
            ]
        else:
            raise ValueError("Geometries not available for this model")

        for vtpName in vtpNames:
            url = "https://mc-opencap-public.s3.us-west-2.amazonaws.com/geometries_vtp/{}/{}.vtp".format(
                modelType, vtpName
            )
            filename = os.path.join(geometryFolder, "{}.vtp".format(vtpName))
            download_file(url, filename)
    except:
        pass


def import_metadata(filePath):
    myYamlFile = open(filePath)
    parsedYamlFile = yaml.load(myYamlFile, Loader=yaml.FullLoader)
    return parsedYamlFile


def download_kinematics(session_id, folder=None, trialNames=None):
    if folder is None:
        folder = os.getcwd()
    os.makedirs(folder, exist_ok=True)

    neutral_id = get_neutral_trial_id(session_id)
    get_motion_data(neutral_id, folder)
    modelName = get_model_and_metadata(session_id, folder)
    modelName = modelName.replace(".osim", "")

    sessionJson = get_session_json(session_id)
    sessionTrialNames = [t["name"] for t in sessionJson["trials"]]
    if trialNames is not None:
        [print(t + " not in session trial names.") for t in trialNames if t not in sessionTrialNames]

    loadedTrialNames = []
    for trialDict in sessionJson["trials"]:
        if trialNames is not None and trialDict["name"] not in trialNames:
            continue
        trial_id = trialDict["id"]
        get_motion_data(trial_id, folder)
        loadedTrialNames.append(trialDict["name"])

    get_geometries(folder, modelName=modelName)
    return loadedTrialNames, modelName


def download_trial(trial_id, neutral_id, folder, session_id=None):
    trial = get_trial_json(trial_id)
    if session_id is None:
        session_id = trial["session_id"]

    os.makedirs(folder, exist_ok=True)
    get_model_and_metadata(session_id, folder)

    get_motion_data(trial_id, folder)
    get_motion_data(neutral_id, folder)  # EYM Edit for scaling for ML

    return trial["name"]


def get_trial_id(session_id, trial_name):
    session = get_session_json(session_id)
    trial_id = [t["id"] for t in session["trials"] if t["name"] == trial_name]
    return trial_id[0]


# =============================================================================
# Storage helpers
# =============================================================================
def storage_to_numpy(storage_file, excess_header_entries=0):
    """
    Returns the data from a storage file in a numpy format.
    Skips all lines up to and including the line that says 'endheader'.
    """
    f = open(storage_file, "r")

    header_line = False
    for i, line in enumerate(f):
        if header_line:
            column_names = line.split()
            break
        if line.count("endheader") != 0:
            line_number_of_line_containing_endheader = i + 1
            header_line = True
    f.close()

    if excess_header_entries == 0:
        names = True
        skip_header = line_number_of_line_containing_endheader
    else:
        names = column_names[:-excess_header_entries]
        skip_header = line_number_of_line_containing_endheader + 1

    data = np.genfromtxt(storage_file, names=names, skip_header=skip_header)
    return data


def storage_to_dataframe(storage_file, headers):
    data = storage_to_numpy(storage_file)
    out = pd.DataFrame(data=data["time"], columns=["time"])
    for count, header in enumerate(headers):
        out.insert(count + 1, header, data[header])
    return out


def load_storage(file_path, outputFormat="numpy"):
    table = opensim.TimeSeriesTable(file_path)
    data = table.getMatrix().to_numpy()
    time = np.asarray(table.getIndependentColumn()).reshape(-1, 1)
    data = np.hstack((time, data))
    headers = ["time"] + list(table.getColumnLabels())

    if outputFormat == "numpy":
        return data, headers
    elif outputFormat == "dataframe":
        return pd.DataFrame(data, columns=headers)
    else:
        return None


def numpy_to_storage(labels, data, storage_file, datatype=None):
    assert data.shape[1] == len(labels), "# labels doesn't match columns"
    assert labels[0] == "time"

    f = open(storage_file, "w")

    if datatype is None:
        f.write("name %s\n" % storage_file)
        f.write("datacolumns %d\n" % data.shape[1])
        f.write("datarows %d\n" % data.shape[0])
        f.write("range %f %f\n" % (np.min(data[:, 0]), np.max(data[:, 0])))
        f.write("endheader \n")
    else:
        if datatype == "IK":
            f.write("Coordinates\n")
        elif datatype == "ID":
            f.write("Inverse Dynamics Generalized Forces\n")
        elif datatype == "GRF":
            f.write("%s\n" % storage_file)
        elif datatype == "muscle_forces":
            f.write("ModelForces\n")

        f.write("version=1\n")
        f.write("nRows=%d\n" % data.shape[0])
        f.write("nColumns=%d\n" % data.shape[1])

        if datatype == "IK":
            f.write("inDegrees=yes\n\n")
            f.write("Units are S.I. units (second, meters, Newtons, ...)\n")
            f.write(
                "If the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no).\n\n"
            )
        elif datatype == "ID":
            f.write("inDegrees=no\n")
        elif datatype == "GRF":
            f.write("inDegrees=yes\n")
        elif datatype == "muscle_forces":
            f.write("inDegrees=yes\n\n")
            f.write("This file contains the forces exerted on a model during a simulation.\n\n")
            f.write(
                "A force is a generalized force, meaning that it can be either a force (N) or a torque (Nm).\n\n"
            )
            f.write("Units are S.I. units (second, meters, Newtons, ...)\n")
            f.write("Angles are in degrees.\n\n")

        f.write("endheader \n")

    for lab in labels:
        f.write("%s\t" % lab)
    f.write("\n")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            f.write("%20.8f\t" % data[i, j])
        f.write("\n")

    f.close()


# =============================================================================
# Video download helpers
# =============================================================================
def download_videos_from_server(
    session_id,
    trial_id,
    isCalibration=False,
    isStaticPose=False,
    trial_name=None,
    session_path=None,
):
    if session_path is None:
        data_dir = os.getcwd()
        session_path = os.path.join(data_dir, "Data", session_id)
    if not os.path.exists(session_path):
        os.makedirs(session_path, exist_ok=True)

    resp = requests.get(
        "{}trials/{}/".format(API_URL, trial_id),
        headers={"Authorization": "Token {}".format(API_TOKEN)},
    )
    trial = resp.json()
    if trial_name is None:
        trial_name = trial["name"]
    trial_name = trial_name.replace(" ", "")

    print("\nDownloading {}".format(trial_name))

    if not os.path.exists(os.path.join(session_path, "Videos", "mappingCamDevice.pickle")):
        mappingCamDevice = {}
        for k, video in enumerate(trial["videos"]):
            os.makedirs(
                os.path.join(session_path, "Videos", "Cam{}".format(k), "InputMedia", trial_name),
                exist_ok=True,
            )
            video_path = os.path.join(
                session_path,
                "Videos",
                "Cam{}".format(k),
                "InputMedia",
                trial_name,
                trial_name + ".mov",
            )
            download_file(video["video"], video_path)
            mappingCamDevice[video["device_id"].replace("-", "").upper()] = k
        with open(os.path.join(session_path, "Videos", "mappingCamDevice.pickle"), "wb") as handle:
            pickle.dump(mappingCamDevice, handle)
    else:
        with open(os.path.join(session_path, "Videos", "mappingCamDevice.pickle"), "rb") as handle:
            mappingCamDevice = pickle.load(handle)
            # ensure upper on deviceID
            for dID in list(mappingCamDevice.keys()):
                mappingCamDevice[dID.upper()] = mappingCamDevice.pop(dID)

        for video in trial["videos"]:
            k = mappingCamDevice[video["device_id"].replace("-", "").upper()]
            videoDir = os.path.join(session_path, "Videos", "Cam{}".format(k), "InputMedia", trial_name)
            os.makedirs(videoDir, exist_ok=True)
            video_path = os.path.join(videoDir, trial_name + ".mov")
            if not os.path.exists(video_path):
                if video["video"]:
                    download_file(video["video"], video_path)

    return trial_name


def get_calibration(session_id, session_path):
    calibration_id = get_calibration_trial_id(session_id)

    resp = requests.get(
        "{}trials/{}/".format(API_URL, calibration_id),
        headers={"Authorization": "Token {}".format(API_TOKEN)},
    )
    trial = resp.json()
    calibResultTags = [res["tag"] for res in trial["results"]]

    videoFolder = os.path.join(session_path, "Videos")
    os.makedirs(videoFolder, exist_ok=True)

    if trial["status"] != "done":
        return

    mapURL = trial["results"][calibResultTags.index("camera_mapping")]["media"]
    mapLocalPath = os.path.join(videoFolder, "mappingCamDevice.pickle")

    download_and_switch_calibration(session_id, session_path, calibTrialID=calibration_id)

    if len(glob.glob(mapLocalPath)) == 0:
        download_file(mapURL, mapLocalPath)


def download_and_switch_calibration(session_id, session_path, calibTrialID=None):
    if calibTrialID is None:
        calibTrialID = get_calibration_trial_id(session_id)

    resp = requests.get(
        "https://api.opencap.ai/trials/{}/".format(calibTrialID),
        headers={"Authorization": "Token {}".format(API_TOKEN)},
    )
    trial = resp.json()

    calibURLs = {t["device_id"]: t["media"] for t in trial["results"] if t["tag"] == "calibration_parameters_options"}
    calibImgURLs = {t["device_id"]: t["media"] for t in trial["results"] if t["tag"] == "calibration-img"}

    _, imgExtension = os.path.splitext(calibImgURLs[list(calibImgURLs.keys())[0]])
    lastIdx = imgExtension.find("?")
    if lastIdx > 0:
        imgExtension = imgExtension[:lastIdx]

    if "meta" in trial.keys() and trial["meta"] is not None and "calibration" in trial["meta"].keys():
        calibDict = trial["meta"]["calibration"]
        calibImgFolder = os.path.join(session_path, "CalibrationImages")
        os.makedirs(calibImgFolder, exist_ok=True)

        for cam, calibNum in calibDict.items():
            camDir = os.path.join(session_path, "Videos", cam)
            os.makedirs(camDir, exist_ok=True)
            file_name = os.path.join(camDir, "cameraIntrinsicsExtrinsics.pickle")
            img_fileName = os.path.join(calibImgFolder, "calib_img" + cam + imgExtension)

            if calibNum == 0:
                download_file(calibURLs[cam + "_soln0"], file_name)
                download_file(calibImgURLs[cam], img_fileName)
            elif calibNum == 1:
                download_file(calibURLs[cam + "_soln1"], file_name)
                download_file(calibImgURLs[cam + "_altSoln"], img_fileName)


def post_file_to_trial(filePath, trial_id, tag, device_id):
    files = {"media": open(filePath, "rb")}
    data = {"trial": trial_id, "tag": tag, "device_id": device_id}

    requests.post(
        "{}results/".format(API_URL),
        files=files,
        data=data,
        headers={"Authorization": "Token {}".format(API_TOKEN)},
    )
    files["media"].close()


def post_video_to_trial(filePath, trial_id, device_id, parameters):
    files = {"video": open(filePath, "rb")}
    data = {"trial": trial_id, "device_id": device_id, "parameters": parameters}

    requests.post(
        "{}videos/".format(API_URL),
        files=files,
        data=data,
        headers={"Authorization": "Token {}".format(API_TOKEN)},
    )
    files["video"].close()


def delete_video_from_trial(video_id):
    requests.delete(
        "{}videos/{}/".format(API_URL, video_id),
        headers={"Authorization": "Token {}".format(API_TOKEN)},
    )


def delete_results(trial_id, tag=None, resultNum=None):
    if resultNum is not None:
        resultNums = [resultNum]
    elif tag is not None:
        trial = get_trial_json(trial_id)
        resultNums = [r["id"] for r in trial["results"] if r["tag"] == tag]
    else:
        trial = get_trial_json(trial_id)
        resultNums = [r["id"] for r in trial["results"]]

    for rNum in resultNums:
        requests.delete(
            API_URL + "results/{}/".format(rNum),
            headers={"Authorization": "Token {}".format(API_TOKEN)},
        )


def set_trial_status(trial_id, status):
    if status not in ["done", "error", "stopped", "reprocess"]:
        raise ValueError("Invalid status. Available statuses: done, error, stopped, reprocess")

    requests.patch(
        API_URL + "trials/{}/".format(trial_id),
        data={"status": status},
        headers={"Authorization": "Token {}".format(API_TOKEN)},
    )


def set_session_subject(session_id, subject_id):
    requests.patch(
        API_URL + "sessions/{}/".format(session_id),
        data={"subject": subject_id},
        headers={"Authorization": "Token {}".format(API_TOKEN)},
    )


def get_syncd_videos(trial_id, session_path):
    trial = requests.get(
        "{}trials/{}/".format(API_URL, trial_id),
        headers={"Authorization": "Token {}".format(API_TOKEN)},
    ).json()
    trial_name = trial["name"]

    if trial["results"]:
        for result in trial["results"]:
            if result["tag"] == "video-sync":
                url = result["media"]
                cam, suff = os.path.splitext(url[url.rfind("_") + 1 :])
                lastIdx = suff.find("?")
                if lastIdx > 0:
                    suff = suff[:lastIdx]

                syncVideoPath = os.path.join(
                    session_path,
                    "Videos",
                    cam,
                    "InputMedia",
                    trial_name,
                    trial_name + "_sync" + suff,
                )
                download_file(url, syncVideoPath)


def download_session(
    session_id,
    sessionBasePath=None,
    zipFolder=False,
    writeToDB=False,
    downloadVideos=True,
    trial_prefix=None,
):
    print("\nDownloading {}".format(session_id))

    if sessionBasePath is None:
        sessionBasePath = os.path.join(os.getcwd(), "Data")

    session = get_session_json(session_id)
    session_path = os.path.join(sessionBasePath, "OpenCapData_" + session_id)

    calib_id = get_calibration_trial_id(session_id)
    neutral_id = get_neutral_trial_id(session_id)

    dynamic_ids = []
    for t in session["trials"]:
        name = t["name"]
        if name not in ("calibration", "neutral"):
            if trial_prefix is None or trial_prefix in name:
                dynamic_ids.append(t["id"])

    try:
        get_camera_mapping(session_id, session_path)
        if downloadVideos:
            download_videos_from_server(
                session_id,
                calib_id,
                isCalibration=True,
                isStaticPose=False,
                session_path=session_path,
            )
        get_calibration(session_id, session_path)
    except:
        pass

    try:
        modelName = get_model_and_metadata(session_id, session_path)
        get_motion_data(neutral_id, session_path)
        if downloadVideos:
            download_videos_from_server(
                session_id,
                neutral_id,
                isCalibration=False,
                isStaticPose=True,
                session_path=session_path,
            )
        get_syncd_videos(neutral_id, session_path)
    except:
        modelName = ""
        pass

    for dynamic_id in dynamic_ids:
        try:
            get_motion_data(dynamic_id, session_path)
            if downloadVideos:
                download_videos_from_server(
                    session_id,
                    dynamic_id,
                    isCalibration=False,
                    isStaticPose=False,
                    session_path=session_path,
                )
            get_syncd_videos(dynamic_id, session_path)
        except:
            pass

    repoDir = os.path.dirname(os.path.abspath(__file__))

    try:
        pathReadme = os.path.join(repoDir, "Resources", "README.txt")
        pathReadmeEnd = os.path.join(session_path, "README.txt")
        shutil.copy2(pathReadme, pathReadmeEnd)
    except:
        pass

    try:
        if "Lai" in modelName:
            modelType = "LaiArnold"
        else:
            raise ValueError("Geometries not available for this model, please contact us")

        if platform.system() == "Windows":
            geometryDir = os.path.join(repoDir, "tmp", modelType, "Geometry")
        else:
            geometryDir = "/tmp/{}/Geometry".format(modelType)

        if not os.path.exists(geometryDir):
            os.makedirs(geometryDir, exist_ok=True)
            get_geometries(session_path, modelName=modelName)

        geometryDirEnd = os.path.join(session_path, "OpenSimData", "Model", "Geometry")
        if os.path.exists(geometryDirEnd):
            pass
        else:
            shutil.copytree(geometryDir, geometryDirEnd)
    except:
        pass

    def zipdir(path, ziph):
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(
                    os.path.join(root, file),
                    os.path.relpath(os.path.join(root, file), os.path.join(path, "..")),
                )

    session_zip = "{}.zip".format(session_path)
    if os.path.isfile(session_zip):
        os.remove(session_zip)

    if zipFolder:
        zipf = zipfile.ZipFile(session_zip, "w", zipfile.ZIP_DEFLATED)
        zipdir(session_path, zipf)
        zipf.close()

    if writeToDB and len(dynamic_ids) > 0:
        post_file_to_trial(session_zip, dynamic_ids[-1], tag="session_zip", device_id="all")


# =============================================================================
# Signal helpers
# =============================================================================
def cross_corr(y1, y2, multCorrGaussianStd=None, visualize=False):
    if len(y1) > len(y2):
        temp = np.zeros(len(y1))
        temp[0 : len(y2)] = y2
        y2 = np.copy(temp)
    elif len(y2) > len(y1):
        temp = np.zeros(len(y2))
        temp[0 : len(y1)] = y1
        y1 = np.copy(temp)

    y1_auto_corr = np.dot(y1, y1) / len(y1)
    y2_auto_corr = np.dot(y2, y2) / len(y1)
    corr = np.correlate(y1, y2, mode="same")
    unbiased_sample_size = np.correlate(np.ones(len(y1)), np.ones(len(y1)), mode="same")
    corr = corr / unbiased_sample_size / np.sqrt(y1_auto_corr * y2_auto_corr)
    shift = len(y1) // 2

    if visualize:
        plt.figure()
        plt.plot(corr)
        plt.title("vertical velocity correlation")

    if multCorrGaussianStd is not None:
        corr = np.multiply(corr, gaussian(len(corr), multCorrGaussianStd))
        if visualize:
            plt.plot(corr, color=[0.4, 0.4, 0.4])
            plt.legend(["corr", "corr*gaussian"])

    argmax_corr = np.argmax(corr)
    max_corr = np.nanmax(corr)
    lag = argmax_corr - shift

    return max_corr, lag


def downsample(data, time, framerate_in, framerate_out):
    downsampling_factor = framerate_in / framerate_out
    original_indices = np.arange(len(data))
    new_indices = np.arange(0, len(data), downsampling_factor)

    downsampled_data = np.ndarray((len(new_indices), data.shape[1]))
    for i in range(data.shape[1]):
        downsampled_data[:, i] = np.interp(new_indices, original_indices, data[:, i])

    downsampled_time = np.interp(new_indices, original_indices, time)
    return downsampled_time, downsampled_data


# =============================================================================
# OpenSim command line discovery
# =============================================================================
def find_opensim_cmd():
    cmd = shutil.which("opensim-cmd") or shutil.which("opensim-cmd.exe")
    if cmd:
        return cmd

    try:
        pkg_dir = os.path.dirname(os.path.abspath(opensim.__file__))
        sdk_dir = os.path.dirname(pkg_dir)
        opensim_root = os.path.dirname(sdk_dir)
        candidate = os.path.join(opensim_root, "bin", "opensim-cmd.exe")
        if os.path.exists(candidate):
            return candidate
    except (AttributeError, TypeError):
        pass

    common_paths = [
        r"C:\OpenSim 4.5\bin\opensim-cmd.exe",
        r"C:\OpenSim 4.4\bin\opensim-cmd.exe",
        r"C:\OpenSim 4.3\bin\opensim-cmd.exe",
        r"C:\Program Files\OpenSim 4.5\bin\opensim-cmd.exe",
        r"C:\Program Files\OpenSim 4.4\bin\opensim-cmd.exe",
        r"C:\Program Files\OpenSim 4.3\bin\opensim-cmd.exe",
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path

    return None


def find_static_trial(session_folder):
    marker_folder = os.path.join(session_folder, "MarkerData")

    if not os.path.isdir(marker_folder):
        raise FileNotFoundError(f"MarkerData folder not found: {marker_folder}")

    files = os.listdir(marker_folder)
    matches = []

    for f in files:
        lower = f.lower()
        if "static" in lower or "neutral" in lower:
            matches.append(os.path.join(marker_folder, f))

    if len(matches) == 0:
        raise FileNotFoundError("No static or neutral calibration file found in MarkerData.")

    if len(matches) > 1:
        raise RuntimeError(
            "Multiple calibration files found, expected only one. Matches:\n" + "\n".join(matches)
        )

    return matches[0]


# =============================================================================
# TRC utilities (including dedup)
# =============================================================================
def _find_trc_marker_header_index(lines):
    """
    Return index of the TRC line that begins with 'Frame#' or 'Time'.
    """
    for i, line in enumerate(lines):
        if line.startswith("Frame#") or line.startswith("Time"):
            return i
    return None


def _parse_trc_marker_names_from_header_line(line):
    toks = line.rstrip("\n\r").split("\t")
    return [t for t in toks[2:] if t.strip()]


def deduplicate_trc_markers(trc_in, trc_out=None, prefer_keep=None, verbose=False):
    """
    Remove duplicate marker names in a TRC by KEEPING the first occurrence
    (or a preferred occurrence), dropping the rest, and fixing NumMarkers.

    This is useful for monocular exports that sometimes include aliases that
    collide after mapping (for example r_big_toe and r_toe).

    Parameters
    ----------
    trc_in : str or Path
    trc_out : str or Path or None
        If None, overwrites trc_in.
    prefer_keep : dict or None
        Optional dict mapping marker_name -> occurrence index to keep (0-based)
        OR marker_name -> "first"/"last".
    verbose : bool

    Returns
    -------
    str
        Output path.
    """
    trc_in = str(trc_in)
    if trc_out is None:
        trc_out = trc_in
    trc_out = str(trc_out)

    if prefer_keep is None:
        prefer_keep = {}

    with open(trc_in, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    hdr_idx = _find_trc_marker_header_index(lines)
    if hdr_idx is None:
        raise ValueError(f"Could not find TRC marker header line (Frame#/Time) in: {trc_in}")

    marker_names = _parse_trc_marker_names_from_header_line(lines[hdr_idx])
    if len(marker_names) == 0:
        shutil.copyfile(trc_in, trc_out)
        return trc_out

    # Build occurrences map
    occ = {}
    for i, m in enumerate(marker_names):
        occ.setdefault(m, []).append(i)

    duplicates = {m: idxs for m, idxs in occ.items() if len(idxs) > 1}
    if not duplicates:
        if trc_in != trc_out:
            shutil.copyfile(trc_in, trc_out)
        return trc_out

    # Decide which indices to keep
    keep_marker_idx = []
    for m, idxs in occ.items():
        if len(idxs) == 1:
            keep_marker_idx.append(idxs[0])
            continue

        pref = prefer_keep.get(m, "first")
        if isinstance(pref, int):
            chosen = idxs[min(max(pref, 0), len(idxs) - 1)]
        else:
            pref_low = str(pref).lower()
            if pref_low == "last":
                chosen = idxs[-1]
            else:
                chosen = idxs[0]
        keep_marker_idx.append(chosen)

    keep_marker_idx = sorted(set(keep_marker_idx))
    keep_marker_names = [marker_names[i] for i in keep_marker_idx]

    # Update NumMarkers on line 2 (0-based index 2) if present
    if len(lines) >= 3:
        num_line = lines[2].rstrip("\n")
        num_toks = num_line.split("\t")
        if len(num_toks) >= 4:
            num_toks[3] = str(len(keep_marker_names))
            lines[2] = "\t".join(num_toks) + "\n"

    # Replace marker header line with deduped names, preserve spacing style by writing tokens with \t\t between markers
    lines[hdr_idx] = "Frame#\tTime\t" + "\t\t".join(keep_marker_names) + "\n"

    # Coordinate label line (hdr_idx+1) should match NumMarkers, rebuild it
    coord_idx = hdr_idx + 1
    if coord_idx < len(lines):
        xyz = ["\t\t"]
        for i_m in range(len(keep_marker_names)):
            xyz.append(f"X{i_m+1}\tY{i_m+1}\tZ{i_m+1}")
        lines[coord_idx] = "".join([xyz[0], "\t".join(xyz[1:]), "\n"])

    # Find data start: usually hdr_idx+2 (may have blank line at hdr_idx+2)
    start_row = hdr_idx + 2
    if start_row < len(lines) and lines[start_row].strip() == "":
        start_row += 1

    # Remap data rows by marker index
    out_lines = lines[:start_row]
    for line in lines[start_row:]:
        if not line.strip():
            continue
        toks = line.rstrip("\n\r").split("\t")
        if len(toks) < 2:
            continue

        frame = toks[0]
        time = toks[1]
        row_out = [frame, time]

        # Each marker occupies 3 columns, starting at col 2
        for mi in keep_marker_idx:
            base = 2 + 3 * mi
            if base + 2 < len(toks):
                row_out.extend([toks[base], toks[base + 1], toks[base + 2]])
            else:
                row_out.extend(["0", "0", "0"])

        out_lines.append("\t".join(row_out) + "\n")

    if verbose:
        print(f"[deduplicate_trc_markers] {os.path.basename(trc_in)}: removed duplicates: {list(duplicates.keys())}")

    with open(trc_out, "w", encoding="utf-8") as f:
        f.writelines(out_lines)

    return trc_out


# =============================================================================
# Model scaling (LaiUhlrich)
# =============================================================================
def create_LaiUhlrich_model(
    generic_model_path: str,
    generic_scale_setup_xml: str,
    session_metadata_path: str,
    static_trc_path: str,
    output_dir: str,
    opensim_install_dir: str = None,
):
    """
    Scale a LaiUhlrich2022 OpenSim model using static trial data.

    Supports:
      1) OpenCap "study_offsetRemoved" marker TRCs
      2) legacy "*_study" marker TRCs
      3) OpenCap monocular TRCs (no *_study suffixes, different naming)

    Key behaviors:
      - Automatically selects the correct marker mapping based on TRC header
      - Handles TRCs where columns are either:
          (a) source-named (e.g., r_ASIS) and need mapping to target names, OR
          (b) already target-named (e.g., r.ASIS) and can be used directly
      - Deduplicates collisions deterministically
    """
    os.makedirs(output_dir, exist_ok=True)
    trc_dir = os.path.join(os.path.dirname(session_metadata_path), "ForGaitDynamics", "TRC_Files")
    os.makedirs(trc_dir, exist_ok=True)

    # optional safety: remove exact duplicate marker names in the source static TRC (in place copy)
    static_trc_path = deduplicate_trc_markers(static_trc_path, trc_out=None, verbose=False)

    opensim_cmd_path = None
    if opensim_install_dir:
        candidate_path = os.path.join(opensim_install_dir, "bin", "opensim-cmd.exe")
        if os.path.exists(candidate_path):
            opensim_cmd_path = candidate_path
    if not opensim_cmd_path:
        opensim_cmd_path = find_opensim_cmd()
    if not opensim_cmd_path:
        raise FileNotFoundError(
            "Could not find opensim-cmd.exe. Please ensure OpenSim is installed and either provide "
            "opensim_install_dir or add OpenSim to your PATH."
        )

    height_m = None
    mass_kg = None
    with open(session_metadata_path, "r") as f:
        for line in f:
            line = line.strip()
            if "height_m:" in line:
                try:
                    height_m = float(line.split("height_m:")[-1].strip())
                except ValueError:
                    pass
            if "mass_kg:" in line:
                try:
                    mass_kg = float(line.split("mass_kg:")[-1].strip())
                except ValueError:
                    pass
    if height_m is None or mass_kg is None:
        raise ValueError(f"Could not read height or mass from {session_metadata_path}")

    marker_mapping_offset = {
        "C7_study_offsetRemoved": "C7",
        "r_shoulder_study_offsetRemoved": "R_Shoulder",
        "L_shoulder_study_offsetRemoved": "L_Shoulder",
        "r.ASIS_study_offsetRemoved": "r.ASIS",
        "L.ASIS_study_offsetRemoved": "L.ASIS",
        "r.PSIS_study_offsetRemoved": "r.PSIS",
        "L.PSIS_study_offsetRemoved": "L.PSIS",
        "r_knee_study_offsetRemoved": "r_knee",
        "L_knee_study_offsetRemoved": "L_knee",
        "r_mknee_study_offsetRemoved": "r_mknee",
        "L_mknee_study_offsetRemoved": "L_mknee",
        "r_ankle_study_offsetRemoved": "r_ankle",
        "L_ankle_study_offsetRemoved": "L_ankle",
        "r_mankle_study_offsetRemoved": "r_mankle",
        "L_mankle_study_offsetRemoved": "L_mankle",
        "r_calc_study_offsetRemoved": "r_calc",
        "L_calc_study_offsetRemoved": "L_calc",
        "r_toe_study_offsetRemoved": "r_toe",
        "L_toe_study_offsetRemoved": "L_toe",
        "r_5meta_study_offsetRemoved": "r_5meta",
        "L_5meta_study_offsetRemoved": "L_5meta",
        "r_thigh1_study_offsetRemoved": "r_thigh1",
        "r_thigh2_study_offsetRemoved": "r_thigh2",
        "r_thigh3_study_offsetRemoved": "r_thigh3",
        "L_thigh1_study_offsetRemoved": "L_thigh1",
        "L_thigh2_study_offsetRemoved": "L_thigh2",
        "L_thigh3_study_offsetRemoved": "L_thigh3",
        "r_sh1_study_offsetRemoved": "r_sh1",
        "r_sh2_study_offsetRemoved": "r_sh2",
        "r_sh3_study_offsetRemoved": "r_sh3",
        "L_sh1_study_offsetRemoved": "L_sh1",
        "L_sh2_study_offsetRemoved": "L_sh2",
        "L_sh3_study_offsetRemoved": "L_sh3",
        "RHJC_study_offsetRemoved": "R_HJC",
        "LHJC_study_offsetRemoved": "L_HJC",
        "r_lelbow_study_offsetRemoved": "r_lelbow",
        "L_lelbow_study_offsetRemoved": "L_lelbow",
        "r_melbow_study_offsetRemoved": "r_melbow",
        "L_melbow_study_offsetRemoved": "L_melbow",
        "r_lwrist_study_offsetRemoved": "r_lwrist",
        "L_lwrist_study_offsetRemoved": "L_lwrist",
        "r_mwrist_study_offsetRemoved": "r_mwrist",
        "L_mwrist_study_offsetRemoved": "L_mwrist",
    }

    marker_mapping_legacy = {
        "C7_study": "C7",
        "r_shoulder_study": "R_Shoulder",
        "L_shoulder_study": "L_Shoulder",
        "r.ASIS_study": "r.ASIS",
        "L.ASIS_study": "L.ASIS",
        "r.PSIS_study": "r.PSIS",
        "L.PSIS_study": "L.PSIS",
        "r_knee_study": "r_knee",
        "L_knee_study": "L_knee",
        "r_mknee_study": "r_mknee",
        "L_mknee_study": "L_mknee",
        "r_ankle_study": "r_ankle",
        "L_ankle_study": "L_ankle",
        "r_mankle_study": "r_mankle",
        "L_mankle_study": "L_mankle",
        "r_calc_study": "r_calc",
        "L_calc_study": "L_calc",
        "r_toe_study": "r_toe",
        "L_toe_study": "L_toe",
        "r_5meta_study": "r_5meta",
        "L_5meta_study": "L_5meta",
        "r_thigh1_study": "r_thigh1",
        "r_thigh2_study": "r_thigh2",
        "r_thigh3_study": "r_thigh3",
        "L_thigh1_study": "L_thigh1",
        "L_thigh2_study": "L_thigh2",
        "L_thigh3_study": "L_thigh3",
        "r_sh1_study": "r_sh1",
        "r_sh2_study": "r_sh2",
        "r_sh3_study": "r_sh3",
        "L_sh1_study": "L_sh1",
        "L_sh2_study": "L_sh2",
        "L_sh3_study": "L_sh3",
        "RHJC_study": "R_HJC",
        "LHJC_study": "L_HJC",
        "r_lelbow_study": "r_lelbow",
        "L_lelbow_study": "L_lelbow",
        "r_melbow_study": "r_melbow",
        "L_melbow_study": "L_melbow",
        "r_lwrist_study": "r_lwrist",
        "L_lwrist_study": "L_lwrist",
        "r_mwrist_study": "r_mwrist",
        "L_mwrist_study": "L_mwrist",
    }

    monocular_mapping = {
        "C7": "C7",
        "sternum": "sternum",
        "L4": "L4",
        "T6": "T6",
        "r_ASIS": "r.ASIS",
        "l_ASIS": "L.ASIS",
        "r_PSIS": "r.PSIS",
        "l_PSIS": "L.PSIS",
        "r_knee": "r_knee",
        "l_knee": "L_knee",
        "r_mknee": "r_mknee",
        "l_mknee": "L_mknee",
        "r_ankle": "r_ankle",
        "l_ankle": "L_ankle",
        "r_mankle": "r_mankle",
        "l_mankle": "L_mankle",
        "r_calc": "r_calc",
        "l_calc": "L_calc",
        "r_toe": "r_toe",
        "l_toe": "L_toe",
        "r_5meta": "r_5meta",
        "l_5meta": "L_5meta",
        "r_big_toe": "r_toe",
        "l_big_toe": "L_toe",
        "r_shoulder": "R_Shoulder",
        "l_shoulder": "L_Shoulder",
        "r_elbow": "r_lelbow",
        "l_elbow": "L_lelbow",
        "r_melbow": "r_melbow",
        "l_melbow": "L_melbow",
        "r_wrist_radius": "r_lwrist",
        "l_wrist_radius": "L_lwrist",
        "r_wrist_ulna": "r_mwrist",
        "l_wrist_ulna": "L_mwrist",
    }

    PREFERRED_SOURCE_FOR_TARGET = {
        "R_Shoulder": "r_shoulder",
        "L_Shoulder": "l_shoulder",
        "r_calc": "r_calc",
        "L_calc": "l_calc",
        "r_toe": "r_toe",
        "L_toe": "l_toe",
        "r_lelbow": "r_elbow",
        "L_lelbow": "l_elbow",
        "r_lwrist": "r_wrist_radius",
        "L_lwrist": "l_wrist_radius",
    }

    with open(static_trc_path, "r") as f:
        lines = f.readlines()

    data_start_idx = _find_trc_marker_header_index(lines)
    if data_start_idx is None:
        raise ValueError(f"Could not find TRC column header line in: {static_trc_path}")

    header_line = lines[data_start_idx].rstrip("\n\r").split("\t")
    all_marker_names = [name for name in header_line if name][2:]

    has_offset_removed = any("offsetremoved" in n.lower() for n in all_marker_names)
    has_study_markers = any(n.lower().endswith("_study") or "_study" in n.lower() for n in all_marker_names)

    if has_offset_removed:
        mapping = marker_mapping_offset
    elif has_study_markers:
        mapping = marker_mapping_legacy
    else:
        mapping = monocular_mapping

    keys_in_trc = [k for k in mapping.keys() if k in all_marker_names]
    targets = list(dict.fromkeys(mapping.values()))
    vals_in_trc = [v for v in targets if v in all_marker_names]

    if len(keys_in_trc) >= 10:
        chosen_for_target = {}
        for src in keys_in_trc:
            tgt = mapping[src]
            if tgt not in chosen_for_target:
                chosen_for_target[tgt] = src
                continue
            preferred_src = PREFERRED_SOURCE_FOR_TARGET.get(tgt, None)
            if preferred_src is not None and src == preferred_src:
                chosen_for_target[tgt] = src

        for tgt, preferred_src in PREFERRED_SOURCE_FOR_TARGET.items():
            if tgt in chosen_for_target and preferred_src in keys_in_trc:
                chosen_for_target[tgt] = preferred_src

        final_marker_names = list(chosen_for_target.keys())
        trc_marker_names_to_extract = [chosen_for_target[t] for t in final_marker_names]

    elif len(vals_in_trc) >= 10:
        final_marker_names = vals_in_trc[:]
        trc_marker_names_to_extract = vals_in_trc[:]
    else:
        raise ValueError(
            "Too few usable markers found in TRC for scaling "
            f"({max(len(keys_in_trc), len(vals_in_trc))} found). "
            f"TRC markers look like: {all_marker_names[:10]}"
        )

    if len(final_marker_names) < 10:
        raise ValueError(f"Too few unique markers after mapping/dedup for scaling ({len(final_marker_names)}).")
    if len(set(final_marker_names)) != len(final_marker_names):
        dups = sorted({m for m in final_marker_names if final_marker_names.count(m) > 1})
        raise ValueError(f"Internal error: duplicate final marker names remain: {dups}")

    data_lines = lines[data_start_idx + 2 :]
    data = []
    for line in data_lines:
        if not line.strip():
            continue
        parts = line.strip().split("\t")
        row = []
        for val in parts:
            try:
                row.append(float(val))
            except ValueError:
                pass
        if row:
            data.append(row)

    if not data:
        raise ValueError("No numeric data found in TRC file.")
    data = np.array(data)

    if data.shape[1] < 2:
        raise ValueError("TRC data has too few columns for frame/time and markers.")

    OC_time = data[:, 1]
    OC_time_zeroed = OC_time - np.min(OC_time)

    marker_indices = []
    for m in trc_marker_names_to_extract:
        if m not in all_marker_names:
            raise ValueError(f"Requested marker '{m}' not found in TRC header.")
        marker_indices.append(all_marker_names.index(m))

    OC_mrkdata_specific = []
    for idx in marker_indices:
        c0 = 2 + idx * 3
        if c0 + 2 >= data.shape[1]:
            raise ValueError(f"Marker '{all_marker_names[idx]}' index {idx} out of range for TRC data columns.")
        OC_mrkdata_specific.extend([data[:, c0], data[:, c0 + 1], data[:, c0 + 2]])

    OC_mrkdata_specific = np.array(OC_mrkdata_specific).T

    trc_filename = "OpenCap_static_LaiUhlrich_markers.trc"
    processed_trc_path = os.path.join(trc_dir, trc_filename)

    with open(processed_trc_path, "w") as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{trc_filename}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write("100.00\t100.00\t{}\t{}\tmm\t100.00\t1\t{}\n".format(len(OC_time_zeroed), len(final_marker_names), len(OC_time_zeroed)))
        f.write("Frame#\tTime\t")
        f.write("\t\t".join(final_marker_names))
        f.write("\n")
        f.write("\t\t")
        for i_m in range(len(final_marker_names)):
            f.write(f"X{i_m+1}\tY{i_m+1}\tZ{i_m+1}")
            if i_m < len(final_marker_names) - 1:
                f.write("\t")
        f.write("\n")
        for i_fr in range(len(OC_time_zeroed)):
            f.write(f"{i_fr+1}\t{OC_time_zeroed[i_fr]:.6f}")
            for j in range(OC_mrkdata_specific.shape[1] // 3):
                k = 3 * j
                f.write("\t{:.6f}\t{:.6f}\t{:.6f}".format(
                    OC_mrkdata_specific[i_fr, k],
                    OC_mrkdata_specific[i_fr, k + 1],
                    OC_mrkdata_specific[i_fr, k + 2],
                ))
            f.write("\n")

    if not os.path.exists(generic_scale_setup_xml):
        raise FileNotFoundError(f"Scale setup XML not found: {generic_scale_setup_xml}")

    scale_tool = opensim.ScaleTool(generic_scale_setup_xml)
    scale_tool.setSubjectMass(mass_kg)
    scale_tool.setSubjectHeight(height_m * 1000.0)
    scale_tool.getModelScaler().setMarkerFileName(processed_trc_path)
    scale_tool.getMarkerPlacer().setMarkerFileName(processed_trc_path)
    scale_tool.setName("LaiUhlrich2022-scaled_OC")

    scaled_model_path = os.path.join(output_dir, "LaiUhlrich2022_scaled.osim")
    scale_tool.getMarkerPlacer().setOutputModelFileName(scaled_model_path)
    scale_tool.getGenericModelMaker().setModelFileName(generic_model_path)

    setup_xml_path = os.path.join(output_dir, "scale_setup_LaiUhlrich2022.xml")
    scale_tool.printToXML(setup_xml_path)

    files_before = set()
    for root, _, files in os.walk(output_dir):
        for fn in files:
            files_before.add(os.path.join(root, fn))

    cmd = [opensim_cmd_path, "run-tool", setup_xml_path]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=output_dir)

    if result.returncode != 0:
        raise RuntimeError(
            "OpenSim scaling failed with return code "
            f"{result.returncode}\n\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )

    if os.path.exists(scaled_model_path):
        return scaled_model_path, height_m, mass_kg

    osim_files = []
    for root, _, files in os.walk(output_dir):
        for fn in files:
            if fn.endswith(".osim"):
                osim_files.append(os.path.join(root, fn))

    if osim_files:
        prioritized = sorted(osim_files, key=lambda p: (0 if "scaled" in os.path.basename(p).lower() else 1, p))
        return prioritized[0], height_m, mass_kg

    files_after = set()
    for root, _, files in os.walk(output_dir):
        for fn in files:
            files_after.add(os.path.join(root, fn))
    new_files = files_after - files_before

    raise FileNotFoundError(
        "Scaling completed but no scaled model file was created.\n"
        f"Expected: {scaled_model_path}\n"
        f"New files created during scaling: {list(new_files)}"
    )


# =============================================================================
# TRC unit conversion
# =============================================================================
def convert_trc_to_mm_in_place(trc_path):
    with open(trc_path, "r") as f:
        lines = f.readlines()

    if len(lines) < 6:
        raise ValueError(f"TRC file looks too short: {trc_path}")

    delimiter = "\t"
    metadata_keys = lines[1].strip().split(delimiter)
    metadata_vals = lines[2].strip().split(delimiter)

    units_idx = None
    for i, key in enumerate(metadata_keys):
        if key.strip().lower() == "units":
            units_idx = i
            break
    if units_idx is None:
        raise ValueError(f"Could not find 'Units' field in TRC header for {trc_path}")

    current_units = metadata_vals[units_idx].strip().lower()
    if current_units in ("mm", "millimeter", "millimeters"):
        return False

    if current_units not in ("m", "meter", "meters"):
        raise ValueError(
            f"Units field is '{metadata_vals[units_idx]}' in {trc_path}, expected 'm' or 'mm'. Not converting."
        )

    frame_idx = _find_trc_marker_header_index(lines)
    if frame_idx is None:
        raise ValueError(f"Could not find 'Frame#' line in TRC header for {trc_path}")

    if frame_idx + 2 < len(lines) and lines[frame_idx + 2].strip() == "":
        start_row = frame_idx + 3
    else:
        start_row = frame_idx + 2

    data_lines = lines[start_row:]
    data = np.genfromtxt(data_lines, delimiter=delimiter, filling_values=np.nan)
    if data.ndim == 1:
        data = data[None, :]

    if data.shape[1] < 3:
        raise ValueError(f"TRC data in {trc_path} has too few columns to contain markers.")

    data[:, 2:] *= 1000.0
    metadata_vals[units_idx] = "mm"
    lines[2] = delimiter.join(metadata_vals) + "\n"

    new_data_lines = []
    for row in data:
        frame = int(row[0])
        time = row[1]
        coords = row[2:]
        line = f"{frame}\t{time:.7f}"
        line += "".join(f"\t{val:.6f}" for val in coords)
        line += "\n"
        new_data_lines.append(line)

    with open(trc_path, "w") as f:
        f.writelines(lines[:start_row])
        f.writelines(new_data_lines)

    return True


def convert_all_trc_in_folder_to_mm(folder):
    folder = os.path.abspath(folder)
    for name in os.listdir(folder):
        if not name.lower().endswith(".trc"):
            continue
        trc_path = os.path.join(folder, name)
        try:
            convert_trc_to_mm_in_place(trc_path)
        except Exception as e:
            print(f"  Skipped {name} due to error: {e}")


# =============================================================================
# Model scaling (Rajagopal Armless)
# =============================================================================

def create_rajogopal_from_laiuhlrich(
    generic_lai_path: str,
    scaled_lai_path: str,
    generic_ra_path: str,
    generic_ra_scale_setup_xml: str,
    session_metadata_path: str,
    output_dir: str,
    opensim_install_dir: str = None,
):
    """
    Create a scaled Rajogopal Armless model by transferring scale factors from an
    existing scaled LaiUhlrich2022 model.  No static TRC is required.

    Scale factors are derived by comparing joint-frame translations (parent
    PhysicalOffsetFrame translations) between the generic and scaled LaiUhlrich
    models.  These are geometric scale factors unaffected by OpenSim's total-mass
    normalisation step.  Bodies that have no usable joint translations (torso, toes)
    fall back to mass-ratio estimates that are corrected by the ratio between
    joint-based and mass-based factors for the bodies where both are available.
    """
    import xml.etree.ElementTree as ET
    from collections import defaultdict

    os.makedirs(output_dir, exist_ok=True)

    opensim_cmd_path = None
    if opensim_install_dir:
        candidate = os.path.join(opensim_install_dir, "bin", "opensim-cmd.exe")
        if os.path.exists(candidate):
            opensim_cmd_path = candidate
    if not opensim_cmd_path:
        opensim_cmd_path = find_opensim_cmd()
    if not opensim_cmd_path:
        raise FileNotFoundError(
            "Could not find opensim-cmd.exe. Ensure OpenSim is installed or provide opensim_install_dir."
        )

    # Read subject height/mass from sessionMetadata.yaml
    height_m, mass_kg = None, None
    with open(session_metadata_path, "r") as _f:
        for _line in _f:
            _line = _line.strip()
            if "height_m:" in _line:
                try:
                    height_m = float(_line.split("height_m:")[-1].strip())
                except ValueError:
                    pass
            if "mass_kg:" in _line:
                try:
                    mass_kg = float(_line.split("mass_kg:")[-1].strip())
                except ValueError:
                    pass
    if height_m is None or mass_kg is None:
        raise ValueError(f"Could not read height/mass from {session_metadata_path}")

    # ------------------------------------------------------------------
    # Helper: parse all PhysicalOffsetFrame translations across every joint type
    # ------------------------------------------------------------------
    def _parse_frames(model_path):
        tree = ET.parse(model_path)
        root = tree.getroot()
        result = {}  # (joint_name, frame_index) -> (body_name, [tx, ty, tz])
        for tag in ("CustomJoint", "PinJoint", "WeldJoint"):
            for joint in root.iter(tag):
                jname = joint.get("name", "")
                for idx, frame in enumerate(joint.iter("PhysicalOffsetFrame")):
                    s = frame.find("socket_parent")
                    t = frame.find("translation")
                    if s is None or t is None:
                        continue
                    body = s.text.strip().split("/")[-1]
                    trans = [float(x) for x in t.text.strip().split()]
                    result[(jname, idx)] = (body, trans)
        return result

    def _parse_masses(model_path):
        tree = ET.parse(model_path)
        root = tree.getroot()
        masses = {}
        for body in root.iter("Body"):
            name = body.get("name", "")
            m = body.find("mass")
            if name and m is not None:
                try:
                    masses[name] = float(m.text.strip())
                except (ValueError, AttributeError):
                    pass
        return masses

    # ------------------------------------------------------------------
    # Compute per-body, per-axis scale factors from joint translations
    # ------------------------------------------------------------------
    gen_frames = _parse_frames(generic_lai_path)
    sca_frames = _parse_frames(scaled_lai_path)

    body_axis_ratios = defaultdict(lambda: [[], [], []])
    for key, (gb, gt) in gen_frames.items():
        if key not in sca_frames:
            continue
        _, st = sca_frames[key]
        for ax in range(3):
            gv, sv = gt[ax], st[ax]
            if abs(gv) > 0.001:
                ratio = sv / gv
                if 0.5 < ratio < 2.5:  # reject implausible values
                    body_axis_ratios[gb][ax].append(ratio)

    # Median per axis per body (robust to multiple joints feeding the same body)
    sf_joint = {}
    for body, (rx, ry, rz) in body_axis_ratios.items():
        sfs = []
        for r_list in (rx, ry, rz):
            sfs.append(float(np.median(r_list)) if r_list else None)
        sf_joint[body] = sfs  # [sf_x|None, sf_y|None, sf_z|None]

    # ------------------------------------------------------------------
    # Correct mass-based scale factors using joint-based ones as reference
    # ------------------------------------------------------------------
    gen_masses = _parse_masses(generic_lai_path)
    sca_masses = _parse_masses(scaled_lai_path)

    mass_sf = {}
    for bname, gm in gen_masses.items():
        if bname in sca_masses and gm > 1e-6:
            mass_sf[bname] = (sca_masses[bname] / gm) ** (1.0 / 3.0)

    # K = systematic bias in mass-ratio estimates relative to joint-based truth
    corrections = []
    for bname, joint_sfs in sf_joint.items():
        known_joint = [v for v in joint_sfs if v is not None]
        if known_joint and bname in mass_sf:
            corrections.append(mass_sf[bname] / float(np.median(known_joint)))
    K = float(np.median(corrections)) if corrections else 1.0

    # ------------------------------------------------------------------
    # Build final [sf_x, sf_y, sf_z] for every body
    # ------------------------------------------------------------------
    def _resolve(body):
        if body in sf_joint:
            sfs = sf_joint[body]
            known = [v for v in sfs if v is not None]
            fill = float(np.median(known)) if known else (mass_sf.get(body, 1.0) / K)
            return [v if v is not None else fill for v in sfs]
        if body in mass_sf:
            sf = mass_sf[body] / K
            return [sf, sf, sf]
        return [1.0, 1.0, 1.0]

    # Get all body names from the Rajogopal armless generic
    ra_masses = _parse_masses(generic_ra_path)

    # ------------------------------------------------------------------
    # Modify the Rajogopal scale setup XML: use manualscale + our ScaleSet,
    # disable MarkerPlacer (no TRC needed)
    # ------------------------------------------------------------------
    ET.register_namespace("", "")  # avoid ns0: prefixes
    tree = ET.parse(generic_ra_scale_setup_xml)
    xml_root = tree.getroot()

    # Locate ScaleTool element (may be the root or nested under OpenSimDocument)
    st_elem = xml_root if xml_root.tag == "ScaleTool" else xml_root.find(".//ScaleTool")
    if st_elem is None:
        st_elem = xml_root

    # Set mass / height
    for tag, val in [("mass", str(mass_kg)), ("height", str(height_m * 1000.0))]:
        e = st_elem.find(tag)
        if e is None:
            e = ET.SubElement(st_elem, tag)
        e.text = val

    # Generic model path
    gmm = st_elem.find(".//GenericModelMaker")
    if gmm is not None:
        mf = gmm.find("model_file")
        if mf is None:
            mf = ET.SubElement(gmm, "model_file")
        mf.text = generic_ra_path

    # ModelScaler: switch to manualscale, populate ScaleSet, clear marker_file
    ms = st_elem.find(".//ModelScaler")
    if ms is not None:
        so = ms.find("scaling_order")
        if so is None:
            so = ET.SubElement(ms, "scaling_order")
        so.text = " manualScale"

        ss = ms.find("ScaleSet")
        if ss is None:
            ss = ET.SubElement(ms, "ScaleSet")
        objs = ss.find("objects")
        if objs is None:
            objs = ET.SubElement(ss, "objects")
        else:
            objs.clear()

        for bname in ra_masses:
            sfxyz = _resolve(bname)
            sc = ET.SubElement(objs, "Scale")
            ET.SubElement(sc, "scales").text = f"{sfxyz[0]:.8f} {sfxyz[1]:.8f} {sfxyz[2]:.8f}"
            ET.SubElement(sc, "segment").text = bname
            ET.SubElement(sc, "apply").text = "true"

        # Clear stale marker file so opensim-cmd won't choke
        mf_elem = ms.find("marker_file")
        if mf_elem is not None:
            mf_elem.text = "Unassigned"

        scaled_model_path = os.path.join(output_dir, "scaled_RagagopalArmless.osim")
        omf = ms.find("output_model_file")
        if omf is None:
            omf = ET.SubElement(ms, "output_model_file")
        omf.text = scaled_model_path

    # Disable MarkerPlacer (no TRC)
    mp = st_elem.find(".//MarkerPlacer")
    if mp is not None:
        apply_elem = mp.find("apply")
        if apply_elem is None:
            apply_elem = ET.SubElement(mp, "apply")
        apply_elem.text = "false"
        # Also set its output path in case opensim still references it
        omf_mp = mp.find("output_model_file")
        if omf_mp is None:
            omf_mp = ET.SubElement(mp, "output_model_file")
        omf_mp.text = scaled_model_path

    setup_xml_path = os.path.join(output_dir, "scale_setup_RagagopalArmless_fromLai.xml")
    tree.write(setup_xml_path, xml_declaration=True, encoding="unicode")

    cmd = [opensim_cmd_path, "run-tool", setup_xml_path]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=output_dir)

    # Check for the output file first — OpenSim may return exit code 1 if the
    # disabled MarkerPlacer stage reports an error even though ModelScaler already
    # wrote the scaled model successfully.
    if os.path.exists(scaled_model_path):
        return scaled_model_path, height_m, mass_kg

    if result.returncode != 0:
        raise RuntimeError(
            f"OpenSim scaling failed (return code {result.returncode}).\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Fallback: find any newly created .osim
    osim_files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".osim")
    ]
    if osim_files:
        best = sorted(osim_files, key=lambda p: (0 if "scaled" in os.path.basename(p).lower() else 1, p))
        return best[0], height_m, mass_kg

    raise FileNotFoundError(
        f"Scaling completed but no scaled model file was created in {output_dir}.\n"
        f"Expected: {scaled_model_path}"
    )


def create_rajogopal_armless_model(
    generic_model_path: str,
    generic_scale_setup_xml: str,
    session_metadata_path: str,
    static_trc_path: str,
    output_dir: str,
    opensim_install_dir: str = None,
):
    os.makedirs(output_dir, exist_ok=True)
    trc_dir = os.path.join(os.path.dirname(session_metadata_path), "ForGaitDynamics", "TRC_Files")
    os.makedirs(trc_dir, exist_ok=True)

    static_trc_path = deduplicate_trc_markers(static_trc_path, trc_out=None, verbose=False)

    opensim_cmd_path = None
    if opensim_install_dir:
        candidate_path = os.path.join(opensim_install_dir, "bin", "opensim-cmd.exe")
        if os.path.exists(candidate_path):
            opensim_cmd_path = candidate_path
    if not opensim_cmd_path:
        opensim_cmd_path = find_opensim_cmd()
    if not opensim_cmd_path:
        raise FileNotFoundError(
            "Could not find opensim-cmd.exe. Please ensure OpenSim is installed and either provide "
            "opensim_install_dir or add OpenSim to your PATH."
        )

    height_m = None
    mass_kg = None
    with open(session_metadata_path, "r") as f:
        for line in f:
            line = line.strip()
            if "height_m:" in line:
                try:
                    height_m = float(line.split("height_m:")[-1].strip())
                except ValueError:
                    pass
            if "mass_kg:" in line:
                try:
                    mass_kg = float(line.split("mass_kg:")[-1].strip())
                except ValueError:
                    pass
    if height_m is None or mass_kg is None:
        raise ValueError(f"Could not read height or mass from {session_metadata_path}")

    marker_mapping_offset = {
        "C7_study_offsetRemoved": "C7",
        "r_shoulder_study_offsetRemoved": "R_Shoulder",
        "L_shoulder_study_offsetRemoved": "L_Shoulder",
        "r.ASIS_study_offsetRemoved": "r.ASIS",
        "L.ASIS_study_offsetRemoved": "L.ASIS",
        "r.PSIS_study_offsetRemoved": "r.PSIS",
        "L.PSIS_study_offsetRemoved": "L.PSIS",
        "r_knee_study_offsetRemoved": "r_knee",
        "L_knee_study_offsetRemoved": "L_knee",
        "r_mknee_study_offsetRemoved": "r_mknee",
        "L_mknee_study_offsetRemoved": "L_mknee",
        "r_ankle_study_offsetRemoved": "r_ankle",
        "L_ankle_study_offsetRemoved": "L_ankle",
        "r_mankle_study_offsetRemoved": "r_mankle",
        "L_mankle_study_offsetRemoved": "L_mankle",
        "r_calc_study_offsetRemoved": "r_calc",
        "L_calc_study_offsetRemoved": "L_calc",
        "r_toe_study_offsetRemoved": "r_toe",
        "L_toe_study_offsetRemoved": "L_toe",
        "r_5meta_study_offsetRemoved": "r_5meta",
        "L_5meta_study_offsetRemoved": "L_5meta",
        "r_thigh1_study_offsetRemoved": "r_thigh1",
        "r_thigh2_study_offsetRemoved": "r_thigh2",
        "r_thigh3_study_offsetRemoved": "r_thigh3",
        "L_thigh1_study_offsetRemoved": "L_thigh1",
        "L_thigh2_study_offsetRemoved": "L_thigh2",
        "L_thigh3_study_offsetRemoved": "L_thigh3",
        "r_sh1_study_offsetRemoved": "r_sh1",
        "r_sh2_study_offsetRemoved": "r_sh2",
        "r_sh3_study_offsetRemoved": "r_sh3",
        "L_sh1_study_offsetRemoved": "L_sh1",
        "L_sh2_study_offsetRemoved": "L_sh2",
        "L_sh3_study_offsetRemoved": "L_sh3",
        "RHJC_study_offsetRemoved": "R_HJC",
        "LHJC_study_offsetRemoved": "L_HJC",
        "r_lelbow_study_offsetRemoved": "r_lelbow",
        "L_lelbow_study_offsetRemoved": "L_lelbow",
        "r_melbow_study_offsetRemoved": "r_melbow",
        "L_melbow_study_offsetRemoved": "L_melbow",
        "r_lwrist_study_offsetRemoved": "r_lwrist",
        "L_lwrist_study_offsetRemoved": "L_lwrist",
        "r_mwrist_study_offsetRemoved": "r_mwrist",
        "L_mwrist_study_offsetRemoved": "L_mwrist",
    }

    marker_mapping_legacy = {
        "C7_study": "C7",
        "r_shoulder_study": "R_Shoulder",
        "L_shoulder_study": "L_Shoulder",
        "r.ASIS_study": "r.ASIS",
        "L.ASIS_study": "L.ASIS",
        "r.PSIS_study": "r.PSIS",
        "L.PSIS_study": "L.PSIS",
        "r_knee_study": "r_knee",
        "L_knee_study": "L_knee",
        "r_mknee_study": "r_mknee",
        "L_mknee_study": "L_mknee",
        "r_ankle_study": "r_ankle",
        "L_ankle_study": "L_ankle",
        "r_mankle_study": "r_mankle",
        "L_mankle_study": "L_mankle",
        "r_calc_study": "r_calc",
        "L_calc_study": "L_calc",
        "r_toe_study": "r_toe",
        "L_toe_study": "L_toe",
        "r_5meta_study": "r_5meta",
        "L_5meta_study": "L_5meta",
        "r_thigh1_study": "r_thigh1",
        "r_thigh2_study": "r_thigh2",
        "r_thigh3_study": "r_thigh3",
        "L_thigh1_study": "L_thigh1",
        "L_thigh2_study": "L_thigh2",
        "L_thigh3_study": "L_thigh3",
        "r_sh1_study": "r_sh1",
        "r_sh2_study": "r_sh2",
        "r_sh3_study": "r_sh3",
        "L_sh1_study": "L_sh1",
        "L_sh2_study": "L_sh2",
        "L_sh3_study": "L_sh3",
        "RHJC_study": "R_HJC",
        "LHJC_study": "L_HJC",
        "r_lelbow_study": "r_lelbow",
        "L_lelbow_study": "L_lelbow",
        "r_melbow_study": "r_melbow",
        "L_melbow_study": "L_melbow",
        "r_lwrist_study": "r_lwrist",
        "L_lwrist_study": "L_lwrist",
        "r_mwrist_study": "r_mwrist",
        "L_mwrist_study": "L_mwrist",
    }

    monocular_mapping = {
        "C7": "C7",
        "sternum": "sternum",
        "L4": "L4",
        "T6": "T6",
        "r_ASIS": "r.ASIS",
        "l_ASIS": "L.ASIS",
        "r_PSIS": "r.PSIS",
        "l_PSIS": "L.PSIS",
        "r_knee": "r_knee",
        "l_knee": "L_knee",
        "r_mknee": "r_mknee",
        "l_mknee": "L_mknee",
        "r_ankle": "r_ankle",
        "l_ankle": "L_ankle",
        "r_mankle": "r_mankle",
        "l_mankle": "L_mankle",
        "r_calc": "r_calc",
        "l_calc": "L_calc",
        "r_toe": "r_toe",
        "l_toe": "L_toe",
        "r_5meta": "r_5meta",
        "l_5meta": "L_5meta",
        "r_big_toe": "r_toe",
        "l_big_toe": "L_toe",
        "r_shoulder": "R_Shoulder",
        "l_shoulder": "L_Shoulder",
        "r_elbow": "r_lelbow",
        "l_elbow": "L_lelbow",
        "r_melbow": "r_melbow",
        "l_melbow": "L_melbow",
        "r_wrist_radius": "r_lwrist",
        "l_wrist_radius": "L_lwrist",
        "r_wrist_ulna": "r_mwrist",
        "l_wrist_ulna": "L_mwrist",
    }

    PREFERRED_SOURCE_FOR_TARGET = {
        "R_Shoulder": "r_shoulder",
        "L_Shoulder": "l_shoulder",
        "r_calc": "r_calc",
        "L_calc": "l_calc",
        "r_toe": "r_toe",
        "L_toe": "l_toe",
        "r_lelbow": "r_elbow",
        "L_lelbow": "l_elbow",
        "r_lwrist": "r_wrist_radius",
        "L_lwrist": "l_wrist_radius",
    }

    with open(static_trc_path, "r") as f:
        lines = f.readlines()

    data_start_idx = _find_trc_marker_header_index(lines)
    if data_start_idx is None:
        raise ValueError(f"Could not find TRC column header line in: {static_trc_path}")

    header_line = lines[data_start_idx].strip().split("\t")
    all_marker_names = [name for name in header_line if name][2:]

    has_offset_removed = any("offsetremoved" in name.lower() for name in all_marker_names)
    has_study_markers = any(name.lower().endswith("_study") or "_study" in name.lower() for name in all_marker_names)

    if has_offset_removed:
        marker_mapping = marker_mapping_offset
    elif has_study_markers:
        marker_mapping = marker_mapping_legacy
    else:
        marker_mapping = monocular_mapping

    keys_in_trc = [k for k in marker_mapping.keys() if k in all_marker_names]
    vals_in_trc = [v for v in sorted(set(marker_mapping.values())) if v in all_marker_names]

    if len(keys_in_trc) >= 10:
        chosen_for_target = {}
        for src in keys_in_trc:
            tgt = marker_mapping[src]
            if tgt not in chosen_for_target:
                chosen_for_target[tgt] = src
                continue
            preferred_src = PREFERRED_SOURCE_FOR_TARGET.get(tgt, None)
            if preferred_src is not None and src == preferred_src:
                chosen_for_target[tgt] = src

        for tgt, preferred_src in PREFERRED_SOURCE_FOR_TARGET.items():
            if tgt in chosen_for_target and preferred_src in keys_in_trc:
                chosen_for_target[tgt] = preferred_src

        final_marker_names = list(chosen_for_target.keys())
        trc_marker_names_to_extract = [chosen_for_target[t] for t in final_marker_names]

    elif len(vals_in_trc) >= 10:
        final_marker_names = []
        seen = set()
        for v in vals_in_trc:
            if v not in seen:
                final_marker_names.append(v)
                seen.add(v)
        trc_marker_names_to_extract = final_marker_names[:]

    else:
        raise ValueError(
            "Too few usable markers found in TRC for scaling "
            f"({max(len(keys_in_trc), len(vals_in_trc))} found)."
        )

    if len(final_marker_names) < 10:
        raise ValueError(f"Too few unique markers after mapping/dedup for scaling ({len(final_marker_names)}).")
    if len(set(final_marker_names)) != len(final_marker_names):
        dups = sorted({m for m in final_marker_names if final_marker_names.count(m) > 1})
        raise ValueError(f"Internal error: duplicate final marker names remain: {dups}")

    data_lines = lines[data_start_idx + 2 :]
    data = []
    for line in data_lines:
        if not line.strip():
            continue
        parts = line.strip().split("\t")
        row = []
        for val in parts:
            try:
                row.append(float(val))
            except ValueError:
                continue
        if row:
            data.append(row)

    if not data:
        raise ValueError("No numeric data found in TRC file.")
    data = np.array(data)
    if data.shape[1] < 2:
        raise ValueError("TRC data has too few columns for time and markers.")

    OC_time = data[:, 1]
    marker_indices = [all_marker_names.index(m) for m in trc_marker_names_to_extract]

    OC_mrkdata_specific = []
    for idx in marker_indices:
        c0 = 2 + idx * 3
        if c0 + 2 >= data.shape[1]:
            raise ValueError(f"Marker index {idx} out of range for TRC data columns.")
        OC_mrkdata_specific.extend([data[:, c0], data[:, c0 + 1], data[:, c0 + 2]])

    OC_mrkdata_specific = np.array(OC_mrkdata_specific).T
    OC_time_zeroed = OC_time - np.min(OC_time)

    trc_filename = "OpenCap_static_specific_markers.trc"
    processed_trc_path = os.path.join(trc_dir, trc_filename)

    with open(processed_trc_path, "w") as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{trc_filename}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write("100.00\t100.00\t{}\t{}\tmm\t100.00\t1\t{}\n".format(len(OC_time_zeroed), len(final_marker_names), len(OC_time_zeroed)))
        f.write("Frame#\tTime\t")
        f.write("\t\t".join(final_marker_names))
        f.write("\n")
        f.write("\t\t")
        for i in range(len(final_marker_names)):
            f.write(f"X{i+1}\tY{i+1}\tZ{i+1}")
            if i < len(final_marker_names) - 1:
                f.write("\t")
        f.write("\n")
        for i in range(len(OC_time_zeroed)):
            f.write(f"{i+1}\t{OC_time_zeroed[i]:.6f}")
            for j in range(OC_mrkdata_specific.shape[1] // 3):
                marker_idx = j * 3
                f.write("\t{:.6f}\t{:.6f}\t{:.6f}".format(
                    OC_mrkdata_specific[i, marker_idx],
                    OC_mrkdata_specific[i, marker_idx + 1],
                    OC_mrkdata_specific[i, marker_idx + 2],
                ))
            f.write("\n")

    if not os.path.exists(generic_scale_setup_xml):
        raise FileNotFoundError(f"Scale setup XML not found: {generic_scale_setup_xml}")

    scale_tool = opensim.ScaleTool(generic_scale_setup_xml)
    scale_tool.setSubjectMass(mass_kg)
    scale_tool.setSubjectHeight(height_m * 1000.0)
    scale_tool.getModelScaler().setMarkerFileName(processed_trc_path)
    scale_tool.getMarkerPlacer().setMarkerFileName(processed_trc_path)
    scale_tool.setName("ArmlessRajagopal-scaled_OC")

    scaled_model_path = os.path.join(output_dir, "scaled_RagagopalArmless.osim")
    scale_tool.getMarkerPlacer().setOutputModelFileName(scaled_model_path)
    scale_tool.getGenericModelMaker().setModelFileName(generic_model_path)

    setup_xml_path = os.path.join(output_dir, "scale_setup_RagagopalArmless.xml")
    scale_tool.printToXML(setup_xml_path)

    files_before = set()
    for root, _, files in os.walk(output_dir):
        for file in files:
            files_before.add(os.path.join(root, file))

    cmd = [opensim_cmd_path, "run-tool", setup_xml_path]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=output_dir)

    if result.returncode != 0:
        raise RuntimeError(
            "OpenSim scaling failed with return code "
            f"{result.returncode}\n\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )

    if os.path.exists(scaled_model_path):
        return scaled_model_path, height_m, mass_kg

    osim_files = []
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".osim"):
                osim_files.append(os.path.join(root, file))

    if osim_files:
        prioritized = sorted(osim_files, key=lambda p: (0 if "scaled" in os.path.basename(p).lower() else 1, p))
        return prioritized[0], height_m, mass_kg

    files_after = set()
    for root, _, files in os.walk(output_dir):
        for file in files:
            files_after.add(os.path.join(root, file))

    new_files = files_after - files_before
    raise FileNotFoundError(
        "Scaling completed but no scaled model file was created.\n"
        f"Expected: {scaled_model_path}\n"
        f"New files created during scaling: {list(new_files)}"
    )


# =============================================================================
# IK + MOT utilities
# =============================================================================
def reformat_ik_mot_for_gaitdynamics(ik_path):
    ik_path = Path(ik_path)
    script_dir = Path(__file__).resolve().parent
    template_path = Path(os.path.join(script_dir, "OpenSimPipeline", "ForGaitDynamics", "GaitDynamics_Template_IK.mot"))
    if not template_path.is_file():
        raise FileNotFoundError(
            f"Template file not found: {template_path}. Ensure GaitDynamics_Template_IK.mot exists."
        )

    with template_path.open("r") as f:
        template_header_lines = []
        template_cols_line = None
        for line in f:
            template_header_lines.append(line)
            if line.strip() == "endheader":
                template_cols_line = f.readline()
                break
        if template_cols_line is None:
            raise ValueError("Template file missing column header after endheader")

    with ik_path.open("r") as f:
        ik_cols_line = None
        for line in f:
            if line.strip() == "endheader":
                ik_cols_line = f.readline()
                break
        if ik_cols_line is None:
            raise ValueError("IK file missing column header after endheader")
        data = np.loadtxt(f)

    if data.ndim == 1:
        data = data[np.newaxis, :]

    nrows, ncols = data.shape

    new_header_lines = []
    for line in template_header_lines:
        if line.startswith("nRows="):
            new_header_lines.append(f"nRows={nrows}\n")
        elif line.startswith("nColumns="):
            new_header_lines.append(f"nColumns={ncols}\n")
        else:
            new_header_lines.append(line)

    template_cols = template_cols_line.strip().split()
    ik_cols = ik_cols_line.strip().split()

    if len(template_cols) == ncols:
        cols_line_out = template_cols_line
    elif len(ik_cols) == ncols:
        cols_line_out = ik_cols_line
    else:
        raise ValueError(
            f"Column count mismatch: data has {ncols} columns, template header has {len(template_cols)}, IK header has {len(ik_cols)}"
        )

    with ik_path.open("w") as out:
        for line in new_header_lines:
            out.write(line)
        out.write(cols_line_out)
        for row in data:
            out.write("\t".join(f"{val:.8f}" for val in row) + "\n")


def get_marker_set_names(model: opensim.Model) -> list:
    marker_set = model.getMarkerSet()
    return [marker_set.get(i).getName() for i in range(marker_set.getSize())]


def read_trc_file(trc_path: str):
    with open(trc_path, "r") as f:
        lines = f.readlines()

    line3_parts = lines[2].strip().split()
    data_rate = float(line3_parts[0]) if line3_parts else 100.0

    marker_line = lines[3].rstrip("\n\r")
    marker_tokens = marker_line.split("\t")
    marker_names = [tok for tok in marker_tokens[2:] if tok.strip()]

    time_data = []
    marker_data = []
    for line in lines[6:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            time_data.append(float(parts[1]))
            coords = [float(x) for x in parts[2:]]
            marker_data.append(coords)

    return marker_names, time_data, marker_data, data_rate


def write_trc_file(marker_names, time_data, marker_data, output_path, data_rate=100.0, units="mm"):
    """
    Write a TRC file that OpenSim's TRCFileAdapter can read reliably.

    Key points:
      - Metadata keys line has 8 fields, values line has exactly 8 fields (single tabs).
      - Marker header line uses the standard TRC convention: marker names separated by TWO tabs
        so each marker occupies three columns (X/Y/Z) in the next line.
      - XYZ labels line uses X1/Y1/Z1 ... with single tabs.
      - Writes one blank line after XYZ labels, matching common TRC formatting.
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    num_frames = int(len(time_data))
    num_markers = int(len(marker_names))

    # Sanity checks
    if num_frames == 0:
        raise ValueError("time_data is empty, cannot write TRC.")
    if num_markers == 0:
        raise ValueError("marker_names is empty, cannot write TRC.")
    if len(marker_data) != num_frames:
        raise ValueError(
            f"marker_data length ({len(marker_data)}) must match time_data length ({num_frames})."
        )

    # Ensure each row has at least 3*num_markers values
    # Allow list/np.ndarray
    for i, row in enumerate(marker_data):
        if len(row) < 3 * num_markers:
            raise ValueError(
                f"marker_data row {i} has {len(row)} values, expected at least {3*num_markers} "
                f"for {num_markers} markers (X/Y/Z each)."
            )

    units = units.strip().lower()
    if units not in ("mm", "m"):
        raise ValueError("units must be 'mm' or 'm'")

    # TRC 'PathFileType' 4th field is usually the file name, but OpenSim is tolerant.
    # We'll write the file basename to be conventional.
    trc_basename = os.path.basename(output_path)

    with open(output_path, "w", newline="\n") as f:
        # Line 1
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{trc_basename}\n")

        # Line 2 (8 keys)
        f.write(
            "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n"
        )

        # Line 3 (8 values, SINGLE tabs, no empty fields)
        # Use fixed formatting but keep it simple.
        f.write(
            f"{float(data_rate):.2f}\t{float(data_rate):.2f}\t{num_frames}\t{num_markers}\t{units}\t"
            f"{float(data_rate):.2f}\t1\t{num_frames}\n"
        )

        # Line 4: marker names, two tabs between markers so each marker spans 3 columns
        f.write("Frame#\tTime\t")
        f.write("\t\t".join(marker_names))
        f.write("\n")

        # Line 5: XYZ labels, one tab between each label group
        f.write("\t\t")
        f.write("\t".join([f"X{i}\tY{i}\tZ{i}" for i in range(1, num_markers + 1)]))
        f.write("\n")

        # Line 6: blank line (common TRC formatting)
        f.write("\n")

        # Data rows
        for frame_idx in range(num_frames):
            t = float(time_data[frame_idx])
            row = marker_data[frame_idx]
            coords_to_write = row[: 3 * num_markers]

            f.write(f"{frame_idx + 1}\t{t:.8f}")
            for c in coords_to_write:
                f.write(f"\t{float(c):.6f}")
            f.write("\n")

    return output_path



def reorder_ik_mot_to_opensim_standard(mot_path: str):
    standard_order = [
        "time",
        "pelvis_tilt",
        "pelvis_list",
        "pelvis_rotation",
        "pelvis_tx",
        "pelvis_ty",
        "pelvis_tz",
        "hip_flexion_r",
        "hip_adduction_r",
        "hip_rotation_r",
        "knee_angle_r",
        "knee_angle_r_beta",
        "ankle_angle_r",
        "subtalar_angle_r",
        "mtp_angle_r",
        "hip_flexion_l",
        "hip_adduction_l",
        "hip_rotation_l",
        "knee_angle_l",
        "knee_angle_l_beta",
        "ankle_angle_l",
        "subtalar_angle_l",
        "mtp_angle_l",
        "lumbar_extension",
        "lumbar_bending",
        "lumbar_rotation",
    ]

    if not os.path.exists(mot_path):
        print(f"[reorder_ik_mot] File not found: {mot_path}")
        return None

    header_lines = []
    with open(mot_path, "r") as f:
        for line in f:
            header_lines.append(line)
            if line.strip().lower() == "endheader":
                break
        df = pd.read_csv(f, sep=r"\s+")

    available_cols = [col for col in standard_order if col in df.columns]
    extra_cols = [col for col in df.columns if col not in standard_order]
    reordered_cols = available_cols + extra_cols

    df_reordered = df[reordered_cols]

    for i, line in enumerate(header_lines):
        if line.strip().lower().startswith("ncolumns="):
            header_lines[i] = f"nColumns={len(df_reordered.columns)}\n"
            break

    with open(mot_path, "w") as f:
        f.writelines(header_lines)
        df_reordered.to_csv(f, sep="\t", index=False, float_format="%.8f")

    return reordered_cols


def run_ik_for_gait_dynamics(
    session_folder: str,
    scaled_model_path: str,
    ik_setup_xml: str,
    trial_prefix: str,
    output_dir: str,
    model_type: str,
):
    os.makedirs(output_dir, exist_ok=True)

    marker_dir = os.path.join(session_folder, "ForGaitDynamics")
    if not os.path.isdir(marker_dir):
        raise FileNotFoundError(f"ForGaitDynamics not found: {marker_dir}")

    if not os.path.exists(scaled_model_path):
        raise FileNotFoundError(f"Model not found: {scaled_model_path}")

    ik_outputs = []

    for fname in os.listdir(marker_dir):
        if not fname.lower().endswith(".trc"):
            continue

        trial_name = os.path.splitext(fname)[0]
        if trial_prefix and trial_prefix not in trial_name:
            continue

        input_trc = os.path.join(marker_dir, fname)

        # Safety: ensure no duplicate marker names in TRC (in place)
        deduplicate_trc_markers(input_trc, trc_out=None, verbose=False)

        ik_tool = opensim.InverseKinematicsTool(ik_setup_xml)
        model = opensim.Model(scaled_model_path)
        model.initSystem()
        ik_tool.setModel(model)

        ik_tool.setName(trial_name)
        ik_tool.setMarkerDataFileName(input_trc)

        tt = opensim.TimeSeriesTableVec3(input_trc)
        time_vec = tt.getIndependentColumn()
        if len(time_vec) > 0:
            ik_tool.setStartTime(float(time_vec[0]))
            ik_tool.setEndTime(float(time_vec[-1]))

        trial_name_clean = trial_name.replace("MarkerData_optfeet_", "").replace("_videoAndMocap", "")
        if model_type == "rajogopal":
            ik_output_mot = os.path.join(output_dir, f"{trial_name_clean}_IK_forGaitDynamics.mot")
        elif model_type == "lai":
            ik_output_mot = os.path.join(output_dir, f"{trial_name_clean}_IK_forSimulations.mot")
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        ik_tool.setOutputMotionFileName(ik_output_mot)
        ik_tool.setResultsDir(output_dir)
        ik_tool.run()

        reorder_ik_mot_to_opensim_standard(ik_output_mot)
        lowpass_filter_pelvis_ty(ik_output_mot)

        ik_outputs.append(ik_output_mot)

    return ik_outputs


# =============================================================================
# TRC harmonization to template
# =============================================================================
KNOWN_SUFFIXES = ["_study_offsetRemoved", "_offsetRemoved", "_study"]


def _split_base_suffix(name):
    for suf in KNOWN_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)], suf
    return name, ""


def _candidate_variants(canon_name):
    base, _ = _split_base_suffix(canon_name)
    cands = []
    cands.append(base + "_study_offsetRemoved")
    cands.append(base + "_offsetRemoved")
    cands.append(canon_name)
    cands.append(base + "_study")
    cands.append(base)

    out = []
    seen = set()
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _build_marker_mapping(template_markers, input_markers):
    name_to_idx = {name: i for i, name in enumerate(input_markers)}
    mapping = {}
    missing = []

    for tname in template_markers:
        found_idx = None
        for cand in _candidate_variants(tname):
            if cand in name_to_idx:
                found_idx = name_to_idx[cand]
                break
        mapping[tname] = found_idx
        if found_idx is None:
            missing.append(tname)

    return mapping, missing


def harmonize_trc_markers_to_template(trc_in, template_trc, trc_out=None):
    trc_in = Path(trc_in)
    template_trc = Path(template_trc)

    if trc_out is None:
        trc_out = trc_in.with_name(trc_in.stem + "_harmonized.trc")
    trc_out = Path(trc_out)

    with template_trc.open("r", encoding="utf-8", errors="ignore") as f:
        tmpl_lines = f.readlines()

    tmpl_marker_line = tmpl_lines[3]
    tmpl_coord_line = tmpl_lines[4]

    tmpl_tok = tmpl_marker_line.strip().split("\t")
    template_markers = [t for t in tmpl_tok[2:] if t.strip() != ""]
    n_template = len(template_markers)

    with trc_in.open("r", encoding="utf-8", errors="ignore") as f:
        in_lines = f.readlines()

    in_marker_line = in_lines[3]
    in_tok = in_marker_line.strip().split("\t")
    input_markers = [t for t in in_tok[2:] if t.strip() != ""]

    mapping, missing = _build_marker_mapping(template_markers, input_markers)
    if missing:
        print("Warning: some template markers were not found in input:")
        for m in missing:
            print("  ", m)

    out_lines = []
    out_lines.append(in_lines[0])
    out_lines.append(in_lines[1])

    num_line = in_lines[2].rstrip("\n")
    num_toks = num_line.split("\t")
    if len(num_toks) < 4:
        num_toks += [""] * (4 - len(num_toks))
    num_toks[3] = str(n_template)
    out_lines.append("\t".join(num_toks) + "\n")

    out_lines.append(tmpl_marker_line)
    out_lines.append(tmpl_coord_line)
    out_lines.append("\n")

    data_start = None
    for i in range(5, len(in_lines)):
        if in_lines[i].strip() == "":
            continue
        data_start = i
        break

    if data_start is None:
        with trc_out.open("w", encoding="utf-8") as f:
            f.writelines(out_lines)
        return trc_out

    n_input = len(input_markers)

    for i in range(data_start, len(in_lines)):
        line = in_lines[i].strip()
        if not line:
            continue
        toks = line.split("\t")
        if len(toks) < 2:
            continue

        frame = toks[0]
        time = toks[1]
        row_vals = [frame, time]

        for tname in template_markers:
            idx_in = mapping.get(tname)
            if idx_in is None or idx_in < 0 or idx_in >= n_input:
                row_vals.extend(["0", "0", "0"])
            else:
                base = 2 + 3 * idx_in
                if base + 2 < len(toks):
                    row_vals.extend(toks[base : base + 3])
                else:
                    row_vals.extend(["0", "0", "0"])

        out_lines.append("\t".join(row_vals) + "\n")

    with trc_out.open("w", encoding="utf-8") as f:
        f.writelines(out_lines)

    return trc_out


# =============================================================================
# Filtering and MOT manipulation
# =============================================================================
def lowpass_filter_pelvis_ty(ik_path: str, out_path: str = None, cutoff_hz: float = 2.0, order: int = 4):
    if not os.path.isfile(ik_path):
        raise FileNotFoundError(f"IK file not found: {ik_path}")

    if out_path is None:
        out_path = ik_path

    header_lines = []
    with open(ik_path, "r") as f:
        for line in f:
            header_lines.append(line)
            if line.strip().lower() == "endheader":
                break
        df = pd.read_csv(f, sep=r"\s+")

    if "time" not in df.columns:
        raise ValueError(f"'time' column not found in IK file: {ik_path}")
    if "pelvis_ty" not in df.columns:
        raise ValueError(f"'pelvis_ty' column not found in IK file: {ik_path}")

    time = df["time"].to_numpy(dtype=float)
    if len(time) < 2:
        raise ValueError("Not enough time samples to filter")

    fs = 1.0 / np.mean(np.diff(time))
    wn = cutoff_hz / (fs * 0.5)

    sos = butter(order // 2, wn, btype="low", output="sos")
    pelvis_ty = df["pelvis_ty"].to_numpy(dtype=float)
    df["pelvis_ty"] = sosfiltfilt(sos, pelvis_ty)

    nrows = len(df)
    for i, line in enumerate(header_lines):
        if line.strip().lower().startswith("nrows="):
            header_lines[i] = f"nRows={nrows}\n"
            break

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f_out:
        f_out.writelines(header_lines)
        df.to_csv(f_out, sep="\t", index=False, float_format="%.8f")


def make_long_ik(ik_path: str, out_path: str, min_duration: float = 5.0, copy_if_long: bool = True):
    """
    Ensure the IK .mot file lasts at least `min_duration` seconds.

    Returns:
        added_rows : int
        final_duration : float
    """
    if not os.path.isfile(ik_path):
        print(f"[make_long_ik] IK file not found, skipping: {ik_path}")
        return 0, 0.0

    header_lines = []
    with open(ik_path, "r") as f:
        for line in f:
            header_lines.append(line)
            if line.strip().lower() == "endheader":
                break
        df = pd.read_csv(f, sep=r"\s+")

    if "time" not in df.columns:
        raise ValueError(f"'time' column not found in IK file: {ik_path}")

    t0 = float(df["time"].iloc[0])
    t_last = float(df["time"].iloc[-1])
    duration = t_last - t0

    if duration >= min_duration:
        if copy_if_long:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            shutil.copyfile(ik_path, out_path)
        return 0, duration

    df_long = df.copy()
    if len(df_long) >= 2:
        dt = df_long["time"].iloc[-1] - df_long["time"].iloc[-2]
        if dt <= 0:
            dt = 0.01
    else:
        dt = 0.01

    added_rows = 0
    t_curr = float(df_long["time"].iloc[-1])

    while (t_curr - t0) < min_duration:
        t_curr += dt
        new_row = df_long.iloc[-1].copy()
        new_row["time"] = t_curr
        df_long = pd.concat([df_long, new_row.to_frame().T], ignore_index=True)
        added_rows += 1

    final_duration = t_curr - t0

    nrows = len(df_long)
    for i, line in enumerate(header_lines):
        if line.strip().lower().startswith("nrows="):
            header_lines[i] = f"nRows={nrows}\n"
            break

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f_out:
        f_out.writelines(header_lines)
        df_long.to_csv(f_out, sep="\t", index=False, float_format="%.8f")

    return added_rows, final_duration


def trim_mot_file(mot_path: str, rows_to_trim: int):
    if rows_to_trim <= 0:
        return

    header_lines = []
    with open(mot_path, "r") as f:
        for line in f:
            header_lines.append(line)
            if line.strip().lower() == "endheader":
                break
        df = pd.read_csv(f, sep=r"\s+")

    new_len = len(df) - rows_to_trim
    if new_len <= 0:
        raise ValueError(f"Cannot trim {rows_to_trim} rows from {len(df)} total rows!")

    df_trimmed = df.iloc[:new_len].copy()

    for i, line in enumerate(header_lines):
        if line.strip().lower().startswith("nrows="):
            header_lines[i] = f"nRows={new_len}\n"
            break

    with open(mot_path, "w") as f_out:
        f_out.writelines(header_lines)
        df_trimmed.to_csv(f_out, sep="\t", index=False, float_format="%.8f")


def rename_grf_mot_columns(mot_path: str):
    rename_map = {
        "force_r_vx": "R_ground_force_vx",
        "force_r_vy": "R_ground_force_vy",
        "force_r_vz": "R_ground_force_vz",
        "force_r_px": "R_ground_force_px",
        "force_r_py": "R_ground_force_py",
        "force_r_pz": "R_ground_force_pz",
        "cop_r_x": "R_ground_force_px",
        "cop_r_y": "R_ground_force_py",
        "cop_r_z": "R_ground_force_pz",
        "torque_r_x": "R_ground_torque_x",
        "torque_r_y": "R_ground_torque_y",
        "torque_r_z": "R_ground_torque_z",
        "force_l_vx": "L_ground_force_vx",
        "force_l_vy": "L_ground_force_vy",
        "force_l_vz": "L_ground_force_vz",
        "force_l_px": "L_ground_force_px",
        "force_l_py": "L_ground_force_py",
        "force_l_pz": "L_ground_force_pz",
        "cop_l_x": "L_ground_force_px",
        "cop_l_y": "L_ground_force_py",
        "cop_l_z": "L_ground_force_pz",
        "torque_l_x": "L_ground_torque_x",
        "torque_l_y": "L_ground_torque_y",
        "torque_l_z": "L_ground_torque_z",
    }

    header_lines = []
    with open(mot_path, "r") as f:
        for line in f:
            header_lines.append(line)
            if line.strip().lower() == "endheader":
                break
        df = pd.read_csv(f, sep=r"\s+")

    df = df.rename(columns=rename_map)

    for i, line in enumerate(header_lines):
        if line.strip().lower().startswith("ncolumns="):
            header_lines[i] = f"nColumns={df.shape[1]}\n"

    with open(mot_path, "w") as f_out:
        f_out.writelines(header_lines)
        df.to_csv(f_out, sep="\t", index=False, float_format="%.8f")


def _read_mot(path):
    with open(path, "r") as f:
        lines = f.readlines()

    header_lines = []
    it = iter(lines)
    for line in it:
        header_lines.append(line)
        if line.strip().lower() == "endheader":
            break

    for line in it:
        if line.strip():
            col_header = line
            break
    col_names = re.split(r"\s+|\t+", col_header.strip())

    df = pd.read_csv(
        StringIO("".join(it)),
        sep=r"\s+|\t+",
        names=col_names,
        engine="python",
    )
    df = df.dropna(how="all")
    return header_lines, col_names, df


def clean_mot_in_place(mot_path, decimals=8):
    mot_path = os.path.abspath(mot_path)
    header_lines, col_names, df = _read_mot(mot_path)

    if len(col_names) == 0:
        raise ValueError(f"No columns found in {mot_path}")

    time_col = col_names[0]
    df = df.dropna(how="all").reset_index(drop=True)

    t = df[time_col].values
    keep_mask = np.ones(len(df), dtype=bool)
    keep_mask[1:] = t[1:] > t[:-1]
    df = df.loc[keep_mask].reset_index(drop=True)

    all_nan_cols = df.columns[(df.isna().all())]
    for col in all_nan_cols:
        if col == time_col:
            continue
        df[col] = 0.0

    partial_nan_cols = df.columns[df.isna().any() & ~df.isna().all()]
    if len(partial_nan_cols) > 0:
        df_interp = df.set_index(time_col)
        df_interp[partial_nan_cols] = df_interp[partial_nan_cols].interpolate(axis=0, limit_direction="both")
        df = df_interp.reset_index()

    df = df.fillna(0.0)

    n_rows, n_cols = df.shape
    new_header = []
    for line in header_lines:
        low = line.strip().lower()
        if low.startswith("nrows"):
            new_header.append(f"nRows={n_rows}\n")
        elif low.startswith("ncolumns"):
            new_header.append(f"nColumns={n_cols}\n")
        else:
            new_header.append(line)

    float_fmt = f"%.{decimals}f"
    with open(mot_path, "w") as f:
        f.writelines(new_header)
        f.write("\t".join(col_names) + "\n")
        df.to_csv(f, sep="\t", index=False, header=False, float_format=float_fmt)


def reformat_ik_mot(orig_path, out_path, decimals=8):
    header_lines = []
    with open(orig_path, "r") as f:
        for line in f:
            header_lines.append(line)
            if line.strip().lower() == "endheader":
                break

        col_line = f.readline()
        while col_line.strip() == "":
            col_line = f.readline()

        col_names = re.split(r"\s+|\t+", col_line.strip())

        df = pd.read_csv(
            f,
            sep=r"\s+|\t+",
            names=col_names,
            engine="python",
            comment="#",
        )

    df = df.dropna(how="all")

    n_rows = len(df)
    n_cols = len(df.columns)

    new_header = []
    for line in header_lines:
        low = line.strip().lower()
        if low.startswith("nrows"):
            new_header.append(f"nRows={n_rows}\n")
        elif low.startswith("ncolumns"):
            new_header.append(f"nColumns={n_cols}\n")
        else:
            new_header.append(line)

    float_fmt_time = f"{{:11.{decimals}f}}"
    float_fmt_other = f"{{:14.{decimals}f}}"

    with open(out_path, "w") as f:
        for line in new_header:
            f.write(line)

        f.write("\n" + "\t".join(df.columns) + "\n")

        for _, row in df.iterrows():
            vals = row.to_numpy()
            time_val = vals[0]
            other_vals = vals[1:]

            f.write(float_fmt_time.format(time_val))
            for v in other_vals:
                f.write("\t" + float_fmt_other.format(v))
            f.write("\n")

    clean_mot_in_place(out_path)


def close_all_loggers():
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    loggers.append(logging.getLogger())

    for log in loggers:
        for handler in getattr(log, "handlers", []):
            handler.close()
            log.removeHandler(handler)
