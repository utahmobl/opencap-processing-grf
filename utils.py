'''
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
'''

import os
import requests
import urllib.request
import shutil
import numpy as np
import pandas as pd
import yaml
import pickle
import re
import glob
import zipfile
import platform
import opensim
import subprocess
import logging
from pathlib import Path
from scipy.signal import butter, sosfiltfilt
from utilsAPI import get_api_url
from utilsAuthentication import get_token
import matplotlib.pyplot as plt
from scipy.signal import gaussian
from io import StringIO


API_URL = get_api_url()
API_TOKEN = get_token()

def download_file(url, file_name):
    with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

def get_session_json(session_id):
    resp = requests.get(
        API_URL + "sessions/{}/".format(session_id),
        headers = {"Authorization": "Token {}".format(API_TOKEN)})
    
    if resp.status_code == 500:
        raise Exception('No server response. Likely not a valid session id.')
        
    sessionJson = resp.json()
    if 'trials' not in sessionJson.keys():
        raise Exception('This session is not in your username, nor is it public. You do not have access.')
    
    # Sort trials by time recorded.
    def get_created_at(trial):
        return trial['created_at']
    sessionJson['trials'].sort(key=get_created_at)
    
    return sessionJson
    
# Returns a list of all sessions of the user.
def get_user_sessions():
    sessions = requests.get(
        API_URL + "sessions/valid/", 
        headers = {"Authorization": "Token {}".format(API_TOKEN)}).json()
    
    return sessions

# Returns a list of all sessions of the user.
# TODO: this also contains public sessions of other users.
def get_user_sessions_all(user_token=API_TOKEN):
    sessions = requests.get(
        API_URL + "sessions/", 
        headers = {"Authorization": "Token {}".format(user_token)}).json()
    
    return sessions

# Returns a list of all subjects of the user.
def get_user_subjects(user_token=API_TOKEN):
    subjects = requests.get(
            API_URL + "subjects/", 
            headers = {"Authorization": "Token {}".format(user_token)}).json()
    
    return subjects

# Returns a list of all sessions of a subject.
def get_subject_sessions(subject_id, user_token=API_TOKEN):
    sessions = requests.get(
        API_URL + "subjects/{}/".format(subject_id),
        headers = {"Authorization": "Token {}".format(user_token)}).json()['sessions']
    
    return sessions

def get_trial_json(trial_id):
    trialJson = requests.get(
        API_URL + "trials/{}/".format(trial_id),
        headers = {"Authorization": "Token {}".format(API_TOKEN)}).json()
    
    return trialJson

def get_neutral_trial_id(session_id):
    session = get_session_json(session_id)    
    neutral_ids = [t['id'] for t in session['trials'] if t['name']=='neutral']
    
    if len(neutral_ids)>0:
        neutralID = neutral_ids[-1]
    elif session['meta']['neutral_trial']:
        neutralID = session['meta']['neutral_trial']['id']
    else:
        raise Exception('No neutral trial in session.')
    
    return neutralID 
 

def get_calibration_trial_id(session_id):
    session = get_session_json(session_id)
    
    calib_ids = [t['id'] for t in session['trials'] if t['name'] == 'calibration']
                                                          
    if len(calib_ids)>0:
        calibID = calib_ids[-1]
    elif session['meta']['sessionWithCalibration']:
        calibID = get_calibration_trial_id(session['meta']['sessionWithCalibration']['id'])
    else:
        raise Exception('No calibration trial in session.')
    
    return calibID

def get_camera_mapping(session_id, session_path):
    calibration_id = get_calibration_trial_id(session_id)
    trial = get_trial_json(calibration_id)
    resultTags = [res['tag'] for res in trial['results']]

    mappingPath = os.path.join(session_path,'Videos','mappingCamDevice.pickle')
    os.makedirs(os.path.join(session_path,'Videos'), exist_ok=True)
    if not os.path.exists(mappingPath):
        mappingURL = trial['results'][resultTags.index('camera_mapping')]['media']
        download_file(mappingURL, mappingPath)
    

def get_model_and_metadata(session_id, session_path):
    neutral_id = get_neutral_trial_id(session_id)
    trial = get_trial_json(neutral_id)
    resultTags = [res['tag'] for res in trial['results']]
    
    # Metadata.
    metadataPath = os.path.join(session_path,'sessionMetadata.yaml')
    if not os.path.exists(metadataPath) :
        metadataURL = trial['results'][resultTags.index('session_metadata')]['media']
        download_file(metadataURL, metadataPath)
    
    # Model.
    modelURL = trial['results'][resultTags.index('opensim_model')]['media']
    modelName = modelURL[modelURL.rfind('-')+1:modelURL.rfind('?')]
    modelFolder = os.path.join(session_path, 'OpenSimData', 'Model')
    modelPath = os.path.join(modelFolder, modelName)
    if not os.path.exists(modelPath):
        os.makedirs(modelFolder, exist_ok=True)
        download_file(modelURL, modelPath)
        
    return modelName

def get_main_settings(session_folder,trial_name):
    settings_path = os.path.join(session_folder,'MarkerData',
                                 'Settings','settings_' + trial_name + '.yaml')
    main_settings = import_metadata(settings_path)
    
    return main_settings

        
def get_model_name_from_metadata(sessionFolder,appendText='_scaled'):
    metadataPath = os.path.join(sessionFolder,'sessionMetadata.yaml')
    
    if os.path.exists(metadataPath):
        metadata = import_metadata(os.path.join(sessionFolder,'sessionMetadata.yaml'))
        modelName = metadata['openSimModel'] + appendText + '.osim'
    else:
        raise Exception('Session metadata not found, could not identify OpenSim model.')
        
    return modelName

        
def get_motion_data(trial_id, session_path):
    trial = get_trial_json(trial_id)
    trial_name = trial['name']
    resultTags = [res['tag'] for res in trial['results']]

    # Marker data.
    if 'marker_data' in resultTags:
        markerFolder = os.path.join(session_path, 'MarkerData')
        markerPath = os.path.join(markerFolder, trial_name + '.trc')
        os.makedirs(markerFolder, exist_ok=True)
        if not os.path.exists(markerPath):
            markerURL = trial['results'][resultTags.index('marker_data')]['media']
            download_file(markerURL, markerPath)

    
    # IK data.
    if 'ik_results' in resultTags:
        ikFolder = os.path.join(session_path, 'OpenSimData', 'Kinematics')
        ikPath = os.path.join(ikFolder, trial_name + '.mot')
        os.makedirs(ikFolder, exist_ok=True)
        if not os.path.exists(ikPath):
            ikURL = trial['results'][resultTags.index('ik_results')]['media']
            download_file(ikURL, ikPath)
        
    # Main settings
    if 'main_settings' in resultTags:
        settingsFolder = os.path.join(session_path, 'MarkerData', 'Settings')
        settingsPath = os.path.join(settingsFolder, 'settings_' + trial_name + '.yaml')
        os.makedirs(settingsFolder, exist_ok=True)
        if not os.path.exists(settingsPath):
            settingsURL = trial['results'][resultTags.index('main_settings')]['media']
            download_file(settingsURL, settingsPath)
        
        
def get_geometries(session_path, modelName='LaiUhlrich2022_scaled'):
        
    geometryFolder = os.path.join(session_path, 'OpenSimData', 'Model', 'Geometry')
    try:
        # Download.
        os.makedirs(geometryFolder, exist_ok=True)
        if 'Lai' in modelName:
            modelType = 'LaiArnold'
            vtpNames = [
                'capitate_lvs','capitate_rvs','hamate_lvs','hamate_rvs',
                'hat_jaw','hat_ribs_scap','hat_skull','hat_spine','humerus_lv',
                'humerus_rv','index_distal_lvs','index_distal_rvs',
                'index_medial_lvs', 'index_medial_rvs','index_proximal_lvs',
                'index_proximal_rvs','little_distal_lvs','little_distal_rvs',
                'little_medial_lvs','little_medial_rvs','little_proximal_lvs',
                'little_proximal_rvs','lunate_lvs','lunate_rvs','l_bofoot',
                'l_femur','l_fibula','l_foot','l_patella','l_pelvis','l_talus',
                'l_tibia','metacarpal1_lvs','metacarpal1_rvs',
                'metacarpal2_lvs','metacarpal2_rvs','metacarpal3_lvs',
                'metacarpal3_rvs','metacarpal4_lvs','metacarpal4_rvs',
                'metacarpal5_lvs','metacarpal5_rvs','middle_distal_lvs',
                'middle_distal_rvs','middle_medial_lvs','middle_medial_rvs',
                'middle_proximal_lvs','middle_proximal_rvs','pisiform_lvs',
                'pisiform_rvs','radius_lv','radius_rv','ring_distal_lvs',
                'ring_distal_rvs','ring_medial_lvs','ring_medial_rvs',
                'ring_proximal_lvs','ring_proximal_rvs','r_bofoot','r_femur',
                'r_fibula','r_foot','r_patella','r_pelvis','r_talus','r_tibia',
                'sacrum','scaphoid_lvs','scaphoid_rvs','thumb_distal_lvs',
                'thumb_distal_rvs','thumb_proximal_lvs','thumb_proximal_rvs',
                'trapezium_lvs','trapezium_rvs','trapezoid_lvs','trapezoid_rvs',
                'triquetrum_lvs','triquetrum_rvs','ulna_lv','ulna_rv']
        else:
            raise ValueError("Geometries not available for this model")                
        for vtpName in vtpNames:
            url = 'https://mc-opencap-public.s3.us-west-2.amazonaws.com/geometries_vtp/{}/{}.vtp'.format(modelType, vtpName)
            filename = os.path.join(geometryFolder, '{}.vtp'.format(vtpName))                
            download_file(url, filename)
    except:
        pass
    
def import_metadata(filePath):
    myYamlFile = open(filePath)
    parsedYamlFile = yaml.load(myYamlFile, Loader=yaml.FullLoader)
    
    return parsedYamlFile
    
def download_kinematics(session_id, folder=None, trialNames=None):
    
    # Login to access opencap data from server. 
    
    # Create folder.
    if folder is None:
        folder = os.getcwd()    
    os.makedirs(folder, exist_ok=True)
    
    # Model and metadata.
    neutral_id = get_neutral_trial_id(session_id)
    get_motion_data(neutral_id, folder)
    modelName = get_model_and_metadata(session_id, folder)
    # Remove extension from modelName
    modelName = modelName.replace('.osim','')
    
    # Session trial names.
    sessionJson = get_session_json(session_id)
    sessionTrialNames = [t['name'] for t in sessionJson['trials']]
    if trialNames != None:
        [print(t + ' not in session trial names.') 
         for t in trialNames if t not in sessionTrialNames]
    
    # Motion data.
    loadedTrialNames = []
    for trialDict in sessionJson['trials']:
        if trialNames is not None and trialDict['name'] not in trialNames:
            continue        
        trial_id = trialDict['id']
        get_motion_data(trial_id,folder)
        loadedTrialNames.append(trialDict['name'])
        
    # Remove 'calibration' and 'neutral' from loadedTrialNames.    
    #loadedTrialNames = [i for i in loadedTrialNames if i!='neutral' and i!='calibration']
        
    # Geometries.
    get_geometries(folder, modelName=modelName)
        
    return loadedTrialNames, modelName

# Download pertinent trial data.
def download_trial(trial_id,neutral_id, folder, session_id=None):
    
    trial = get_trial_json(trial_id)
    if session_id is None:
        session_id = trial['session_id']
        
    os.makedirs(folder,exist_ok=True)
    
    # download model
    get_model_and_metadata(session_id, folder)
    
    # download trc and mot
    get_motion_data(trial_id,folder)
    get_motion_data(neutral_id,folder) # EYM Edit for scaling for ML
    
    return trial['name']


# Get trial ID from name.
def get_trial_id(session_id,trial_name):
    session = get_session_json(session_id)
    
    trial_id = [t['id'] for t in session['trials'] if t['name'] == trial_name]
    
    return trial_id[0]

# %%  Storage file to numpy array.
def storage_to_numpy(storage_file, excess_header_entries=0):
    """Returns the data from a storage file in a numpy format. Skips all lines
    up to and including the line that says 'endheader'.
    Parameters
    ----------
    storage_file : str
        Path to an OpenSim Storage (.sto) file.
    Returns
    -------
    data : np.ndarray (or numpy structure array or something?)
        Contains all columns from the storage file, indexable by column name.
    excess_header_entries : int, optional
        If the header row has more names in it than there are data columns.
        We'll ignore this many header row entries from the end of the header
        row. This argument allows for a hacky fix to an issue that arises from
        Static Optimization '.sto' outputs.
    Examples
    --------
    Columns from the storage file can be obtained as follows:
        >>> data = storage2numpy('<filename>')
        >>> data['ground_force_vy']
    """
    # What's the line number of the line containing 'endheader'?
    f = open(storage_file, 'r')

    header_line = False
    for i, line in enumerate(f):
        if header_line:
            column_names = line.split()
            break
        if line.count('endheader') != 0:
            line_number_of_line_containing_endheader = i + 1
            header_line = True
    f.close()

    # With this information, go get the data.
    if excess_header_entries == 0:
        names = True
        skip_header = line_number_of_line_containing_endheader
    else:
        names = column_names[:-excess_header_entries]
        skip_header = line_number_of_line_containing_endheader + 1
    data = np.genfromtxt(storage_file, names=names,
            skip_header=skip_header)

    return data

# %%  Storage file to dataframe.
def storage_to_dataframe(storage_file, headers):
    # Extract data
    data = storage_to_numpy(storage_file)
    out = pd.DataFrame(data=data['time'], columns=['time'])    
    for count, header in enumerate(headers):
        out.insert(count + 1, header, data[header])    
    
    return out

# %% Load storage and output as dataframe or numpy
def load_storage(file_path,outputFormat='numpy'):
    table = opensim.TimeSeriesTable(file_path)    
    data = table.getMatrix().to_numpy()
    time = np.asarray(table.getIndependentColumn()).reshape(-1, 1)
    data = np.hstack((time,data))
    headers = ['time'] + list(table.getColumnLabels())
    
    if outputFormat == 'numpy':
        return data,headers
    elif outputFormat == 'dataframe':
        return pd.DataFrame(data, columns=headers)
    else:
        return None    
    
# %%  Numpy array to storage file.
def numpy_to_storage(labels, data, storage_file, datatype=None):
    
    assert data.shape[1] == len(labels), "# labels doesn't match columns"
    assert labels[0] == "time"
    
    f = open(storage_file, 'w')
    # Old style
    if datatype is None:
        f = open(storage_file, 'w')
        f.write('name %s\n' %storage_file)
        f.write('datacolumns %d\n' %data.shape[1])
        f.write('datarows %d\n' %data.shape[0])
        f.write('range %f %f\n' %(np.min(data[:, 0]), np.max(data[:, 0])))
        f.write('endheader \n')
    # New style
    else:
        if datatype == 'IK':
            f.write('Coordinates\n')
        elif datatype == 'ID':
            f.write('Inverse Dynamics Generalized Forces\n')
        elif datatype == 'GRF':
            f.write('%s\n' %storage_file)
        elif datatype == 'muscle_forces':
            f.write('ModelForces\n')
        f.write('version=1\n')
        f.write('nRows=%d\n' %data.shape[0])
        f.write('nColumns=%d\n' %data.shape[1])    
        if datatype == 'IK':
            f.write('inDegrees=yes\n\n')
            f.write('Units are S.I. units (second, meters, Newtons, ...)\n')
            f.write("If the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no).\n\n")
        elif datatype == 'ID':
            f.write('inDegrees=no\n')
        elif datatype == 'GRF':
            f.write('inDegrees=yes\n')
        elif datatype == 'muscle_forces':
            f.write('inDegrees=yes\n\n')
            f.write('This file contains the forces exerted on a model during a simulation.\n\n')
            f.write("A force is a generalized force, meaning that it can be either a force (N) or a torque (Nm).\n\n")
            f.write('Units are S.I. units (second, meters, Newtons, ...)\n')
            f.write('Angles are in degrees.\n\n')
            
        f.write('endheader \n')
    
    for i in range(len(labels)):
        f.write('%s\t' %labels[i])
    f.write('\n')
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            f.write('%20.8f\t' %data[i, j])
        f.write('\n')
        
    f.close()

def download_videos_from_server(session_id,trial_id,
                             isCalibration=False, isStaticPose=False,
                             trial_name= None, session_path = None):
    
    if session_path is None:
        data_dir = os.getcwd() 
        session_path = os.path.join(data_dir,'Data', session_id)  
    if not os.path.exists(session_path): 
        os.makedirs(session_path, exist_ok=True)
    
    resp = requests.get("{}trials/{}/".format(API_URL,trial_id),
                         headers = {"Authorization": "Token {}".format(API_TOKEN)})
    trial = resp.json()
    if trial_name is None:
        trial_name = trial['name']
    trial_name = trial_name.replace(' ', '')

    print("\nDownloading {}".format(trial_name))

    # The videos are not always organized in the same order. Here, we save
    # the order during the first trial processed in the session such that we
    # can use the same order for the other trials.
    if not os.path.exists(os.path.join(session_path, "Videos", 'mappingCamDevice.pickle')):
        mappingCamDevice = {}
        for k, video in enumerate(trial["videos"]):
            os.makedirs(os.path.join(session_path, "Videos", "Cam{}".format(k), "InputMedia", trial_name), exist_ok=True)
            video_path = os.path.join(session_path, "Videos", "Cam{}".format(k), "InputMedia", trial_name, trial_name + ".mov")
            download_file(video["video"], video_path)                
            mappingCamDevice[video["device_id"].replace('-', '').upper()] = k
        with open(os.path.join(session_path, "Videos", 'mappingCamDevice.pickle'), 'wb') as handle:
            pickle.dump(mappingCamDevice, handle)
    else:
        with open(os.path.join(session_path, "Videos", 'mappingCamDevice.pickle'), 'rb') as handle:
            mappingCamDevice = pickle.load(handle) 
            # ensure upper on deviceID
            for dID in mappingCamDevice.keys():
                mappingCamDevice[dID.upper()] = mappingCamDevice.pop(dID)
        for video in trial["videos"]:            
            k = mappingCamDevice[video["device_id"].replace('-', '').upper()] 
            videoDir = os.path.join(session_path, "Videos", "Cam{}".format(k), "InputMedia", trial_name)
            os.makedirs(videoDir, exist_ok=True)
            video_path = os.path.join(videoDir, trial_name + ".mov")
            if not os.path.exists(video_path):
                if video['video'] :
                    download_file(video["video"], video_path)
              
    return trial_name
   
    
def get_calibration(session_id,session_path):
    calibration_id = get_calibration_trial_id(session_id)

    resp = requests.get("{}trials/{}/".format(API_URL,calibration_id),
                         headers = {"Authorization": "Token {}".format(API_TOKEN)})
    trial = resp.json()
    calibResultTags = [res['tag'] for res in trial['results']]
   
    videoFolder = os.path.join(session_path,'Videos')
    os.makedirs(videoFolder, exist_ok=True)
    
    if trial['status'] != 'done':
        return
    
    mapURL = trial['results'][calibResultTags.index('camera_mapping')]['media']
    mapLocalPath = os.path.join(videoFolder,'mappingCamDevice.pickle')

    download_and_switch_calibration(session_id,session_path,calibTrialID=calibration_id)
    
    # Download mapping
    if len(glob.glob(mapLocalPath)) == 0:
        download_file(mapURL,mapLocalPath)
                        

def download_and_switch_calibration(session_id,session_path,calibTrialID = None):
    if calibTrialID == None:
        calibTrialID = get_calibration_trial_id(session_id)
    resp = requests.get("https://api.opencap.ai/trials/{}/".format(calibTrialID),
                         headers = {"Authorization": "Token {}".format(API_TOKEN)})
    trial = resp.json()
       
    calibURLs = {t['device_id']:t['media'] for t in trial['results'] if t['tag'] == 'calibration_parameters_options'}
    calibImgURLs = {t['device_id']:t['media'] for t in trial['results'] if t['tag'] == 'calibration-img'}
    _,imgExtension = os.path.splitext(calibImgURLs[list(calibImgURLs.keys())[0]])
    lastIdx = imgExtension.find('?') 
    if lastIdx >0:
        imgExtension = imgExtension[:lastIdx]
    
    if 'meta' in trial.keys() and trial['meta'] is not None and 'calibration' in trial['meta'].keys():
        calibDict = trial['meta']['calibration']
        calibImgFolder = os.path.join(session_path,'CalibrationImages')
        os.makedirs(calibImgFolder,exist_ok=True)
        for cam,calibNum in calibDict.items():
            camDir = os.path.join(session_path,'Videos',cam)
            os.makedirs(camDir,exist_ok=True)
            file_name = os.path.join(camDir,'cameraIntrinsicsExtrinsics.pickle')
            img_fileName = os.path.join(calibImgFolder,'calib_img' + cam + imgExtension)
            if calibNum == 0:
                download_file(calibURLs[cam+'_soln0'], file_name)
                download_file(calibImgURLs[cam],img_fileName)
            elif calibNum == 1:
                download_file(calibURLs[cam+'_soln1'], file_name) 
                download_file(calibImgURLs[cam + '_altSoln'],img_fileName)
                
            
def post_file_to_trial(filePath,trial_id,tag,device_id):
    files = {'media': open(filePath, 'rb')}
    data = {
        "trial": trial_id,
        "tag": tag,
        "device_id" : device_id
    }

    requests.post("{}results/".format(API_URL), files=files, data=data,
                         headers = {"Authorization": "Token {}".format(API_TOKEN)})
    files["media"].close()

def post_video_to_trial(filePath,trial_id,device_id,parameters):
    files = {'video': open(filePath, 'rb')}
    data = {
        "trial": trial_id,
        "device_id" : device_id,
        "parameters": parameters
    }

    requests.post("{}videos/".format(API_URL), files=files, data=data,
                         headers = {"Authorization": "Token {}".format(API_TOKEN)})
    files["video"].close()

def delete_video_from_trial(video_id):

    requests.delete("{}videos/{}/".format(API_URL, video_id),
                        headers = {"Authorization": "Token {}".format(API_TOKEN)})
    
def delete_results(trial_id, tag=None, resultNum=None):
    # Delete specific result number, or all results with a specific tag, or all results if tag==None
    if resultNum != None:
        resultNums = [resultNum]
    elif tag != None:
        trial = get_trial_json(trial_id)
        resultNums = [r['id'] for r in trial['results'] if r['tag']==tag]
        
    elif tag == None: 
        trial = get_trial_json(trial_id)
        resultNums = [r['id'] for r in trial['results']]

    for rNum in resultNums:
        requests.delete(API_URL + "results/{}/".format(rNum),
                        headers = {"Authorization": "Token {}".format(API_TOKEN)})
        
def set_trial_status(trial_id, status):

    # Available statuses: 'done', 'error', 'stopped', 'reprocess'
    # 'processing' and 'recording also exist, but it does not make sense to set them manually.
    # Throw error if status is not one of the above.
    if status not in ['done', 'error', 'stopped', 'reprocess']:
        raise ValueError('Invalid status. Available statuses: done, error, stopped, reprocess')

    requests.patch(API_URL+"trials/{}/".format(trial_id), data={'status': status},
                     headers = {"Authorization": "Token {}".format(API_TOKEN)})
    
def set_session_subject(session_id, subject_id):
    requests.patch(API_URL+"sessions/{}/".format(session_id), data={'subject': subject_id},
                     headers = {"Authorization": "Token {}".format(API_TOKEN)})  

def get_syncd_videos(trial_id,session_path):
    trial = requests.get("{}trials/{}/".format(API_URL,trial_id),
                         headers = {"Authorization": "Token {}".format(API_TOKEN)}).json()
    trial_name = trial['name']
    
    if trial['results']:
        for result in trial['results']:
            if result['tag'] == 'video-sync':
                url = result['media']
                cam,suff = os.path.splitext(url[url.rfind('_')+1:])
                lastIdx = suff.find('?') 
                if lastIdx >0:
                    suff = suff[:lastIdx]
                
                syncVideoPath = os.path.join(session_path,'Videos',cam,'InputMedia',trial_name,trial_name + '_sync' + suff)
                download_file(url,syncVideoPath)
        
        
def download_session(session_id, sessionBasePath=None,
                     zipFolder=False, writeToDB=False,
                     downloadVideos=True, trial_prefix=None):
    print('\nDownloading {}'.format(session_id))
    
    if sessionBasePath is None:
        sessionBasePath = os.path.join(os.getcwd(), 'Data')
    
    session = get_session_json(session_id)
    session_path = os.path.join(sessionBasePath, 'OpenCapData_' + session_id)
    
    calib_id = get_calibration_trial_id(session_id)
    neutral_id = get_neutral_trial_id(session_id)

    # ------------------------------------------
    # Filter dynamic IDs using trial_prefix
    # ------------------------------------------
    dynamic_ids = []
    for t in session['trials']:
        name = t['name']
        if name not in ('calibration', 'neutral'):
            if trial_prefix is None or trial_prefix in name:
                dynamic_ids.append(t['id'])

    # Calibration
    try:
        get_camera_mapping(session_id, session_path)
        if downloadVideos:
            download_videos_from_server(
                session_id, calib_id,
                isCalibration=True, isStaticPose=False,
                session_path=session_path
            )
        get_calibration(session_id, session_path)
    except:
        pass
    
    # Neutral
    try:
        modelName = get_model_and_metadata(session_id, session_path)
        get_motion_data(neutral_id, session_path)
        if downloadVideos:
            download_videos_from_server(
                session_id, neutral_id,
                isCalibration=False, isStaticPose=True,
                session_path=session_path
            )
        get_syncd_videos(neutral_id, session_path)
    except:
        pass

    # Dynamic
    for dynamic_id in dynamic_ids:
        try:
            get_motion_data(dynamic_id, session_path)
            if downloadVideos:
                download_videos_from_server(
                    session_id, dynamic_id,
                    isCalibration=False, isStaticPose=False,
                    session_path=session_path
                )
            get_syncd_videos(dynamic_id, session_path)
        except:
            pass

    repoDir = os.path.dirname(os.path.abspath(__file__))
    
    # Readme  
    try:        
        pathReadme = os.path.join(repoDir, 'Resources', 'README.txt')
        pathReadmeEnd = os.path.join(session_path, 'README.txt')
        shutil.copy2(pathReadme, pathReadmeEnd)
    except:
        pass
        
    # Geometry
    try:
        if 'Lai' in modelName:
            modelType = 'LaiArnold'
        else:
            raise ValueError("Geometries not available for this model, please contact us")
        if platform.system() == 'Windows':
            geometryDir = os.path.join(repoDir, 'tmp', modelType, 'Geometry')
        else:
            geometryDir = "/tmp/{}/Geometry".format(modelType)

        if not os.path.exists(geometryDir):
            os.makedirs(geometryDir, exist_ok=True)
            get_geometries(session_path, modelName=modelName)

        geometryDirEnd = os.path.join(session_path, 'OpenSimData', 'Model', 'Geometry')
        shutil.copytree(geometryDir, geometryDirEnd)
    except:
        pass
    
    # Zip helper
    def zipdir(path, ziph):
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(
                    os.path.join(root, file),
                    os.path.relpath(os.path.join(root, file),
                    os.path.join(path, '..'))
                )

    session_zip = '{}.zip'.format(session_path)
    if os.path.isfile(session_zip):
        os.remove(session_zip)

    if zipFolder:
        zipf = zipfile.ZipFile(session_zip, 'w', zipfile.ZIP_DEFLATED)
        zipdir(session_path, zipf)
        zipf.close()
    
    if writeToDB and len(dynamic_ids) > 0:
        post_file_to_trial(
            session_zip, dynamic_ids[-1],
            tag='session_zip', device_id='all'
        )

    
def cross_corr(y1, y2,multCorrGaussianStd=None,visualize=False):
    """Calculates the cross correlation and lags without normalization.
    
    The definition of the discrete cross-correlation is in:
    https://www.mathworks.com/help/matlab/ref/xcorr.html
    
    Args:
    y1, y2: Should have the same length.
    
    Returns:
    max_corr: Maximum correlation without normalization.
    lag: The lag in terms of the index.
    """
    # Pad shorter signal with 0s
    if len(y1) > len(y2):
        temp = np.zeros(len(y1))
        temp[0:len(y2)] = y2
        y2 = np.copy(temp)
    elif len(y2)>len(y1):
        temp = np.zeros(len(y2))
        temp[0:len(y1)] = y1
        y1 = np.copy(temp)
        
    y1_auto_corr = np.dot(y1, y1) / len(y1)
    y2_auto_corr = np.dot(y2, y2) / len(y1)
    corr = np.correlate(y1, y2, mode='same')
    # The unbiased sample size is N - lag.
    unbiased_sample_size = np.correlate(np.ones(len(y1)), np.ones(len(y1)), mode='same')
    corr = corr / unbiased_sample_size / np.sqrt(y1_auto_corr * y2_auto_corr)
    shift = len(y1) // 2
    max_corr = np.max(corr)
    argmax_corr = np.argmax(corr)    
        
    if visualize:
        plt.figure()
        plt.plot(corr)
        plt.title('vertical velocity correlation')
        
    # Multiply correlation curve by gaussian (prioritizing lag solution closest to 0)
    if multCorrGaussianStd is not None:
        corr = np.multiply(corr,gaussian(len(corr),multCorrGaussianStd))
        if visualize: 
            plt.plot(corr,color=[.4,.4,.4])
            plt.legend(['corr','corr*gaussian'])  
    
    argmax_corr = np.argmax(corr)
    max_corr = np.nanmax(corr)
    
    lag = argmax_corr-shift
    
    return max_corr, lag

def downsample(data,time,framerate_in,framerate_out):
    # Calculate the downsampling factor
    downsampling_factor = framerate_in / framerate_out
    
    # Create new indices for downsampling
    original_indices = np.arange(len(data))
    new_indices = np.arange(0, len(data), downsampling_factor)
    
    # Perform downsampling with interpolation
    downsampled_data = np.ndarray((len(new_indices), data.shape[1]))
    for i in range(data.shape[1]):
        downsampled_data[:,i] = np.interp(new_indices, original_indices, data[:,i])
    
    downsampled_time = np.interp(new_indices, original_indices, time)
    
    return downsampled_time, downsampled_data


def find_opensim_cmd():
    """
    Automatically find the opensim-cmd executable.
    
    Returns
    -------
    str or None
        Path to opensim-cmd executable, or None if not found.
    """

    # 1) First try whatever is on PATH
    cmd = shutil.which("opensim-cmd") or shutil.which("opensim-cmd.exe")
    if cmd:
        return cmd

    # 2) Fall back to the OpenSim Python package location
    try:
        pkg_dir = os.path.dirname(os.path.abspath(opensim.__file__))
        sdk_dir = os.path.dirname(pkg_dir)
        opensim_root = os.path.dirname(sdk_dir)
        candidate = os.path.join(opensim_root, "bin", "opensim-cmd.exe")
        if os.path.exists(candidate):
            return candidate
    except (AttributeError, TypeError):
        pass

    # 3) Try common installation directories
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
            f"Multiple calibration files found, expected only one. Matches:\n" +
            "\n".join(matches)
        )

    return matches[0]

def create_LaiUhlrich_model(
    generic_model_path: str,
    generic_scale_setup_xml: str,
    session_metadata_path: str,
    static_trc_path: str,
    output_dir: str,
    opensim_install_dir: str = None
):
    """
    Scale a LaiUhlrich2022 OpenSim model using static trial data.
    
    Parameters
    ----------
    generic_model_path : str
        Path to unscaled generic model (.osim), e.g., LaiUhlrich2022_Generic.osim
    generic_scale_setup_xml : str
        Path to generic scaling setup XML, e.g., Setup_LaiUhlrich_Scaling_generic.xml
    session_metadata_path : str
        Path to sessionMetadata.yaml with height_m and mass_kg
    static_trc_path : str
        Path to static TRC file
    output_dir : str
        Directory to save scaled model and processed files
    opensim_install_dir : str, optional
        Path to OpenSim installation directory. If None, auto detection is used.
        
    Returns
    -------
    scaled_model_path : str
        Path to the scaled model file
    height_m : float
        Subject height in meters
    mass_kg : float
        Subject mass in kg
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    trc_dir = os.path.join(os.path.dirname(session_metadata_path), 'ForGaitDynamics', 'TRC_Files')
    os.makedirs(trc_dir, exist_ok=True)

    # =========================================================================
    # 1. Find OpenSim command line tool
    # =========================================================================
    opensim_cmd_path = None

    if opensim_install_dir:
        candidate_path = os.path.join(opensim_install_dir, 'bin', 'opensim-cmd.exe')
        if os.path.exists(candidate_path):
            opensim_cmd_path = candidate_path

    if not opensim_cmd_path:
        opensim_cmd_path = find_opensim_cmd()

    if not opensim_cmd_path:
        raise FileNotFoundError(
            "Could not find opensim-cmd.exe. Please ensure OpenSim is installed and "
            "either provide opensim_install_dir or add OpenSim to your PATH."
        )

    # =========================================================================
    # 2. Read height and mass from session metadata
    # =========================================================================
    height_m = None
    mass_kg = None

    with open(session_metadata_path, 'r') as f:
        for line in f:
            line = line.strip()
            if 'height_m:' in line:
                try:
                    height_m = float(line.split('height_m:')[-1].strip())
                except ValueError:
                    pass
            if 'mass_kg:' in line:
                try:
                    mass_kg = float(line.split('mass_kg:')[-1].strip())
                except ValueError:
                    pass

    if height_m is None or mass_kg is None:
        raise ValueError(f"Could not read height or mass from {session_metadata_path}")

     # =========================================================================
    # 3. Define marker mappings (new and legacy, for back compatibility)
    # =========================================================================
    marker_mapping_offset = {
        'C7_study_offsetRemoved': 'C7',
        'r_shoulder_study_offsetRemoved': 'R_Shoulder',
        'L_shoulder_study_offsetRemoved': 'L_Shoulder',
        'r.ASIS_study_offsetRemoved': 'r.ASIS',
        'L.ASIS_study_offsetRemoved': 'L.ASIS',
        'r.PSIS_study_offsetRemoved': 'r.PSIS',
        'L.PSIS_study_offsetRemoved': 'L.PSIS',
        'r_knee_study_offsetRemoved': 'r_knee',
        'L_knee_study_offsetRemoved': 'L_knee',
        'r_mknee_study_offsetRemoved': 'r_mknee',
        'L_mknee_study_offsetRemoved': 'L_mknee',
        'r_ankle_study_offsetRemoved': 'r_ankle',
        'L_ankle_study_offsetRemoved': 'L_ankle',
        'r_mankle_study_offsetRemoved': 'r_mankle',
        'L_mankle_study_offsetRemoved': 'L_mankle',
        'r_calc_study_offsetRemoved': 'r_calc',
        'L_calc_study_offsetRemoved': 'L_calc',
        'r_toe_study_offsetRemoved': 'r_toe',
        'L_toe_study_offsetRemoved': 'L_toe',
        'r_5meta_study_offsetRemoved': 'r_5meta',
        'L_5meta_study_offsetRemoved': 'L_5meta',
        'r_thigh1_study_offsetRemoved': 'r_thigh1',
        'r_thigh2_study_offsetRemoved': 'r_thigh2',
        'r_thigh3_study_offsetRemoved': 'r_thigh3',
        'L_thigh1_study_offsetRemoved': 'L_thigh1',
        'L_thigh2_study_offsetRemoved': 'L_thigh2',
        'L_thigh3_study_offsetRemoved': 'L_thigh3',
        'r_sh1_study_offsetRemoved': 'r_sh1',
        'r_sh2_study_offsetRemoved': 'r_sh2',
        'r_sh3_study_offsetRemoved': 'r_sh3',
        'L_sh1_study_offsetRemoved': 'L_sh1',
        'L_sh2_study_offsetRemoved': 'L_sh2',
        'L_sh3_study_offsetRemoved': 'L_sh3',
        'RHJC_study_offsetRemoved': 'R_HJC',
        'LHJC_study_offsetRemoved': 'L_HJC',
        'r_lelbow_study_offsetRemoved': 'r_lelbow',
        'L_lelbow_study_offsetRemoved': 'L_lelbow',
        'r_melbow_study_offsetRemoved': 'r_melbow',
        'L_melbow_study_offsetRemoved': 'L_melbow',
        'r_lwrist_study_offsetRemoved': 'r_lwrist',
        'L_lwrist_study_offsetRemoved': 'L_lwrist',
        'r_mwrist_study_offsetRemoved': 'r_mwrist',
        'L_mwrist_study_offsetRemoved': 'L_mwrist'
    }

    # legacy naming without "offsetRemoved"
    marker_mapping = {
        'C7_study': 'C7',
        'r_shoulder_study': 'R_Shoulder',
        'L_shoulder_study': 'L_Shoulder',
        'r.ASIS_study': 'r.ASIS',
        'L.ASIS_study': 'L.ASIS',
        'r.PSIS_study': 'r.PSIS',
        'L.PSIS_study': 'L.PSIS',
        'r_knee_study': 'r_knee',
        'L_knee_study': 'L_knee',  # fixed typo L_knee_stud
        'r_mknee_study': 'r_mknee',
        'L_mknee_study': 'L_mknee',
        'r_ankle_study': 'r_ankle',
        'L_ankle_study': 'L_ankle',
        'r_mankle_study': 'r_mankle',
        'L_mankle_study': 'L_mankle',
        'r_calc_study': 'r_calc',
        'L_calc_study': 'L_calc',
        'r_toe_study': 'r_toe',
        'L_toe_study': 'L_toe',
        'r_5meta_study': 'r_5meta',
        'L_5meta_study': 'L_5meta',
        'r_thigh1_study': 'r_thigh1',
        'r_thigh2_study': 'r_thigh2',
        'r_thigh3_study': 'r_thigh3',
        'L_thigh1_study': 'L_thigh1',
        'L_thigh2_study': 'L_thigh2',
        'L_thigh3_study': 'L_thigh3',
        'r_sh1_study': 'r_sh1',
        'r_sh2_study': 'r_sh2',
        'r_sh3_study': 'r_sh3',
        'L_sh1_study': 'L_sh1',
        'L_sh2_study': 'L_sh2',
        'L_sh3_study': 'L_sh3',
        'RHJC_study': 'R_HJC',
        'LHJC_study': 'L_HJC',
        'r_lelbow_study': 'r_lelbow',
        'L_lelbow_study': 'L_lelbow',
        'r_melbow_study': 'r_melbow',
        'L_melbow_study': 'L_melbow',
        'r_lwrist_study': 'r_lwrist',
        'L_lwrist_study': 'L_lwrist',
        'r_mwrist_study': 'r_mwrist',
        'L_mwrist_study': 'L_mwrist'
    }

    # =========================================================================
    # 4. Load and process static TRC file and choose mapping
    # =========================================================================
    with open(static_trc_path, 'r') as f:
        lines = f.readlines()

    data_start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('Frame#') or line.startswith('Time'):
            data_start_idx = i
            break

    header_line = lines[data_start_idx].strip().split('\t')
    all_marker_names = [name for name in header_line if name][2:]

    # decide which mapping to use based on presence of "offsetRemoved"
    has_offset_removed = any("offsetremoved" in name.lower()
                             for name in all_marker_names)

    if has_offset_removed:
        marker_mapping = marker_mapping_offset
    else:
        marker_mapping = marker_mapping

    # now continue as before, but using the chosen marker_mapping
    available_markers = [name for name in marker_mapping.keys()
                         if name in all_marker_names]

    if len(available_markers) < 10:
        raise ValueError(
            f"Too few target markers found in TRC for scaling "
            f"({len(available_markers)} found)."
        )

    data_lines = lines[data_start_idx + 2:]
    data = []
    for line in data_lines:
        if not line.strip():
            continue
        parts = line.strip().split('\t')
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

    marker_indices = [all_marker_names.index(m) for m in available_markers]

    OC_mrkdata_specific = []
    for idx in marker_indices:
        c0 = 2 + idx * 3
        if c0 + 2 >= data.shape[1]:
            raise ValueError(
                f"Marker index {idx} out of range for TRC data columns."
            )
        OC_mrkdata_specific.extend([
            data[:, c0],
            data[:, c0 + 1],
            data[:, c0 + 2]
        ])

    OC_mrkdata_specific = np.array(OC_mrkdata_specific).T
    OC_time_zeroed = OC_time - np.min(OC_time)

    final_marker_names = [marker_mapping[m] for m in available_markers]

    # =========================================================================
    # 5. Create processed TRC file
    # =========================================================================
    trc_filename = 'OpenCap_static_LaiUhlrich_markers.trc'
    processed_trc_path = os.path.join(trc_dir, trc_filename)

    with open(processed_trc_path, 'w') as f:
        f.write(f'PathFileType\t4\t(X/Y/Z)\t{trc_filename}\n')
        f.write('DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')
        f.write('100.00\t100.00\t{}\t{}\tmm\t100.00\t1\t{}\n'.format(
            len(OC_time_zeroed), len(final_marker_names), len(OC_time_zeroed)))
        f.write('Frame#\tTime\t')
        f.write('\t\t'.join(final_marker_names))
        f.write('\n')
        f.write('\t\t')
        for i in range(len(final_marker_names)):
            f.write('X{}\tY{}\tZ{}'.format(i + 1, i + 1, i + 1))
            if i < len(final_marker_names) - 1:
                f.write('\t')
        f.write('\n')

        for i in range(len(OC_time_zeroed)):
            f.write('{}\t{:.6f}'.format(i + 1, OC_time_zeroed[i]))
            for j in range(OC_mrkdata_specific.shape[1] // 3):
                marker_idx = j * 3
                f.write('\t{:.6f}\t{:.6f}\t{:.6f}'.format(
                    OC_mrkdata_specific[i, marker_idx],
                    OC_mrkdata_specific[i, marker_idx + 1],
                    OC_mrkdata_specific[i, marker_idx + 2]
                ))
            f.write('\n')

    # =========================================================================
    # 6. Configure ScaleTool and save setup XML
    # =========================================================================
    if not os.path.exists(generic_scale_setup_xml):
        raise FileNotFoundError(f"Scale setup XML not found: {generic_scale_setup_xml}")

    scale_tool = opensim.ScaleTool(generic_scale_setup_xml)
    scale_tool.setSubjectMass(mass_kg)
    scale_tool.setSubjectHeight(height_m * 1000.0)
    scale_tool.getModelScaler().setMarkerFileName(processed_trc_path)
    scale_tool.getMarkerPlacer().setMarkerFileName(processed_trc_path)
    scale_tool.setName('LaiUhlrich2022-scaled_OC')

    scaled_model_path = os.path.join(output_dir, 'LaiUhlrich2022_scaled.osim')
    scale_tool.getMarkerPlacer().setOutputModelFileName(scaled_model_path)
    scale_tool.getGenericModelMaker().setModelFileName(generic_model_path)

    setup_xml_path = os.path.join(output_dir, 'scale_setup_LaiUhlrich2022.xml')
    scale_tool.printToXML(setup_xml_path)

    # =========================================================================
    # 7. Run scaling via OpenSim command line
    # =========================================================================
    files_before = set()
    for root, _, files in os.walk(output_dir):
        for file in files:
            files_before.add(os.path.join(root, file))

    cmd = [opensim_cmd_path, "run-tool", setup_xml_path]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=output_dir
    )

    if result.returncode != 0:
        raise RuntimeError(
            "OpenSim scaling failed with return code "
            f"{result.returncode}\n\n"
            f"stdout:\n{result.stdout}\n\n"
            f"stderr:\n{result.stderr}"
        )

    # =========================================================================
    # 8. Verify and locate the scaled model
    # =========================================================================
    if os.path.exists(scaled_model_path):
        return scaled_model_path, height_m, mass_kg

    osim_files = []
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.osim'):
                osim_files.append(os.path.join(root, file))

    if osim_files:
        # Prefer any file with "scaled" in the name
        prioritized = sorted(
            osim_files,
            key=lambda p: (0 if 'scaled' in os.path.basename(p).lower() else 1, p)
        )
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


import os
import numpy as np

def convert_trc_to_mm_in_place(trc_path):
    """
    For a single TRC file:
      - Read header and data
      - Check Units (m vs mm)
      - If in meters, convert marker columns to mm and overwrite the file
      - If already mm, do nothing

    Returns
    -------
    bool
        True if conversion was performed, False if not needed.
    """
    with open(trc_path, "r") as f:
        lines = f.readlines()

    if len(lines) < 6:
        raise ValueError(f"TRC file looks too short: {trc_path}")

    delimiter = "\t"

    # Find the metadata keys and values (DataRate, CameraRate, Units, etc.)
    metadata_keys = lines[1].strip().split(delimiter)
    metadata_vals = lines[2].strip().split(delimiter)

    # Locate the Units field
    units_idx = None
    for i, key in enumerate(metadata_keys):
        if key.strip().lower() == "units":
            units_idx = i
            break

    if units_idx is None:
        raise ValueError(f"Could not find 'Units' field in TRC header for {trc_path}")

    current_units = metadata_vals[units_idx].strip().lower()

    # Already in mm, nothing to do
    if current_units in ("mm", "millimeter", "millimeters"):
        return False

    # Need to convert if in meters
    if current_units not in ("m", "meter", "meters"):
        raise ValueError(
            f"Units field is '{metadata_vals[units_idx]}' in {trc_path}, "
            "expected 'm' or 'mm'. Not converting."
        )

    # Find where numeric data begin:
    # Look for "Frame#" line, then skip XYZ labels line, then optional blank line.
    frame_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Frame#"):
            frame_idx = i
            break

    if frame_idx is None:
        raise ValueError(f"Could not find 'Frame#' line in TRC header for {trc_path}")

    # After "Frame#..." (frame_idx), you have:
    # frame_idx+1: XYZ labels line
    # frame_idx+2: often blank
    # Data start at either frame_idx+2 or frame_idx+3 depending on file
    if frame_idx + 2 < len(lines) and lines[frame_idx + 2].strip() == "":
        start_row = frame_idx + 3
    else:
        start_row = frame_idx + 2

    header_lines = lines[:start_row]
    data_lines = lines[start_row:]

    # Load numeric data
    data = np.genfromtxt(data_lines, delimiter=delimiter, filling_values=np.nan)

    if data.ndim == 1:
        data = data[None, :]

    if data.shape[1] < 3:
        raise ValueError(f"TRC data in {trc_path} has too few columns to contain markers.")

    # Convert coordinates from meters to millimeters
    data[:, 2:] *= 1000.0

    # Update Units in header to mm
    metadata_vals[units_idx] = "mm"
    lines[2] = delimiter.join(metadata_vals) + "\n"

    # Rebuild numeric part, preserving frame and time
    new_data_lines = []
    for row in data:
        frame = int(row[0])
        time = row[1]
        coords = row[2:]
        line = f"{frame}\t{time:.7f}"  # keep decent time precision
        line += "".join(f"\t{val:.6f}" for val in coords)
        line += "\n"
        new_data_lines.append(line)

    # Overwrite the same file with updated header and data
    with open(trc_path, "w") as f:
        f.writelines(lines[:start_row])
        f.writelines(new_data_lines)

    return True


def convert_all_trc_in_folder_to_mm(folder):
    """
    Walk through a folder, find all .trc files, and ensure they are in mm.
    Any file with Units == m will be converted in place to mm.

    Parameters
    ----------
    folder : str
        Path to the folder containing TRC files.
    """
    folder = os.path.abspath(folder)
    #print(f"Checking TRC units in {folder}")

    for name in os.listdir(folder):
        if not name.lower().endswith(".trc"):
            continue

        trc_path = os.path.join(folder, name)
        try:
            converted = convert_trc_to_mm_in_place(trc_path)
        except Exception as e:
            print(f"  Skipped {name} due to error: {e}")


# Example usage:
# marker_folder = os.path.join(session_folder, "MarkerData")
# convert_all_trc_in_folder_to_mm(marker_folder)


def create_rajogopal_armless_model(
    generic_model_path: str,
    generic_scale_setup_xml: str,
    session_metadata_path: str,
    static_trc_path: str,
    output_dir: str,
    opensim_install_dir: str = None
):
    """
    Scale an OpenSim model using static trial data using specific marker set.
    
    Parameters
    ----------
    generic_model_path : str
        Path to unscaled generic model (.osim)
    generic_scale_setup_xml : str
        Path to generic scaling setup XML
    session_metadata_path : str
        Path to sessionMetadata.yaml with height_m and mass_kg
    static_trc_path : str
        Path to static TRC file
    output_dir : str
        Directory to save scaled model and processed files
    opensim_install_dir : str, optional
        Path to OpenSim installation directory. If None, auto detection is used.
        
    Returns
    -------
    scaled_model_path : str
        Path to the scaled model file
    height_m : float
        Subject height in meters
    mass_kg : float
        Subject mass in kg
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    trc_dir = os.path.join(os.path.dirname(session_metadata_path), 'ForGaitDynamics', 'TRC_Files')
    os.makedirs(trc_dir, exist_ok=True)

    # =========================================================================
    # 1. Find OpenSim command line tool
    # =========================================================================
    opensim_cmd_path = None

    if opensim_install_dir:
        candidate_path = os.path.join(opensim_install_dir, 'bin', 'opensim-cmd.exe')
        if os.path.exists(candidate_path):
            opensim_cmd_path = candidate_path

    if not opensim_cmd_path:
        opensim_cmd_path = find_opensim_cmd()

    if not opensim_cmd_path:
        raise FileNotFoundError(
            "Could not find opensim-cmd.exe. Please ensure OpenSim is installed and "
            "either provide opensim_install_dir or add OpenSim to your PATH."
        )

    # =========================================================================
    # 2. Read height and mass from session metadata
    # =========================================================================
    height_m = None
    mass_kg = None

    with open(session_metadata_path, 'r') as f:
        for line in f:
            line = line.strip()
            if 'height_m:' in line:
                try:
                    height_m = float(line.split('height_m:')[-1].strip())
                except ValueError:
                    pass
            if 'mass_kg:' in line:
                try:
                    mass_kg = float(line.split('mass_kg:')[-1].strip())
                except ValueError:
                    pass

    if height_m is None or mass_kg is None:
        raise ValueError(f"Could not read height or mass from {session_metadata_path}")

     # =========================================================================
    # 3. Define marker mappings (new and legacy, for back compatibility)
    # =========================================================================
    marker_mapping_offset = {
        'C7_study_offsetRemoved': 'C7',
        'r_shoulder_study_offsetRemoved': 'R_Shoulder',
        'L_shoulder_study_offsetRemoved': 'L_Shoulder',
        'r.ASIS_study_offsetRemoved': 'r.ASIS',
        'L.ASIS_study_offsetRemoved': 'L.ASIS',
        'r.PSIS_study_offsetRemoved': 'r.PSIS',
        'L.PSIS_study_offsetRemoved': 'L.PSIS',
        'r_knee_study_offsetRemoved': 'r_knee',
        'L_knee_study_offsetRemoved': 'L_knee',
        'r_mknee_study_offsetRemoved': 'r_mknee',
        'L_mknee_study_offsetRemoved': 'L_mknee',
        'r_ankle_study_offsetRemoved': 'r_ankle',
        'L_ankle_study_offsetRemoved': 'L_ankle',
        'r_mankle_study_offsetRemoved': 'r_mankle',
        'L_mankle_study_offsetRemoved': 'L_mankle',
        'r_calc_study_offsetRemoved': 'r_calc',
        'L_calc_study_offsetRemoved': 'L_calc',
        'r_toe_study_offsetRemoved': 'r_toe',
        'L_toe_study_offsetRemoved': 'L_toe',
        'r_5meta_study_offsetRemoved': 'r_5meta',
        'L_5meta_study_offsetRemoved': 'L_5meta',
        'r_thigh1_study_offsetRemoved': 'r_thigh1',
        'r_thigh2_study_offsetRemoved': 'r_thigh2',
        'r_thigh3_study_offsetRemoved': 'r_thigh3',
        'L_thigh1_study_offsetRemoved': 'L_thigh1',
        'L_thigh2_study_offsetRemoved': 'L_thigh2',
        'L_thigh3_study_offsetRemoved': 'L_thigh3',
        'r_sh1_study_offsetRemoved': 'r_sh1',
        'r_sh2_study_offsetRemoved': 'r_sh2',
        'r_sh3_study_offsetRemoved': 'r_sh3',
        'L_sh1_study_offsetRemoved': 'L_sh1',
        'L_sh2_study_offsetRemoved': 'L_sh2',
        'L_sh3_study_offsetRemoved': 'L_sh3',
        'RHJC_study_offsetRemoved': 'R_HJC',
        'LHJC_study_offsetRemoved': 'L_HJC',
        'r_lelbow_study_offsetRemoved': 'r_lelbow',
        'L_lelbow_study_offsetRemoved': 'L_lelbow',
        'r_melbow_study_offsetRemoved': 'r_melbow',
        'L_melbow_study_offsetRemoved': 'L_melbow',
        'r_lwrist_study_offsetRemoved': 'r_lwrist',
        'L_lwrist_study_offsetRemoved': 'L_lwrist',
        'r_mwrist_study_offsetRemoved': 'r_mwrist',
        'L_mwrist_study_offsetRemoved': 'L_mwrist'
    }

    # legacy naming without "offsetRemoved"
    marker_mapping = {
        'C7_study': 'C7',
        'r_shoulder_study': 'R_Shoulder',
        'L_shoulder_study': 'L_Shoulder',
        'r.ASIS_study': 'r.ASIS',
        'L.ASIS_study': 'L.ASIS',
        'r.PSIS_study': 'r.PSIS',
        'L.PSIS_study': 'L.PSIS',
        'r_knee_study': 'r_knee',
        'L_knee_study': 'L_knee',  # fixed typo L_knee_stud
        'r_mknee_study': 'r_mknee',
        'L_mknee_study': 'L_mknee',
        'r_ankle_study': 'r_ankle',
        'L_ankle_study': 'L_ankle',
        'r_mankle_study': 'r_mankle',
        'L_mankle_study': 'L_mankle',
        'r_calc_study': 'r_calc',
        'L_calc_study': 'L_calc',
        'r_toe_study': 'r_toe',
        'L_toe_study': 'L_toe',
        'r_5meta_study': 'r_5meta',
        'L_5meta_study': 'L_5meta',
        'r_thigh1_study': 'r_thigh1',
        'r_thigh2_study': 'r_thigh2',
        'r_thigh3_study': 'r_thigh3',
        'L_thigh1_study': 'L_thigh1',
        'L_thigh2_study': 'L_thigh2',
        'L_thigh3_study': 'L_thigh3',
        'r_sh1_study': 'r_sh1',
        'r_sh2_study': 'r_sh2',
        'r_sh3_study': 'r_sh3',
        'L_sh1_study': 'L_sh1',
        'L_sh2_study': 'L_sh2',
        'L_sh3_study': 'L_sh3',
        'RHJC_study': 'R_HJC',
        'LHJC_study': 'L_HJC',
        'r_lelbow_study': 'r_lelbow',
        'L_lelbow_study': 'L_lelbow',
        'r_melbow_study': 'r_melbow',
        'L_melbow_study': 'L_melbow',
        'r_lwrist_study': 'r_lwrist',
        'L_lwrist_study': 'L_lwrist',
        'r_mwrist_study': 'r_mwrist',
        'L_mwrist_study': 'L_mwrist'
    }

    # =========================================================================
    # 4. Load and process static TRC file and choose mapping
    # =========================================================================
    with open(static_trc_path, 'r') as f:
        lines = f.readlines()

    data_start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('Frame#') or line.startswith('Time'):
            data_start_idx = i
            break

    header_line = lines[data_start_idx].strip().split('\t')
    all_marker_names = [name for name in header_line if name][2:]

    # decide which mapping to use based on presence of "offsetRemoved"
    has_offset_removed = any("offsetremoved" in name.lower()
                             for name in all_marker_names)

    if has_offset_removed:
        marker_mapping = marker_mapping_offset
    else:
        marker_mapping = marker_mapping

    # now continue as before, but using the chosen marker_mapping
    available_markers = [name for name in marker_mapping.keys()
                         if name in all_marker_names]

    if len(available_markers) < 10:
        raise ValueError(
            f"Too few target markers found in TRC for scaling "
            f"({len(available_markers)} found)."
        )

    data_lines = lines[data_start_idx + 2:]
    data = []
    for line in data_lines:
        if not line.strip():
            continue
        parts = line.strip().split('\t')
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

    marker_indices = [all_marker_names.index(m) for m in available_markers]

    OC_mrkdata_specific = []
    for idx in marker_indices:
        c0 = 2 + idx * 3
        if c0 + 2 >= data.shape[1]:
            raise ValueError(
                f"Marker index {idx} out of range for TRC data columns."
            )
        OC_mrkdata_specific.extend([
            data[:, c0],
            data[:, c0 + 1],
            data[:, c0 + 2]
        ])

    OC_mrkdata_specific = np.array(OC_mrkdata_specific).T
    OC_time_zeroed = OC_time - np.min(OC_time)

    final_marker_names = [marker_mapping[m] for m in available_markers]

    # =========================================================================
    # 5. Create processed TRC file
    # =========================================================================
    trc_filename = 'OpenCap_static_specific_markers.trc'
    processed_trc_path = os.path.join(trc_dir, trc_filename)

    with open(processed_trc_path, 'w') as f:
        f.write(f'PathFileType\t4\t(X/Y/Z)\t{trc_filename}\n')
        f.write('DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')
        f.write('100.00\t100.00\t{}\t{}\tmm\t100.00\t1\t{}\n'.format(
            len(OC_time_zeroed), len(final_marker_names), len(OC_time_zeroed)))
        f.write('Frame#\tTime\t')
        f.write('\t\t'.join(final_marker_names))
        f.write('\n')
        f.write('\t\t')
        for i in range(len(final_marker_names)):
            f.write('X{}\tY{}\tZ{}'.format(i + 1, i + 1, i + 1))
            if i < len(final_marker_names) - 1:
                f.write('\t')
        f.write('\n')

        for i in range(len(OC_time_zeroed)):
            f.write('{}\t{:.6f}'.format(i + 1, OC_time_zeroed[i]))
            for j in range(OC_mrkdata_specific.shape[1] // 3):
                marker_idx = j * 3
                f.write('\t{:.6f}\t{:.6f}\t{:.6f}'.format(
                    OC_mrkdata_specific[i, marker_idx],
                    OC_mrkdata_specific[i, marker_idx + 1],
                    OC_mrkdata_specific[i, marker_idx + 2]
                ))
            f.write('\n')

    # =========================================================================
    # 6. Configure ScaleTool and save setup XML
    # =========================================================================
    if not os.path.exists(generic_scale_setup_xml):
        raise FileNotFoundError(f"Scale setup XML not found: {generic_scale_setup_xml}")

    scale_tool = opensim.ScaleTool(generic_scale_setup_xml)
    scale_tool.setSubjectMass(mass_kg)
    scale_tool.setSubjectHeight(height_m * 1000.0)
    scale_tool.getModelScaler().setMarkerFileName(processed_trc_path)
    scale_tool.getMarkerPlacer().setMarkerFileName(processed_trc_path)
    scale_tool.setName('ArmlessRajagopal-scaled_OC')

    scaled_model_path = os.path.join(output_dir, 'scaled_RagagopalArmless.osim')
    scale_tool.getMarkerPlacer().setOutputModelFileName(scaled_model_path)
    scale_tool.getGenericModelMaker().setModelFileName(generic_model_path)

    setup_xml_path = os.path.join(output_dir, 'scale_setup_RagagopalArmless.xml')
    scale_tool.printToXML(setup_xml_path)

    # =========================================================================
    # 7. Run scaling via OpenSim command line
    # =========================================================================
    files_before = set()
    for root, _, files in os.walk(output_dir):
        for file in files:
            files_before.add(os.path.join(root, file))

    cmd = [opensim_cmd_path, "run-tool", setup_xml_path]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=output_dir
    )

    if result.returncode != 0:
        raise RuntimeError(
            "OpenSim scaling failed with return code "
            f"{result.returncode}\n\n"
            f"stdout:\n{result.stdout}\n\n"
            f"stderr:\n{result.stderr}"
        )

    # =========================================================================
    # 8. Verify and locate the scaled model
    # =========================================================================
    if os.path.exists(scaled_model_path):
        return scaled_model_path, height_m, mass_kg

    osim_files = []
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.osim'):
                osim_files.append(os.path.join(root, file))

    if osim_files:
        # Prefer any file with "scaled" in the name
        prioritized = sorted(
            osim_files,
            key=lambda p: (0 if 'scaled' in os.path.basename(p).lower() else 1, p)
        )
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






def reformat_ik_mot_for_gaitdynamics(ik_path):
    """
    Reformat an IK .mot file so its header matches example_mot_complete_kinematics.mot.

    Assumes:
      - example_mot_complete_kinematics.mot is in the same folder as this script
      - IK .mot has the same columns (order and count) as the template, or at least
        the same number of columns after any earlier beta-column clean up
    """

    ik_path = Path(ik_path)

    # Template .mot assumed to live next to this script
    script_dir = Path(__file__).resolve().parent
    template_path = Path(os.path.join(script_dir, 'OpenSimPipeline', 'ForGaitDynamics', 'GaitDynamics_Template_IK.mot'))
    if not template_path.is_file():
        raise FileNotFoundError(
            f"Template file not found: {template_path}. "
            "Ensure example_mot_complete_kinematics.mot is next to this script."
        )

    # -----------------------------
    # Read template header and column line
    # -----------------------------
    with template_path.open("r") as f:
        template_header_lines = []
        template_cols_line = None

        for line in f:
            template_header_lines.append(line)
            if line.strip() == "endheader":
                # Next line is the column names
                template_cols_line = f.readline()
                break

        if template_cols_line is None:
            raise ValueError("Template file missing column header after endheader")

    # -----------------------------
    # Read IK column line and numeric data
    # -----------------------------
    with ik_path.open("r") as f:
        ik_cols_line = None

        for line in f:
            if line.strip() == "endheader":
                # Column names line
                ik_cols_line = f.readline()
                break

        if ik_cols_line is None:
            raise ValueError("IK file missing column header after endheader")

        # Load remaining numeric data
        data = np.loadtxt(f)

    if data.ndim == 1:
        # Handle single row case so shape is (1, ncols)
        data = data[np.newaxis, :]

    nrows, ncols = data.shape

    # -----------------------------
    # Update template header with correct nRows and nColumns
    # -----------------------------
    new_header_lines = []
    for line in template_header_lines:
        if line.startswith("nRows="):
            new_header_lines.append(f"nRows={nrows}\n")
        elif line.startswith("nColumns="):
            new_header_lines.append(f"nColumns={ncols}\n")
        else:
            new_header_lines.append(line)

    # Decide which column header to write
    template_cols = template_cols_line.strip().split()
    ik_cols = ik_cols_line.strip().split()

    if len(template_cols) == ncols:
        # Preferred: use template column header so file matches the example exactly
        cols_line_out = template_cols_line
    elif len(ik_cols) == ncols:
        # Fallback: use IK column header if the counts match
        cols_line_out = ik_cols_line
    else:
        raise ValueError(
            f"Column count mismatch: data has {ncols} columns, "
            f"template header has {len(template_cols)}, IK header has {len(ik_cols)}"
        )

    # -----------------------------
    # Write back to the same IK file
    # -----------------------------
    with ik_path.open("w") as out:
        # Header
        for line in new_header_lines:
            out.write(line)
        # Column names
        out.write(cols_line_out)
        # Data
        for row in data:
            out.write("\t".join(f"{val:.8f}" for val in row) + "\n")
            


def get_marker_set_names(model: opensim.Model) -> list:
    """Get marker names from model's MarkerSet."""
    marker_set = model.getMarkerSet()
    return [marker_set.get(i).getName() for i in range(marker_set.getSize())]


def read_trc_file(trc_path: str):
    """Read TRC file, return markers, time, data."""
    with open(trc_path, 'r') as f:
        lines = f.readlines()

    # Parse metadata
    line3_parts = lines[2].strip().split()
    data_rate = float(line3_parts[0]) if line3_parts else 100.0

    # Parse marker names
    marker_line = lines[3].rstrip('\n\r')
    marker_tokens = marker_line.split('\t')
    marker_names = [tok for tok in marker_tokens[2:] if tok.strip()]

    # Parse data
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


def write_trc_file(marker_names, time_data, marker_data, output_path, data_rate=100.0):
    """Write TRC file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    num_frames = len(time_data)
    num_markers = len(marker_names)
    output_dir = os.path.dirname(output_path)

    with open(output_path, 'w') as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{output_dir}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{data_rate:.1f}\t\t{data_rate:.1f}\t\t{num_frames}\t\t{num_markers}\t\tmm\t{data_rate:.1f}\t1\t{num_frames}\n")

        f.write("Frame#\tTime\t")
        for m in marker_names:
            f.write(f"{m}\t\t\t")
        f.write("\n")

        f.write("\t\t")
        for i in range(1, num_markers + 1):
            f.write(f"X{i}\tY{i}\tZ{i}\t")
        f.write("\n\n")

        for frame_idx, (t, coords) in enumerate(zip(time_data, marker_data)):
            # Only write coordinates for the markers we're keeping
            coords_to_write = coords[:num_markers * 3]
            coord_str = "\t".join(f"{c:.3f}" for c in coords_to_write)
            f.write(f"{frame_idx + 1}\t{t:.5f}\t{coord_str}\n")

    return output_path

def reorder_ik_mot_to_opensim_standard(mot_path: str):
    """
    Reorder columns in IK .mot file to match OpenSim standard order (MATLAB style).
    This is CRITICAL for GaitDynamics API compatibility.
    
    Modifies the file in place.
    """
    import pandas as pd
    import os
    
    # Standard OpenSim coordinate order (pelvis → right leg → left leg → trunk)
    standard_order = [
        'time',
        'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
        'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
        'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
        'knee_angle_r', 'knee_angle_r_beta',
        'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r',
        'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',
        'knee_angle_l', 'knee_angle_l_beta',
        'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l',
        'lumbar_extension', 'lumbar_bending', 'lumbar_rotation'
    ]
    
    if not os.path.exists(mot_path):
        print(f"[reorder_ik_mot] File not found: {mot_path}")
        return None
    
    # Read header
    header_lines = []
    with open(mot_path, 'r') as f:
        for line in f:
            header_lines.append(line)
            if line.strip().lower() == 'endheader':
                break
        df = pd.read_csv(f, sep=r'\s+')
    
    # Check which columns exist
    available_cols = [col for col in standard_order if col in df.columns]
    extra_cols = [col for col in df.columns if col not in standard_order]
    reordered_cols = available_cols + extra_cols
    
    # Reorder
    df_reordered = df[reordered_cols]
    
    # Update nColumns in header
    for i, line in enumerate(header_lines):
        if line.strip().lower().startswith('ncolumns='):
            header_lines[i] = f'nColumns={len(df_reordered.columns)}\n'
            break
    
    # Write back with MATLAB-style formatting (8 decimal places)
    with open(mot_path, 'w') as f:
        f.writelines(header_lines)
        df_reordered.to_csv(f, sep='\t', index=False, float_format='%.8f')
    
   # print(f"  ✓ Reordered columns to OpenSim standard order: {os.path.basename(mot_path)}")
    return reordered_cols


def run_ik_for_gait_dynamics(
    session_folder: str,
    scaled_model_path: str,
    ik_setup_xml: str,
    trial_prefix: str,
    output_dir: str,
    model_type: str,
):
    """
    Run IK for trials in session_folder/ForGaitDynamics matching trial_prefix.
    
    Produces output matching MATLAB by:
    1. Processing TRC to use first 71 markers (foot optimized)
    2. Removing duplicate markers from end
    3. Running IK with processed TRC
    """
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

        # Setup IK
        ik_tool = opensim.InverseKinematicsTool(ik_setup_xml)
        model = opensim.Model(scaled_model_path)
        model.initSystem()
        ik_tool.setModel(model)

        ik_tool.setName(trial_name)
        ik_tool.setMarkerDataFileName(input_trc)

        # Get time range
        tt = opensim.TimeSeriesTableVec3(input_trc)
        time_vec = tt.getIndependentColumn()
        if len(time_vec) > 0:
            ik_tool.setStartTime(float(time_vec[0]))
            ik_tool.setEndTime(float(time_vec[-1]))

        # Output path
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
        #print(f"  Output: {ik_output_mot}")
        
        reorder_ik_mot_to_opensim_standard(ik_output_mot)
        lowpass_filter_pelvis_ty(ik_output_mot)

        ik_outputs.append(ik_output_mot)

    return ik_outputs




# Known suffixes we want to treat as "postfixes" on base marker names
KNOWN_SUFFIXES = ['_study_offsetRemoved', '_offsetRemoved', '_study']

def _split_base_suffix(name):
    """Split a marker name into base + known suffix (if any)."""
    for suf in KNOWN_SUFFIXES:
        if name.endswith(suf):
            return name[:-len(suf)], suf
    return name, ''

def _candidate_variants(canon_name):
    """
    Given a canonical marker name (from walk.trc), return possible variants
    in order of preference, so we can map to walking1-style names.

    Priority:
      1) base + '_study_offsetRemoved'
      2) base + '_offsetRemoved'
      3) canonical name itself
      4) base + '_study'
      5) base (no postfix)
    """
    base, tag = _split_base_suffix(canon_name)
    cands = []

    # Highest priority: explicit study_offsetRemoved on base
    cands.append(base + '_study_offsetRemoved')
    # Also allow "_offsetRemoved" on base
    cands.append(base + '_offsetRemoved')
    # Canonical name as written in template
    cands.append(canon_name)
    # Base with _study tag
    cands.append(base + '_study')
    # Plain base
    cands.append(base)

    # Deduplicate while preserving order
    out = []
    seen = set()
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

def _build_marker_mapping(template_markers, input_markers):
    """
    For each template marker, find the best matching name in input_markers,
    using the priority defined in _candidate_variants.
    Returns:
      mapping: dict[template_name] -> input_index (0-based) or None if missing
      missing: list of template names that had no match
    """
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
    """
    Read a TRC file, remap its markers so they match the marker names and
    order in template_trc (e.g., walk.trc), and save a new TRC.

    Rules:
      - If trc_in already matches template_trc (e.g., walk.trc itself),
        the output will be effectively unchanged.
      - For other files (e.g., walking1.trc), we:
          * Prefer *_study_offsetRemoved markers
          * Then *_offsetRemoved
          * Then *_study
          * Then the plain base name
      - Extra markers in trc_in are dropped.
      - Any template markers that truly cannot be found are filled with zeros
        (and a warning is printed).

    Parameters
    ----------
    trc_in : str or Path
        Input TRC file path (e.g., walking1.trc).
    template_trc : str or Path
        Template TRC file path (e.g., walk.trc).
    trc_out : str or Path or None
        Output TRC path. If None, "_harmonized" is appended to the input name.

    Returns
    -------
    Path
        Path to the harmonized TRC file.
    """
    trc_in = Path(trc_in)
    template_trc = Path(template_trc)

    if trc_out is None:
        trc_out = trc_in.with_name(trc_in.stem + "_harmonized.trc")
    trc_out = Path(trc_out)

    # --------- Read template header / markers ---------
    with template_trc.open("r", encoding="utf-8", errors="ignore") as f:
        tmpl_lines = f.readlines()

    # Template marker-name line (4th line, 0-based index 3)
    tmpl_marker_line = tmpl_lines[3]
    tmpl_coord_line = tmpl_lines[4]

    tmpl_tok = tmpl_marker_line.strip().split('\t')
    template_markers = [t for t in tmpl_tok[2:] if t.strip() != ""]
    n_template = len(template_markers)

    # --------- Read input header / markers ---------
    with trc_in.open("r", encoding="utf-8", errors="ignore") as f:
        in_lines = f.readlines()

    in_marker_line = in_lines[3]
    in_tok = in_marker_line.strip().split('\t')
    input_markers = [t for t in in_tok[2:] if t.strip() != ""]

    # Build mapping from template markers -> indices in input
    mapping, missing = _build_marker_mapping(template_markers, input_markers)
    if missing:
        print("Warning: some template markers were not found in input:")
        for m in missing:
            print("  ", m)

    # --------- Build new header ---------
    out_lines = []

    # Line 0 and 1: keep as in input (path, header labels)
    out_lines.append(in_lines[0])
    out_lines.append(in_lines[1])

    # Line 2: numeric header row, adjust NumMarkers to n_template
    num_line = in_lines[2].rstrip('\n')
    num_toks = num_line.split('\t')
    if len(num_toks) < 4:
        num_toks += [''] * (4 - len(num_toks))
    num_toks[3] = str(n_template)  # NumMarkers
    out_lines.append('\t'.join(num_toks) + '\n')

    # Line 3 and 4: copy from template so marker names and X1/Y1/Z1 pattern match walk.trc
    out_lines.append(tmpl_marker_line)
    out_lines.append(tmpl_coord_line)

    # Line 5: blank line (as in standard TRC)
    out_lines.append('\n')

    # --------- Find data start row in input ---------
    data_start = None
    for i in range(5, len(in_lines)):
        if in_lines[i].strip() == '':
            continue
        data_start = i
        break

    if data_start is None:
        # No data, just write header
        with trc_out.open("w", encoding="utf-8") as f:
            f.writelines(out_lines)
        return trc_out

    # --------- Remap data rows ---------
    n_input = len(input_markers)

    for i in range(data_start, len(in_lines)):
        line = in_lines[i].strip()
        if not line:
            continue
        toks = line.split('\t')
        if len(toks) < 2:
            continue

        frame = toks[0]
        time = toks[1]
        row_vals = [frame, time]

        for tname in template_markers:
            idx_in = mapping.get(tname)
            if idx_in is None or idx_in < 0 or idx_in >= n_input:
                # Marker truly missing: fill with zeros
                row_vals.extend(['0', '0', '0'])
            else:
                base = 2 + 3 * idx_in
                if base + 2 < len(toks):
                    row_vals.extend(toks[base:base+3])
                else:
                    row_vals.extend(['0', '0', '0'])

        out_lines.append('\t'.join(row_vals) + '\n')

    # --------- Write output ---------
    with trc_out.open("w", encoding="utf-8") as f:
        f.writelines(out_lines)

    return trc_out

def lowpass_filter_pelvis_ty(
    ik_path: str,
    out_path: str = None,
    cutoff_hz: float = 2.0,
    order: int = 4,
):
    """
    Low pass filter pelvis_ty in an OpenSim MOT or STO file.

    Reads the header up to 'endheader', parses the numeric table,
    applies a zero phase Butterworth low pass filter to pelvis_ty only,
    then writes header plus filtered table.

    Parameters
    ----------
    ik_path : str
        Input MOT or STO path.
    out_path : str, optional
        Output path. If None, overwrites ik_path.
    cutoff_hz : float
        Cutoff frequency in Hz (default 2.0).
    order : int
        Effective MATLAB-style filter "order".
        butter(order / 2, ...) in MATLAB maps to butter(order // 2, ...) here.
    """

    if not os.path.isfile(ik_path):
        raise FileNotFoundError(f"IK file not found: {ik_path}")

    # Use input file as output if not provided
    if out_path is None:
        out_path = ik_path

    # Read header and table, same pattern as make_long_ik
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

    # Sampling frequency from time column
    time = df["time"].to_numpy(dtype=float)
    if len(time) < 2:
        raise ValueError("Not enough time samples to filter")

    fs = 1.0 / np.mean(np.diff(time))
    wn = cutoff_hz / (fs * 0.5)

    # Butterworth design in SOS form
    sos = butter(order // 2, wn, btype="low", output="sos")

    # Filter pelvis_ty only
    pelvis_ty = df["pelvis_ty"].to_numpy(dtype=float)
    pelvis_ty_filt = sosfiltfilt(sos, pelvis_ty)
    df["pelvis_ty"] = pelvis_ty_filt

    # nRows stays the same, but keep your header editing pattern in case it is present
    nrows = len(df)
    for i, line in enumerate(header_lines):
        if line.strip().lower().startswith("nrows="):
            header_lines[i] = f"nRows={nrows}\n"
            break

    # Write header and filtered table
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f_out:
        f_out.writelines(header_lines)
        df.to_csv(
            f_out,
            sep="\t",
            index=False,
            float_format="%.8f"
        )



def make_long_ik(ik_path: str,
                 out_path: str,
                 min_duration: float = 5.0,
                 copy_if_long: bool = True):
    """
    Ensure the IK .mot file lasts at least `min_duration` seconds.

    Returns:
        added_rows : int       # number of rows cloned/appended
        final_duration : float # duration after extension

    If the file is already ≥ min_duration:
        - copy to out_path (if copy_if_long=True)
        - returns (0, duration)
    """
    if not os.path.isfile(ik_path):
        print(f"[make_long_ik] IK file not found, skipping: {ik_path}")
        return 0, 0.0

    # Read header lines
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

    # --- Already long enough ---
    if duration >= min_duration:
        # print(f"[make_long_ik] {os.path.basename(ik_path)} is already "
        #       f"{duration:.3f}s ≥ {min_duration}s. Added rows: 0.")
        if copy_if_long:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            shutil.copyfile(ik_path, out_path)
        return 0, duration

    # --- Extend the file ---
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
        df_long = pd.concat(
            [df_long, new_row.to_frame().T],
            ignore_index=True
        )
        added_rows += 1

    final_duration = t_curr - t0

    # Update nRows in header
    nrows = len(df_long)
    for i, line in enumerate(header_lines):
        if line.strip().lower().startswith("nrows="):
            header_lines[i] = f"nRows={nrows}\n"
            break

    # Write
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f_out:
        f_out.writelines(header_lines)
        df_long.to_csv(f_out,
                       sep="\t",
                       index=False,
                       float_format="%.8f")

 
    return added_rows, final_duration

def trim_mot_file(mot_path: str, rows_to_trim: int):
    """
    Trim the last `rows_to_trim` rows from a .mot file.
    Overwrites the file in place.
    """
    if rows_to_trim <= 0:
        # print(f"[trim_mot_file] Nothing to trim for: {mot_path}")
        return

    # Read header
    header_lines = []
    with open(mot_path, "r") as f:
        for line in f:
            header_lines.append(line)
            if line.strip().lower() == "endheader":
                break
        df = pd.read_csv(f, sep=r"\s+")   # updated to avoid warning

    # Trim dataframe
    new_len = len(df) - rows_to_trim
    if new_len <= 0:
        raise ValueError(f"Cannot trim {rows_to_trim} rows from {len(df)} total rows!")

    df_trimmed = df.iloc[:new_len].copy()

    # Update nRows in header
    for i, line in enumerate(header_lines):
        if line.strip().lower().startswith("nrows="):
            header_lines[i] = f"nRows={new_len}\n"
            break

    # Write trimmed file back
    with open(mot_path, "w") as f_out:
        f_out.writelines(header_lines)
        df_trimmed.to_csv(f_out, sep="\t", index=False, float_format="%.8f")
        



def rename_grf_mot_columns(mot_path: str):
    """
    Rename GRF columns following OpenSim naming conventions,
    using the exact same read and write conventions as trim_mot_file.
    """

    rename_map = {
        # ---------------- RIGHT ----------------
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

        # ---------------- LEFT ----------------
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

    # ---- Read header exactly like trim_mot_file ----
    header_lines = []
    with open(mot_path, "r") as f:
        for line in f:
            header_lines.append(line)
            if line.strip().lower() == "endheader":
                break

        # parse the dataframe with the same whitespace rule used in trim_mot_file
        df = pd.read_csv(f, sep=r"\s+")

    # ---- Apply column renaming inside the dataframe ----
    df = df.rename(columns=rename_map)

    # ---- Update nColumns if present ----
    for i, line in enumerate(header_lines):
        if line.strip().lower().startswith("ncolumns="):
            header_lines[i] = f"nColumns={df.shape[1]}\n"

    # ---- Write back using the exact IK .mot convention ----
    with open(mot_path, "w") as f_out:
        f_out.writelines(header_lines)
        df.to_csv(f_out, sep="\t", index=False, float_format="%.8f")
        
     


def _read_mot(path):
    """
    Read an OpenSim .mot file and return:
      header_lines : list of strings (through 'endheader')
      col_names    : list of column names
      df           : pandas DataFrame with numeric data
    """
    with open(path, "r") as f:
        lines = f.readlines()

    header_lines = []
    it = iter(lines)
    # Header through endheader
    for line in it:
        header_lines.append(line)
        if line.strip().lower() == "endheader":
            break

    # Next non-empty line is column header
    for line in it:
        if line.strip():
            col_header = line
            break
    col_names = re.split(r"\s+|\t+", col_header.strip())

    # Remaining lines are numeric data
    df = pd.read_csv(
        StringIO("".join(it)),
        sep=r"\s+|\t+",
        names=col_names,
        engine="python",
    )

    # Drop completely empty rows
    df = df.dropna(how="all")

    return header_lines, col_names, df


def clean_mot_in_place(mot_path, decimals=8):
    """
    Clean an OpenSim .mot file in place so it behaves more like a
    'well-formatted' IK .mot file.

    Steps:
      - drop fully empty rows
      - enforce strictly increasing time (drops repeated/decreasing time rows)
      - for columns with ALL NaNs (except time), set to zero
      - for columns with SOME NaNs, interpolate over time and fill edges
      - update nRows and nColumns in the header
      - overwrite the original file with tab-separated, clean data

    Parameters
    ----------
    mot_path : str
        Full path to the .mot file (e.g., walking1_Optimized.mot).
    decimals : int
        Number of decimal places when writing floats.
    """
    mot_path = os.path.abspath(mot_path)

    # Read original file
    header_lines, col_names, df = _read_mot(mot_path)

    # Ensure we have at least one column (time)
    if len(col_names) == 0:
        raise ValueError(f"No columns found in {mot_path}")

    time_col = col_names[0]

    # Drop fully empty rows (already done in _read_mot, but safe)
    df = df.dropna(how="all").reset_index(drop=True)

    # Enforce strictly increasing time (drop non-increasing rows)
    t = df[time_col].values
    keep_mask = np.ones(len(df), dtype=bool)
    keep_mask[1:] = t[1:] > t[:-1]
    df = df.loc[keep_mask].reset_index(drop=True)

    # Columns that are all NaN (excluding time) -> set to 0
    all_nan_cols = df.columns[(df.isna().all())]
    for col in all_nan_cols:
        if col == time_col:
            continue
        df[col] = 0.0

    # Columns with partial NaNs -> interpolate along time, then fill edges
    partial_nan_cols = df.columns[df.isna().any() & ~df.isna().all()]
    if len(partial_nan_cols) > 0:
        # Interpolate using time as index to respect ordering
        df_interp = df.set_index(time_col)
        df_interp[partial_nan_cols] = (
            df_interp[partial_nan_cols]
            .interpolate(axis=0, limit_direction="both")
        )
        df = df_interp.reset_index()

    # Final safety: any remaining NaNs -> 0
    df = df.fillna(0.0)

    # Update header nRows / nColumns
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

    # Overwrite original file with cleaned version
    float_fmt = f"%.{decimals}f"
    with open(mot_path, "w") as f:
        # header
        f.writelines(new_header)
        # single line of column names
        f.write("\t".join(col_names) + "\n")
        # data
        df.to_csv(
            f,
            sep="\t",
            index=False,
            header=False,
            float_format=float_fmt,
        )




def reformat_ik_mot(orig_path, out_path, decimals=8):
    """
    Read an OpenSim .mot IK file and rewrite it with clean, consistent
    formatting (no blank lines, uniform spacing, updated nRows/nColumns).

    Parameters
    ----------
    orig_path : str
        Path to the original .mot file (e.g., walking1_orig.mot).
    out_path : str
        Path to the reformatted .mot file to write.
    decimals : int, optional
        Number of decimal places for numeric output.
    """
    header_lines = []
    with open(orig_path, "r") as f:
        # 1. Read header up to 'endheader'
        for line in f:
            header_lines.append(line)
            if line.strip().lower() == "endheader":
                break

        # 2. Read the column header line (skip blank lines)
        col_line = f.readline()
        while col_line.strip() == "":
            col_line = f.readline()

        # Split on whitespace or tabs
        col_names = re.split(r"\s+|\t+", col_line.strip())

        # 3. Read the remaining data into a DataFrame
        df = pd.read_csv(
            f,
            sep=r"\s+|\t+",
            names=col_names,
            engine="python",
            comment="#",
        )

    # Drop any completely empty rows that might have slipped in
    df = df.dropna(how="all")

    # 4. Update nRows and nColumns in the header
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

    # 5. Write out the cleaned file
    float_fmt_time = f"{{:11.{decimals}f}}"
    float_fmt_other = f"{{:14.{decimals}f}}"

    with open(out_path, "w") as f:
        # Header (including endheader, no extra blank line after it)
        for line in new_header:
            f.write(line)

        # Column header (one line)
        f.write("".join([
            "\n",                     # blank line between header and columns,
                                     # remove this if you want it exactly like your optimized file
            "\t".join(df.columns),
            "\n"
        ]))

        # Data rows
        for _, row in df.iterrows():
            vals = row.to_numpy()
            time_val = vals[0]
            other_vals = vals[1:]

            # time column
            f.write(float_fmt_time.format(time_val))
            # remaining columns
            for v in other_vals:
                f.write("\t" + float_fmt_other.format(v))
            f.write("\n")
    
    clean_mot_in_place(out_path)



def close_all_loggers():
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    loggers.append(logging.getLogger())  # root

    for log in loggers:
        for handler in getattr(log, "handlers", []):
            handler.close()
            log.removeHandler(handler)
