"""
    ---------------------------------------------------------------------------
    OpenCap processing: gaitAnalysis.py
    ---------------------------------------------------------------------------

    Copyright 2023 Stanford University and the Authors
    
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
 
import sys
import os

# base_path = r"C:/Users/MoBL3/Documents/opencap-processing/"
# sys.path.append(base_path)  # Add the base path explicitly
# # Append all folders in the base directory to sys.path
# for folder in os.listdir(base_path):
#     folder_path = os.path.join(base_path, folder)
#     if os.path.isdir(folder_path):  # Check if it's a directory
#         sys.path.append(folder_path)

import numpy as np
import copy
import pandas as pd
from scipy.signal import find_peaks
from matplotlib import pyplot as plt

from utilsKinematics_GRF import kinematics
# % Paths.

class gait_analysis_MoCap(kinematics):
    
    def __init__(self, session_dir, trial_name, leg='auto',
                 lowpass_cutoff_frequency_for_coordinate_values=-1,
                 n_gait_cycles=-1, gait_style='auto', trimming_start=0, 
                 trimming_end=0):
        
        # Inherit init from kinematics class.
        super().__init__(
            session_dir, 
            trial_name, 
            lowpass_cutoff_frequency_for_coordinate_values=lowpass_cutoff_frequency_for_coordinate_values)
        
        # We might want to trim the start/end of the trial to remove bad data. 
        # For example, this might be needed with HRNet during overground 
        # walking, where, at the end, the subject is leaving the field of view 
        # but HRNet returns relatively high confidence values. As a result,
        # the trial is not well trimmed. Here, we provide the option to
        # manually trim the start and end of the trial.
        self.trimming_start = trimming_start
        self.trimming_end = trimming_end
                        
        # Marker data load and filter.
        self.markerDict = self.get_marker_dict(session_dir, trial_name, 
            lowpass_cutoff_frequency = lowpass_cutoff_frequency_for_coordinate_values)

        # Coordinate values.
        self.coordinateValues = self.get_coordinate_values()
        
        # Trim marker data and coordinate values.
        if self.trimming_start > 0:
            self.idx_trim_start = np.where(np.round(self.markerDict['time'] - self.trimming_start,6) <= 0)[0][-1]
            self.markerDict['time'] = self.markerDict['time'][self.idx_trim_start:,]
            for marker in self.markerDict['markers']:
                self.markerDict['markers'][marker] = self.markerDict['markers'][marker][self.idx_trim_start:,:]
            self.coordinateValues = self.coordinateValues.iloc[self.idx_trim_start:]
        
        if self.trimming_end > 0:
            self.idx_trim_end = np.where(np.round(self.markerDict['time'],6) <= np.round(self.markerDict['time'][-1] - self.trimming_end,6))[0][-1] + 1
            self.markerDict['time'] = self.markerDict['time'][:self.idx_trim_end,]
            for marker in self.markerDict['markers']:
                self.markerDict['markers'][marker] = self.markerDict['markers'][marker][:self.idx_trim_end,:]
            self.coordinateValues = self.coordinateValues.iloc[:self.idx_trim_end]
        
        # Rotate marker data so x is forward (not using for now, but could be useful for some analyses).
        self.rotation_about_y, self.markerDictRotated = self.rotate_x_forward()

        # Segment gait cycles.
        self.gaitEvents = self.segment_walking(n_gait_cycles=n_gait_cycles,leg=leg)
    
    
    def rotate_x_forward(self):
        # Find the midpoint of the PSIS markers
        try:
            psis_midpoint = (self.markerDict['markers']['r.PSIS'] +
                         self.markerDict['markers']['L.PSIS']) / 2
        except Exception as e:
            psis_midpoint = (
                self.markerDict['markers']['r.PSIS_study'] + self.markerDict['markers']['L.PSIS_study']) / 2

        # Find the midpoint of the ASIS markers
        try:
            asis_midpoint = (self.markerDict['markers']['r.ASIS'] + self.markerDict['markers']['L.ASIS']) / 2
        except Exception as e:
            asis_midpoint = (self.markerDict['markers']['r.ASIS_study'] + self.markerDict['markers']['L.ASIS_study']) / 2

        # Compute the vector pointing from the PSIS midpoint to the ASIS midpoint
        heading_vector = asis_midpoint - psis_midpoint

        # Compute the angle between the heading vector projected onto x-z plane and x-axis
        angle = np.unwrap(np.arctan2(heading_vector[:,2], heading_vector[:,0]))

        # compute average angle during middle 50% of the trial
        n_frames = len(self.markerDict['time'])
        start_index = int(n_frames * 0.25)
        end_index = int(n_frames * 0.75)
        angle = np.degrees(np.mean(angle[start_index:end_index], axis=0))

        # Apply the rotation to the marker data
        marker_dict_rotated = self.rotate_marker_dict(self.markerDict, {'y':angle})

        return angle, marker_dict_rotated
    
        
    
    def segment_walking(self, n_gait_cycles=-1, leg='auto', visualize=False):

        # n_gait_cycles = -1 finds all accessible gait cycles. Otherwise, it 
        # finds that many gait cycles, working backwards from end of trial.
               
        # Helper functions
        def detect_gait_peaks(r_calc_rel_x,
                              l_calc_rel_x,
                              r_toe_rel_x,
                              l_toe_rel_x,
                              prominence = 0.3):
            # Find HS.
            rHS, _ = find_peaks(r_calc_rel_x, prominence=prominence)
            lHS, _ = find_peaks(l_calc_rel_x, prominence=prominence)
            
            # Find TO.
            rTO, _ = find_peaks(-r_toe_rel_x, prominence=prominence)
            lTO, _ = find_peaks(-l_toe_rel_x, prominence=prominence)
            
            return rHS,lHS,rTO,lTO
        
        def detect_correct_order(rHS, rTO, lHS, lTO):
            # checks if the peaks are in the right order
                    
            expectedOrder = {'rHS': 'lTO',
                             'lTO': 'lHS',
                             'lHS': 'rTO',
                             'rTO': 'rHS'}
                    
            # Identify vector that has the smallest value in it. Put this vector name
            # in vName1
            vectors = {'rHS': rHS, 'rTO': rTO, 'lHS': lHS, 'lTO': lTO}
            non_empty_vectors = {k: v for k, v in vectors.items() if len(v) > 0}
        
            # Check if there are any non-empty vectors
            if not non_empty_vectors:
                return True  # All vectors are empty, consider it correct order
        
            vName1 = min(non_empty_vectors, key=lambda k: non_empty_vectors[k][0])
        
            # While there are any values in any of the vectors (rHS, rTO, lHS, or lTO)
            while any([len(vName) > 0 for vName in vectors.values()]):
                # Delete the smallest value from the vName1
                vectors[vName1] = np.delete(vectors[vName1], 0)
        
                # Then find the vector with the next smallest value. Define vName2 as the
                # name of this vector
                non_empty_vectors = {k: v for k, v in vectors.items() if len(v) > 0}
                if not non_empty_vectors:
                    break
                
                # if only one type of event is left, treat it as a trailing partial cycle
                if len(non_empty_vectors) == 1:
                    break
                
                vName2 = min(non_empty_vectors, key=lambda k: non_empty_vectors[k][0])
                
                if vName2 != expectedOrder[vName1]:
                    return False
        
                # Set vName1 equal to vName2 and clear vName2
                vName1, vName2 = vName2, ''
        
            return True
        
        # Subtract sacrum from foot.
        # Get markers with fallback when _study is missing
        try:
            r_calc_m = self.markerDict['markers']['r_calc_study']
        except KeyError:
            r_calc_m = self.markerDict['markers']['r_calc']
        
        try:
            r_toe_m = self.markerDict['markers']['r_toe_study']
        except KeyError:
            r_toe_m = self.markerDict['markers']['r_toe']
        
        try:
            r_PSIS_m = self.markerDict['markers']['r.PSIS_study']
        except KeyError:
            r_PSIS_m = self.markerDict['markers']['r.PSIS']
        
        try:
            L_calc_m = self.markerDict['markers']['L_calc_study']
        except KeyError:
            L_calc_m = self.markerDict['markers']['L_calc']
        
        try:
            L_toe_m = self.markerDict['markers']['L_toe_study']
        except KeyError:
            L_toe_m = self.markerDict['markers']['L_toe']
        
        try:
            L_PSIS_m = self.markerDict['markers']['L.PSIS_study']
        except KeyError:
            L_PSIS_m = self.markerDict['markers']['L.PSIS']
        
        try:
            r_ASIS_m = self.markerDict['markers']['r.ASIS_study']
        except KeyError:
            r_ASIS_m = self.markerDict['markers']['r.ASIS']
        
        try:
            L_ASIS_m = self.markerDict['markers']['L.ASIS_study']
        except KeyError:
            L_ASIS_m = self.markerDict['markers']['L.ASIS']
        
        
        # Subtract sacrum from foot.
        # It looks like the position-based approach will be more robust.        
        r_calc_rel = r_calc_m - r_PSIS_m
        r_toe_rel  = r_toe_m  - r_PSIS_m
        r_toe_rel_x = r_toe_rel[:,0]
        
        # Repeat for left.
        l_calc_rel = L_calc_m - L_PSIS_m
        l_toe_rel  = L_toe_m  - L_PSIS_m
        
        # Identify which direction the subject is walking.
        mid_psis = (r_PSIS_m + L_PSIS_m) / 2
        mid_asis = (r_ASIS_m + L_ASIS_m) / 2

        mid_dir = mid_asis - mid_psis
        mid_dir_floor = np.copy(mid_dir)
        mid_dir_floor[:,1] = 0
        mid_dir_floor = mid_dir_floor / np.linalg.norm(mid_dir_floor,axis=1,keepdims=True)
        
        # Dot product projections   
        r_calc_rel_x = np.einsum('ij,ij->i', mid_dir_floor,r_calc_rel)
        l_calc_rel_x = np.einsum('ij,ij->i', mid_dir_floor,l_calc_rel)
        r_toe_rel_x = np.einsum('ij,ij->i', mid_dir_floor,r_toe_rel)
        l_toe_rel_x = np.einsum('ij,ij->i', mid_dir_floor,l_toe_rel)
        
        # Old Approach that does not take the heading direction into account.
        # r_psis_x = self.markerDict['markers']['r.PSIS'][:,0]
        # r_asis_x = self.markerDict['markers']['r.ASIS'][:,0]
        # r_dir_x = r_asis_x-r_psis_x
        # position_approach_scaling = np.where(r_dir_x > 0, 1, -1)        
        # r_calc_rel_x = r_calc_rel[:,0] * position_approach_scaling
        # r_toe_rel_x = r_toe_rel[:,0] * position_approach_scaling
        # l_calc_rel_x = l_calc_rel[:,0] * position_approach_scaling
        # l_toe_rel_x = l_toe_rel[:,0] * position_approach_scaling
                       
        # Detect peaks, check if they're in the right order, if not reduce prominence.
        # the peaks can be less prominent with pathological or slower gait patterns
        prominences = [0.3, 0.25, 0.2, 0.15]
        
        for i,prom in enumerate(prominences):
            rHS,lHS,rTO,lTO = detect_gait_peaks(r_calc_rel_x=r_calc_rel_x,
                                  l_calc_rel_x=l_calc_rel_x,
                                  r_toe_rel_x=r_toe_rel_x,
                                  l_toe_rel_x=l_toe_rel_x,
                                  prominence=prom)
            if not detect_correct_order(rHS=rHS, rTO=rTO, lHS=lHS, lTO=lTO):
                if prom == prominences[-1]:
                    raise ValueError('The ordering of gait events is not correct. Consider trimming your trial using the trimming_start and trimming_end options.')
                else:
                    print('The gait events were not in the correct order. Trying peak detection again ' +
                      'with prominence = ' + str(prominences[i+1]) + '.')
            else:
                # everything was in the correct order. continue.
                break
        
        if visualize:
            import matplotlib.pyplot as plt
            plt.close('all')
            plt.figure(1)
            plt.plot(self.markerDict['time'],r_toe_rel_x,label='toe')
            plt.plot(self.markerDict['time'],r_calc_rel_x,label='calc')
            plt.scatter(self.markerDict['time'][rHS], r_calc_rel_x[rHS], color='red', label='rHS')
            plt.scatter(self.markerDict['time'][rTO], r_toe_rel_x[rTO], color='blue', label='rTO')
            plt.legend()

            plt.figure(2)
            plt.plot(self.markerDict['time'],l_toe_rel_x,label='toe')
            plt.plot(self.markerDict['time'],l_calc_rel_x,label='calc')
            plt.scatter(self.markerDict['time'][lHS], l_calc_rel_x[lHS], color='red', label='lHS')
            plt.scatter(self.markerDict['time'][lTO], l_toe_rel_x[lTO], color='blue', label='lTO')
            plt.legend()

        # Find the number of gait cycles for the foot of interest.
        if leg=='auto':
            # Find the last HS of either foot.
            if rHS[-1] > lHS[-1]:
                leg = 'r'
            else:
                leg = 'l'
        
        # Find the number of gait cycles for the foot of interest.
        if leg == 'r':
            hsIps = rHS
            toIps = rTO
            hsCont = lHS
            toCont = lTO
        elif leg == 'l':
            hsIps = lHS
            toIps = lTO
            hsCont = rHS
            toCont = rTO

                            
        gaitEvents = {'hsIps':hsIps,
                      'toIps':toIps,
                      'hsCont':hsCont,
                      'toCont':toCont,
                      'ipsilateralLeg':leg,
                      'time': self.markerDict['time']}
        
        return gaitEvents
    


def build_mask_from_events(hs, to, total_length):
    """
    Create a stance mask (1/0) from heel-strike (hs) and toe-off (to) indices.
    Rules:
      - If there is a TO before the first HS: mark 0..TO as stance.
      - For each HS: pair to the first TO strictly after it -> mark HS..TO.
      - If an HS has no following TO: mark HS..end.
    """
    hs = np.asarray(hs, dtype=int)
    to = np.asarray(to, dtype=int)
    hs = hs[np.isfinite(hs)]
    to = to[np.isfinite(to)]

    # keep within [0, total_length-1]
    hs = hs[(hs >= 0) & (hs < total_length)]
    to = to[(to >= 0) & (to < total_length)]

    hs.sort()
    to.sort()

    mask = np.zeros(total_length, dtype=int)

    # 1) If any TO occurs before the first HS, we started mid-stance
    if to.size and (hs.size == 0 or to[0] < hs[0]):
        t0 = int(to[0])
        mask[:t0 + 1] = 1  # inclusive of TO

        # drop that TO from further pairing
        to = to[1:]

    # 2) Pair each HS with the first TO after it
    j = 0  # pointer into 'to'
    for h in hs:
        # advance 'j' until to[j] > h
        while j < to.size and to[j] <= h:
            j += 1

        if j < to.size:
            t = int(to[j])
            mask[h:t + 1] = 1  # inclusive segment HS..TO
            j += 1  # consume that TO
        else:
            # no TO after this HS: stance runs to end
            mask[h:] = 1
            break

    return mask

def create_binary_masks_from_gait_events(gait_events, total_length):
    """
    Robust mask builder using HS/TO pairing for ipsilateral and contralateral legs.
    """
    hsIps  = gait_events['hsIps']
    toIps  = gait_events['toIps']
    hsCont = gait_events['hsCont']
    toCont = gait_events['toCont']

    mask_ips  = build_mask_from_events(hsIps,  toIps,  total_length)
    mask_cont = build_mask_from_events(hsCont, toCont, total_length)

    return mask_ips, mask_cont


# Main function to load data and generate masks
def process_gait_data(session_dir, trial_name, leg, lowpass_cutoff_frequency, n_gait_cycles, gait_style, trimming_start, trimming_end):
    """
    Main function to process gait data, apply filters, and create binary masks.

    Parameters:
    - session_dir: Path to session data directory.
    - trial_name: Name of the trial to analyze.
    - leg: Specify the leg to analyze, e.g., "l" or "r".
    - lowpass_cutoff_frequency: Apply a lowpass filter with cutoff frequency.
    - n_gait_cycles: Number of gait cycles to analyze.
    - gait_style: Gait style, e.g., "treadmill" or "overground".
    - trimming_start: Trim the first few seconds of data.
    - trimming_end: Trim the last few seconds of data.
    
    Returns:
    - mask_ips: Binary mask for Ips.
    - mask_cont: Binary mask for Cont.
    """
    # Initialize gait analysis 
    gait_analysis_l = gait_analysis_MoCap(
        session_dir=session_dir,
        trial_name=trial_name,
        leg=leg,
        lowpass_cutoff_frequency_for_coordinate_values=lowpass_cutoff_frequency,
        n_gait_cycles=n_gait_cycles,
        gait_style=gait_style,
        trimming_start=trimming_start,
        trimming_end=trimming_end
    )
    
    
    
    # Extract gait events from the gait analysis object
    gait_events = gait_analysis_l.gaitEvents
    
    # Determine the total length based on the 'time' in gait events
    total_length = len(gait_events['time'])
    
    # Generate the binary masks for Ips and Cont
    mask_ips, mask_cont = create_binary_masks_from_gait_events(gait_events, total_length)
    
    marker_dict = gait_analysis_l.markerDictRotated
    
    # Extract foot marker positions from the rotated marker data
    try:
        r_calc = marker_dict['markers']['r_calc_study']
    except KeyError:
        r_calc = marker_dict['markers']['r_calc']
    
    try:
        r_toe = marker_dict['markers']['r_toe_study']
    except KeyError:
        r_toe = marker_dict['markers']['r_toe']
    
    try:
        L_calc = marker_dict['markers']['L_calc_study']
    except KeyError:
        L_calc = marker_dict['markers']['L_calc']
    
    try:
        L_toe = marker_dict['markers']['L_toe_study']
    except KeyError:
        L_toe = marker_dict['markers']['L_toe']
    
    foot_positions = {
        'right': {
            'calc': r_calc/1000,
            'toe': r_toe/1000
        },
        'left': {
            'calc': L_calc/1000,
            'toe': L_toe/1000
        }
    }

    
    # Extract time array
    time = marker_dict['time']
    
    return mask_ips, mask_cont, gait_events, foot_positions, time



       



    