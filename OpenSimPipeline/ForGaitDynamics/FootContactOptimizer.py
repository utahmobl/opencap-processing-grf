# -*- coding: utf-8 -*-
"""
Foot Position Kinematic Optimizer

@author: Emily Miller
"""
import os
import numpy as np
import torch
from pathlib import Path
from scipy.signal import butter, sosfiltfilt
from gait_analysis import process_gait_data

# ---------------------------------------------------------------------
# Default marker mapping for renaming OpenCap markers
# ---------------------------------------------------------------------
DEFAULT_MARKER_MAPPING = {
    'C7_study': 'C7',
    'r_shoulder_study': 'R_Shoulder',
    'L_shoulder_study': 'L_Shoulder',
    'r.ASIS_study': 'r.ASIS',
    'L.ASIS_study': 'L.ASIS',
    'r.PSIS_study': 'r.PSIS',
    'L.PSIS_study': 'L.PSIS',
    'r_knee_study': 'r_knee',
    'L_knee_study': 'L_knee',
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
    'L_mwrist_study': 'L_mwrist',
}

# Older Legacy naming scheme WITH offsetRemoved marker names
OFFSETREMOVED_MARKER_MAPPING = {
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
    'L_mwrist_study_offsetRemoved': 'L_mwrist',
}
# ---------------------------------------------------------------------
# TRC utilities
# ---------------------------------------------------------------------

def TRCload(filename):
    """
    Load data from a TRC file.

    Returns
    -------
    header : dict
        Header information, including marker names.
    data : np.ndarray
        Numeric data (frames x cols).
    header_lines : list[str]
        Raw header lines.
    """
    delimiter = '\t'
    start_row = 6  # data starts on row 7 in TRC (0 based index)

    with open(filename, 'r') as f:
        lines = f.readlines()

    header_lines = lines[:start_row]
    data_lines = lines[start_row:]

    header = {}
    header['PathFileType'] = header_lines[0].strip()

    metadata_keys = header_lines[1].strip().split(delimiter)
    metadata_values = header_lines[2].strip().split(delimiter)
    header['metadata'] = {k: v for k, v in zip(metadata_keys, metadata_values)}

    marker_names = header_lines[3].strip().split(delimiter)
    header['markername'] = marker_names

    data = np.genfromtxt(data_lines, delimiter=delimiter, filling_values=np.nan)

    return header, data, header_lines


def extract_marker_names(header):
    """
    Extract marker names from TRC header, removing empty entries.

    Returns
    -------
    list[str]
    """
    raw_names = header['markername']
    cleaned = [name for name in raw_names if name != '']
    return cleaned


def rename_markers(marker_names, marker_mapping=None):
    """
    Rename markers according to a mapping dictionary.
    
    Parameters
    ----------
    marker_names : list[str]
        Original marker names.
    marker_mapping : dict, optional
        Dictionary mapping old names to new names.
        If None, uses DEFAULT_MARKER_MAPPING.
    
    Returns
    -------
    list[str]
        Renamed marker names.
    """
    # Decide which marker mapping to use if none was supplied
    if marker_mapping is None:
        has_offset_removed = any('offsetremoved' in n.lower() for n in marker_names)
        if has_offset_removed:
            marker_mapping = OFFSETREMOVED_MARKER_MAPPING
        else:
            marker_mapping = DEFAULT_MARKER_MAPPING
    
    renamed = []
    for name in marker_names:
        if name in marker_mapping:
            renamed.append(marker_mapping[name])
        else:
            renamed.append(name)
    return renamed


def filter_markers_by_mapping_keys(marker_names, marker_data, marker_mapping=None):
    """
    Filter markers to only include those present as KEYS in the marker_mapping.
    This should be used BEFORE renaming markers.
    
    Parameters
    ----------
    marker_names : list[str]
        Original marker names.
    marker_data : np.ndarray
        Marker data array, shape (nSamples, nMarkers*3).
    marker_mapping : dict, optional
        Dictionary mapping old names to new names.
        If None, uses DEFAULT_MARKER_MAPPING.
    
    Returns
    -------
    filtered_names : list[str]
        Filtered marker names (original names, not yet renamed).
    filtered_data : np.ndarray
        Filtered marker data, shape (nSamples, nFilteredMarkers*3).
    """
    if marker_mapping is None:
        marker_mapping = DEFAULT_MARKER_MAPPING
    
    # Get the set of desired marker names (KEYS from mapping)
    desired_markers = set(marker_mapping.keys())
    
    # Find which markers to keep
    indices_to_keep = []
    filtered_names = []
    
    for i, marker in enumerate(marker_names):
        if marker in desired_markers:
            indices_to_keep.append(i)
            filtered_names.append(marker)
    
    # Filter the data columns
    # Each marker has 3 columns (X, Y, Z)
    col_indices = []
    for idx in indices_to_keep:
        col_indices.extend([idx*3, idx*3 + 1, idx*3 + 2])
    
    filtered_data = marker_data[:, col_indices]
    
    return filtered_names, filtered_data


def write_trc_file(time, mrkdata, mrknames, directory, file):
    """
    Write a TRC file.

    Parameters
    ----------
    time : array_like
        Vector of times (nSamples).
    mrkdata : np.ndarray
        Marker data, shape (nSamples, nMarkers*3).
    mrknames : list[str]
        Marker names (length nMarkers).
    directory : str
        Output directory.
    file : str
        Output base name (no extension).
    """
    os.makedirs(directory, exist_ok=True)

    time = np.asarray(time)
    if time.size < 2:
        time = np.append(time, 1.0)

    T = time[1] - time[0]
    f = 1.0 / T
    mk, nk = mrkdata.shape
    n_markers = nk // 3

    filepath = os.path.join(directory, f"{file}.trc")

    try:
        with open(filepath, 'w') as fid:
            # header
            fid.write(f"PathFileType  4\t(X/Y/Z) {directory}\n")
            fid.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
            fid.write(f"{f:.1f}\t\t{f:.1f}\t\t{mk}\t\t{n_markers}\t\tmm\t{f:.1f}\t1\t{mk}\n")

            # marker names
            fid.write("Frame#\tTime\t")
            for m in mrknames:
                fid.write(f"{m}\t\t\t")
            fid.write("\n")

            # XYZ labels
            fid.write("\t\t")
            for i in range(1, n_markers + 1):
                fid.write(f"X{i}\tY{i}\tZ{i}\t")
            fid.write("\n\n")

            # data rows
            for i in range(mk):
                fid.write(f"{i+1}\t{time[i]:.5f}")
                fid.write("\t" + "\t".join(f"{val:.3f}" for val in mrkdata[i, :]))
                fid.write("\n")

        return True
    except Exception as e:
        print(f"Error writing TRC: {e}")
        return False


# ---------------------------------------------------------------------
# Foot marker utilities
# ---------------------------------------------------------------------
# for backwards and forwards compatability with various opencap marker names
def select_foot_marker_names(marker_names):
    """
    Select foot marker names with priority:
    1) *_study_offsetRemoved
    2) *_study
    3) base name (e.g., r_toe, L_calc)

    Returns a list in the order:
    [L_toe_variant, L_calc_variant, r_toe_variant, r_calc_variant]
    """
    priority_map = {
        'L_toe':  ['L_toe_study_offsetRemoved', 'L_toe_study', 'L_toe'],
        'L_calc': ['L_calc_study_offsetRemoved', 'L_calc_study', 'L_calc'],
        'r_toe':  ['r_toe_study_offsetRemoved', 'r_toe_study', 'r_toe'],
        'r_calc': ['r_calc_study_offsetRemoved', 'r_calc_study', 'r_calc'],
    }

    resolved = []
    for base, candidates in priority_map.items():
        chosen = None
        for name in candidates:
            if name in marker_names:
                chosen = name
                break
        if chosen is None:
            raise ValueError(
                f"No suitable marker found for base '{base}'. "
                f"Tried {candidates} but none are in TRC header."
            )
        resolved.append(chosen)

    return resolved


def create_foot_marker_matrix(foot_marker_names, markernames, marker_positions):
        """
        Create a foot marker matrix with shape (1, 4, 3, n_samples).
        
        Parameters:
        - foot_marker_names: List of foot marker names (e.g., ['L_toe_study_offsetRemoved', 'L_calc_study_offsetRemoved', ...])
        - markernames: List of all marker names that correspond to columns in marker_positions
        - marker_positions: A 2D numpy array of shape (n_samples, 531) representing 3D positions of markers
        
        Returns:
        - foot_marker_matrix: A numpy array of shape (1, 4, 3, n_samples) containing the x, y, z coordinates of the foot markers
        """
        
        # Extract the number of samples from the shape of marker_positions
        n_samples = marker_positions.shape[0]  # First dimension represents n_samples
        
        # Initialize the foot marker matrix with zeros, shape (1, 4, 3, n_samples)
        foot_marker_matrix = np.zeros((1, 4, 3, n_samples))
        
        # Loop through the foot marker names and extract the coordinates
        for i, foot_marker in enumerate(foot_marker_names):
            # Find the marker index in markernames (assuming we know the position)
            marker_index = markernames.index(foot_marker)
            
            # Each marker has x, y, and z in consecutive columns, so extract them
            foot_marker_matrix[0, i, 0, :] = marker_positions[:, marker_index * 3]  # x coordinates
            foot_marker_matrix[0, i, 1, :] = marker_positions[:, marker_index * 3 + 1]  # y coordinates
            foot_marker_matrix[0, i, 2, :] = marker_positions[:, marker_index * 3 + 2]  # z coordinates
        
        return foot_marker_matrix


def make_contact_mask(coordinates, mask_ips, mask_cont, toe_threshold, heel_threshold):
    """
    Build a contact mask from vertical coordinates.

    Parameters
    ----------
    coordinates : np.ndarray
        Shape (T, 4) y coordinates for [L_toe, L_heel, R_toe, R_heel].
    mask_ips : np.ndarray
        Ipsilateral mask, length T.
    mask_cont : np.ndarray
        Contralateral mask, length T.
    toe_threshold : float
    heel_threshold : float

    Returns
    -------
    np.ndarray
        Contact mask, shape (T, 4), with 0 or 1 entries.
    """
    thresholds = []

    # 0: ipsilateral toe
    cond0 = coordinates[mask_cont == 1, 0]
    row0_min = np.nanmin(cond0)
    thr0 = row0_min + toe_threshold
    thresholds.append(thr0)

    # 1: ipsilateral heel
    cond1 = coordinates[mask_cont == 1, 1]
    row1_min = np.nanmin(cond1)
    thr1 = row1_min + heel_threshold
    thresholds.append(thr1)

    # 2: contralateral toe
    cond2 = coordinates[mask_ips == 1, 2]
    row2_min = np.nanmin(cond2)
    thr2 = row2_min + toe_threshold
    thresholds.append(thr2)

    # 3: contralateral heel
    cond3 = coordinates[mask_ips == 1, 3]
    row3_min = np.nanmin(cond3)
    thr3 = row3_min + heel_threshold
    thresholds.append(thr3)

    thresholds = np.array(thresholds)[None, :]  # shape (1,4)

    # start from all ones where below threshold, zeros above
    coords_mod = np.where(coordinates < thresholds, 1, coordinates)
    coords_mod[:, 0:2] = coords_mod[:, 0:2] * mask_ips[:, None]
    coords_mod[:, 2:4] = coords_mod[:, 2:4] * mask_cont[:, None]
    coords_mod = np.where(coordinates > thresholds, 0, coords_mod)

    return coords_mod


# ---------------------------------------------------------------------
# FootPositionOptimizer class
# ---------------------------------------------------------------------

class FootPositionOptimizer:
    def __init__(self,
                 marker_positions, 
                 frame_rate, marker_names, 
                 foot_marker_names, 
                 contact,
                 feet_original,
                 device='cpu', 
                 print_loss_terms=False,
                 weights=None):
        """
        Initialize the optimizer with the given marker positions and names.
        
        Args:
            marker_positions (np.array): 3D marker positions (T x N x 3), where T is number of frames, N is number of markers.
            marker_names (list): List of marker names in order.
            foot_marker_names (list): List of foot marker names that should be optimized.
            device (str): Device to run the optimization on ('cpu' or 'cuda').
            print_loss_terms (bool): Whether to print loss terms during optimization.
        """
        self.marker_positions = marker_positions  # (T x N x 3) 3D marker positions
        self.marker_names = marker_names  # List of marker names in order
        self.foot_marker_names = foot_marker_names  # Foot marker names to optimize
        self.foot_names = foot_marker_names  # ['LBigToe', 'LHeel', 'RBigToe', 'RHeel']
        self.n_frames = self.marker_positions.shape[0]  # Number of frames
        self.contact = contact
        
       
        # Map foot names to indices (assuming foot_names correspond to the markers in the 3D data)
        # You need to ensure that the correct index is used for each foot marker
        self.foot_name_to_index = {}

        for foot_marker in foot_marker_names:
            if foot_marker in marker_names:
                index = marker_names.index(foot_marker)
                self.foot_name_to_index[foot_marker] = index
            else:
                raise ValueError(f"Marker {foot_marker} not found in the marker names list.")


        self.device = device
        self.frame_rate = frame_rate
        self.iterations = 10000  # You can adjust this based on your needs
        self.conv_tol = 1e-10  # Convergence tolerance
        self.loss_frequency_init = 1.0  # Initialize this value based on your needs
        self.foot_position_loss_init = 1.0  # Initialize the foot position loss scaling factor
        self.offset_deriv_loss_init = 1.0  # Initialize the offset loss scaling factor
        if weights is None:
            self.weights = {
                            'contact_velocity': 10,
                            'contact_position': 1000,
                            'flat_floor': 10,
                            'offset_deriv': 0.0001
            }
        else:
            self.weights = weights

        reshaped_marker_positions = self.marker_positions.reshape(self.marker_positions.shape[0], -1, 3)

        for foot_marker, index in self.foot_name_to_index.items():
           # print(f"Processing marker: {foot_marker}, index: {index}")

            # Check if the index is valid, i.e., within the bounds of reshaped_marker_positions
            if index < reshaped_marker_positions.shape[1]:
                # Extract Z positions (Z is the 3rd column, index 2)
                marker_z_positions = reshaped_marker_positions[:, index, 2]  # Extract Z positions for this marker
                #print(f"Z positions for {foot_marker}: {marker_z_positions}")
            else:
                print(f"Warning: Index {index} for {foot_marker} is out of bounds.")
        
        
        # Initialize the print_loss_terms flag
        self.print_loss_terms = print_loss_terms

        # initialize design vars
        self.design_vars = []

        self.offset = torch.zeros((1,self.n_frames,1,3),dtype=torch.float32, device=self.device,requires_grad = True)
        self.offset.requires_grad = True

        self.design_vars.append(self.offset)

        self.feet_original = feet_original # TODO CHECK DIMENSIONS (1,T,4,3) & make sure torched
        self.feet = self.feet_original # for initial pass thru loss terms

                
        # Handle contact position loss initialization
        if 'contact_position' in self.weights and self.weights['contact_position'] > 0:
            # Precompute the starts and ends of stretches of 1s in contact
            self.contact_mask = self.debounced_threshold(self.contact)  # Assuming 'contact' exists
            # Padding the contact mask for boundary conditions
            padded_mask = torch.cat([torch.zeros(1, self.contact_mask.shape[1], dtype=torch.bool), self.contact_mask, torch.zeros(1, self.contact_mask.shape[1], dtype=torch.bool)])
            # Calculate where contact starts and ends
            self.contact_starts = (padded_mask[:-1] == False) & (padded_mask[1:] == True)
            self.contact_ends = (padded_mask[:-1] == True) & (padded_mask[1:] == False)
            # Precompute the contact position loss
            self.contact_position_loss_init = self.loss_contact_position().clone().detach()
        
        # Handle contact velocity loss initialization
        if 'contact_velocity' in self.weights and self.weights['contact_velocity'] > 0:
            self.contact_velocity_loss_init = self.loss_contact_velocity().clone().detach()
       
        # Handle flat floor loss initialization
        if 'flat_floor' in self.weights and self.weights['flat_floor'] > 0:
            self.flat_floor_loss_init = self.loss_flat_floor().clone().detach()
            
        # Handle offset derivative loss initialization
        if 'offset_deriv' in self.weights and self.weights['offset_deriv'] > 0:
            self.offset_deriv_loss_init = self.loss_offset_deriv().clone().detach()

    def add_offset(self):
        self.feet = self.feet_original + self.offset # TODO CHECK DIMENSIONS
        return

    def loss_contact_position(self,scale=1):
        # position shouldn't change during a contact phase
        position_var_loss = 0
        
        key3d_feet = self.feet# TODO CHECK DIMENSIONS...it wants (1,T,nFootmarkers,3) [note... .unsqueeze(0) may help]

        for n in range(len(key3d_feet)):
            start_indices = torch.where(self.contact_starts[:, n])[0]
            end_indices = torch.where(self.contact_ends[:, n])[0]
            
            # sum across directions of variance in each direction of foot positions in each contact stretch.
            # The position of the foot keypoint can change between contact phases, but should stay the same within one.
            # This should be more powerful than the velocity loss above for long standing activities.
            variances = [torch.var(key3d_feet[:,start:end, n,:],axis=1).sum() for start, end in zip(start_indices, end_indices) if end > start]
            position_var_loss += torch.sum(torch.stack(variances)) if variances else torch.tensor(0.0)

        # if self.print_loss_terms:
        #     print('contact position loss:' , (position_var_loss/scale).detach().cpu().numpy())
        
        return position_var_loss/scale

    def loss_contact_velocity(self,scale=1):
        # weight 0 velocity by contact probability
        # contact: L_toe, L_heel, R_toe, R_heel
        
        # velocity loss
        #key3d_feet = self.feet[:,:,self.foot_name_to_index,:]
        key3d_feet = self.feet 
        speed_feet = self.compute_speed(key3d_feet, self.frame_rate)
        contact_mask_expanded = self.contact.unsqueeze(0).unsqueeze(2)

        contact_loss = ((contact_mask_expanded * speed_feet)**2).sum()
        
   #     print('contact velocity loss: ' , (contact_loss/scale).detach().cpu().numpy())

        return contact_loss/scale
    
    def loss_offset_deriv(self,scale = 1, diff_n = 1):
        # weight 0 velocity by contact probability
        # contact: L_toe, L_heel, R_toe, R_heel
        
        # velocity loss
        #key3d_feet = self.feet[:,:,self.foot_name_to_index,:]
        func_offset = self.offset.detach()
        #func_offset = func_offset[:,:,:,1].squeeze()   
        func_offset = func_offset[:,:,:,1]

     
        # Ensure the tensor is on the right device (e.g., CPU or GPU)
        func_offset = func_offset.to(torch.float32)
         
         # Calculate the time interval between frames
        dt = 1.0 / self.frame_rate
         
         # Compute the difference in position between consecutive frames
        offset_diff = torch.diff(func_offset, dim=1, n=diff_n)
         
         # Check if offset_diff is empty (i.e., no valid difference)
        if offset_diff.numel() == 0:
           #  print("Warning: No valid offset differences to compute.")
             return torch.tensor(0.0, device=self.device)  # Return a default value if no valid differences
         
         # Compute velocity by dividing the position difference by the time interval
        offset_velocity = offset_diff / dt**diff_n
         
         # Check for invalid velocity values
         
         # Ensure we have the correct shape by replicating the first frame's velocity
        offset_velocity = torch.cat([offset_velocity[:, 0:1, :], offset_velocity], dim=1)
         
         # Compute the average velocity by calculating the norm (magnitude) of the velocity vector
        average_velocity = torch.norm(offset_velocity, dim=-1, keepdim=True).sum()
        average_velocity = average_velocity.detach().to(torch.float64)
        
           
       # print('offset 2nd derivative loss: ', average_velocity.detach().cpu().numpy())
        
        return average_velocity
    
    def debounced_threshold(self,v_mask, high_thresh=.5, low_thresh=.5, min_stretch_len=3):
        """
        Apply a debounced threshold to a TxN matrix.
        
        Args:
        - v_mask (torch.Tensor): A TxN matrix.
        - high_thresh (float): Threshold for switching from False to True.
        - low_thresh (float): Threshold for switching from True to False.
        - min_stretch_len (int): Minimum length of a stretch to trigger a state change.
        
        Returns:
        - torch.Tensor: The debounced thresholded TxN matrix.
        """
        T, N = v_mask.shape
        debounced = torch.zeros_like(v_mask, dtype=torch.bool)

        for n in range(N):
            column = v_mask[:, n]
            state = column[0] > high_thresh
            stretch_len = 0

            for t in range(T):
                if (state and column[t] < low_thresh) or (not state and column[t] > high_thresh):
                    stretch_len += 1
                    if stretch_len >= min_stretch_len:
                        state = not state
                        stretch_len = 0
                else:
                    stretch_len = 0

                debounced[t, n] = state

        return debounced


    def compute_speed(self, key_3d, frame_rate, diff_n=1):
        """
        Compute the velocity of points in a 3D trajectory.

        :param key_3d: Tensor or numpy array of shape (B, T, N, 3), where B is batch size, T is time, N is number of markers, and 3 represents XYZ coordinates.
        :param frame_rate: Frame rate of the data (frames per second).
        :param diff_n: The number of frames over which the difference should be computed.
        :return: Tensor of speeds (B, T, N, 1).
        """
        # If key_3d is a numpy array, convert it to a tensor
        if isinstance(key_3d, np.ndarray):
            key_3d = torch.tensor(key_3d, dtype=torch.float32)

        # Ensure the tensor is on the right device (e.g., CPU or GPU)
        key_3d = key_3d.to(torch.float32)

        # Calculate the time interval between frames
        dt = 1.0 / frame_rate

        # Compute the difference in position between consecutive frames
        position_diff = torch.diff(key_3d, dim=1, n=diff_n)

        # Compute velocity by dividing the position difference by the time interval
        velocity = position_diff / dt**diff_n

        # Ensure we have the correct shape by replicating the first frame's velocity
        # Remove extra indices and reshape if necessary
        velocity = torch.cat([velocity[:, 0:1, :], velocity], dim=1)

        # Compute the average velocity by calculating the norm (magnitude) of the velocity vector
        average_velocity = torch.norm(velocity, dim=-1, keepdim=True)

        return average_velocity
    
    def loss_flat_floor(self, scale=1):
        # compute the variance in y-position of foot markers when they are on the ground, as defined by the contact mask
        # this is only valid if all foot contact is on a flat floor
        key3d_feet_y = self.feet[:,:,:,1].squeeze()
        masked_feet_y = key3d_feet_y[self.contact_mask] # TODO CHECK DIMENSIONS...SHOULD BE RIGHT
        loss = torch.var(masked_feet_y)

        # if self.print_loss_terms:
        #     print('flat floor loss:' , (loss/scale).detach().cpu().numpy())

        return loss/scale

    def objective_function(self):
        loss = 0
        if 'contact_position' in self.weights and self.weights['contact_position'] > 0:  
            loss += self.weights['contact_position'] * self.loss_contact_position(scale=self.contact_position_loss_init)
        if 'contact_velocity' in self.weights and self.weights['contact_velocity'] > 0: 
            loss += self.weights['contact_velocity'] * self.loss_contact_velocity(scale=self.contact_velocity_loss_init)
        if 'flat_floor' in self.weights and self.weights['flat_floor'] > 0:
            loss += self.weights['flat_floor'] * self.loss_flat_floor(scale=self.flat_floor_loss_init)
        if 'offset_deriv' in self.weights and self.weights['offset_deriv'] > 0:
            loss += self.weights['offset_deriv'] * self.loss_offset_deriv(scale=self.offset_deriv_loss_init)      
        return loss
    
    def optimize(self):
        # Create an L-BFGS optimizer
        optimizer = torch.optim.LBFGS(self.design_vars,
                                      lr = 2,
                                      tolerance_change=self.conv_tol,
                                      line_search_fn="strong_wolfe")
    
        # Define the closure function that reevaluates the model
        def closure():
            optimizer.zero_grad()
            self.add_offset()
            self.loss = self.objective_function()
          #  print('loss: ', self.loss.detach().cpu().numpy())
            self.loss.backward()
            return self.loss
    
        # Optimization loop
        objective_values = torch.zeros(self.iterations)
        for i in range(self.iterations):  
            optimizer.step(closure)
            objective_values[i] = self.loss.clone().detach().cpu()
            self.last_loss = self.loss.clone().detach().cpu()

        # # Print the loss function components multiplied by their weights, if the weight term exists
        # if 'contact_velocity' in self.weights and self.weights['contact_velocity'] > 0:
        #     print('weighted contact velocity loss: ', self.weights['contact_velocity'] * self.loss_contact_velocity(scale=self.contact_velocity_loss_init).detach().cpu().numpy())
        # if 'contact_position' in self.weights and self.weights['contact_position'] > 0:
        #     print('weighted contact position loss: ', self.weights['contact_position'] * self.loss_contact_position(scale=self.contact_position_loss_init).detach().cpu().numpy())
        # if 'flat_floor' in self.weights and self.weights['flat_floor'] > 0:
        #     print('weighted flat floor loss: ', self.weights['flat_floor'] * self.loss_flat_floor().detach().cpu().numpy())
        # if 'offset_deriv' in self.weights and self.weights['offset_deriv'] > 0:
        #     print('weighted offset 2nd derivative loss: ', self.weights['offset_deriv'] * self.loss_offset_deriv(scale=self.offset_deriv_loss_init).detach().cpu().numpy())
        output = {
                    'offset':self.offset.detach(),
                  }
        return output


# ---------------------------------------------------------------------
# Public function: refine_foot_kinematics_trc
# ---------------------------------------------------------------------

def refine_foot_kinematics_trc(
    trc_path,
    session_dir,
    save_dir,
    trimming_start,
    trimming_end,
    lowpass_cutoff_frequency=6.0,
    n_gait_cycles=-1,
    gait_style='overground',
    frame_rate=60.0,
    toe_threshold=38.0,
    heel_threshold=45.0,
    foot_marker_names=None,
    device='cpu',
    print_loss_terms=False,
    weights=None,
    side='l',
    rename_markers_on_save=True,
    filter_markers_on_save=True,
    marker_mapping=None,
):
    """
    Refine foot kinematics for a single TRC and save the updated TRC.

    Parameters
    ----------
    trc_path : str
        Path to input TRC.
    session_dir : str
        Path to session directory for gait analysis.
    save_dir : str
        Directory to save refined TRC.
    lowpass_cutoff_frequency : float
        Cutoff frequency for lowpass filter in gait analysis.
    n_gait_cycles : int
        Number of gait cycles to analyze.
    gait_style : str
        Gait style ('overground' or 'treadmill').
    trimming_start : float
        Time to trim from start (seconds).
    trimming_end : float
        Time to trim from end (seconds).
    frame_rate : float
        Frame rate of the TRC data.
    toe_threshold : float
        Vertical threshold for toe contact detection.
    heel_threshold : float
        Vertical threshold for heel contact detection.
    foot_marker_names : list[str], optional
        Foot marker names. If None, auto-detected.
    device : str
        Device for optimization ('cpu' or 'cuda').
    print_loss_terms : bool
        Whether to print loss terms during optimization.
    weights : dict, optional
        Custom weights for loss terms.
    side : str
        'l' or 'r' for ipsilateral leg in gait analysis.
    rename_markers_on_save : bool
        If True, rename markers using marker_mapping after optimization.
    filter_markers_on_save : bool
        If True, filter markers to only include those in marker_mapping BEFORE optimization.
        This keeps only markers that are KEYS in the mapping dictionary, then renames them
        to the VALUES after optimization. This ensures no duplicate markers in output.
    marker_mapping : dict, optional
        Dictionary mapping old marker names to new names.
        If None and rename_markers_on_save or filter_markers_on_save is True, 
        uses DEFAULT_MARKER_MAPPING.

    Returns
    -------
    str
        Full path to the refined TRC.
    """
    os.makedirs(save_dir, exist_ok=True)

    # infer trial name from TRC filename
    trc_basename = os.path.basename(trc_path)
    # if trc_basename.endswith('_videoAndMocap.trc'):
    #     trial_name = trc_basename.replace('_videoAndMocap.trc', '')
    # else:
    trial_name = os.path.splitext(trc_basename)[0]

    # gait masks
    mask_ips, mask_cont, gait_events, foot_positions, time = process_gait_data(
        session_dir=session_dir,
        trial_name=trial_name,
        leg=side,
        lowpass_cutoff_frequency=lowpass_cutoff_frequency,
        n_gait_cycles=n_gait_cycles,
        gait_style=gait_style,
        trimming_start=trimming_start,
        trimming_end=trimming_end,
    )

    # load TRC
    header, data, header_lines = TRCload(trc_path)
    data = data[:, ~np.all(np.isnan(data), axis=0)]

    frame = data[:, 0]
    times = data[:, 1]
    marker_xyz = data[:, 2:]
    
    t = np.round(times, 6)
    
    # Total duration
    duration = t[-1] - t[0]
    
    # Basic sanity checks
    if trimming_start < 0 or trimming_end < 0:
        raise ValueError(f"trimming_start and trimming_end must be >= 0, got {trimming_start}, {trimming_end}")
    
    if trimming_start + trimming_end >= duration:
        raise ValueError(
            f"Requested trimming_start {trimming_start}s and trimming_end {trimming_end}s "
            f"remove >= full duration {duration:.4f}s"
        )
    
    # Effective absolute times to keep
    if trimming_start > 0:
        t_start = t[0] + trimming_start
    else:
        t_start = t[0]
    
    if trimming_end > 0:
        t_end = t[-1] - trimming_end
    else:
        t_end = t[-1]
    
    if t_start >= t_end:
        raise ValueError(
            f"After applying trimming_start={trimming_start} and trimming_end={trimming_end}, "
            f"t_start={t_start:.6f} >= t_end={t_end:.6f}"
        )
    
    # Find indices: first frame >= t_start, last frame <= t_end
    idx_start = np.where(t >= np.round(t_start, 6))[0][0]
    idx_end = np.where(t <= np.round(t_end, 6))[0][-1] + 1  # slice end is exclusive
    
    # Final safety check
    if idx_end <= idx_start:
        raise ValueError(
            f"Invalid trimming: idx_start={idx_start}, idx_end={idx_end}. "
            f"t_start={t_start:.6f}, t_end={t_end:.6f}"
        )
    
    # Apply trimming
    times = times[idx_start:idx_end]
    marker_xyz = marker_xyz[idx_start:idx_end, :]
    frame = frame[idx_start:idx_end]

    marker_names = extract_marker_names(header)
    marker_names = [n for n in marker_names if n not in ['Frame#', 'Time']]

    # Filter markers if requested (BEFORE optimization and renaming)
    if filter_markers_on_save:
        marker_names, marker_xyz = filter_markers_by_mapping_keys(
            marker_names, 
            marker_xyz, 
            marker_mapping
        )

    # resolve foot marker names by priority:
    # *_study_offsetRemoved > *_study > base name
    if foot_marker_names is None:
        foot_marker_names = select_foot_marker_names(marker_names)

    # foot matrix and contact mask
    foot_matrix = create_foot_marker_matrix(foot_marker_names, marker_names, marker_xyz)
    y_coords = foot_matrix[0, :, 1, :].T  # (T,4)
    contact_mask_np = make_contact_mask(y_coords, mask_ips, mask_cont, toe_threshold, heel_threshold)
    
    # match old behavior: inherit dtype from numpy (likely float64)
    contact_mask = torch.tensor(contact_mask_np)  
    
    feet_tensor = torch.tensor(foot_matrix).permute(0, 3, 1, 2)
    

    # optimizer
    optimizer = FootPositionOptimizer(
        marker_positions=marker_xyz,
        frame_rate=int(frame_rate),
        marker_names=marker_names,
        foot_marker_names=foot_marker_names,
        contact=contact_mask,
        feet_original=feet_tensor,
        device=device,
        print_loss_terms=print_loss_terms,
        weights=weights,
    )



    output = optimizer.optimize()

    # offset
    offset = output['offset'].detach().cpu().numpy()   # (1,T,1,3)
    offset = offset.reshape(-1, 3)                     # (T,3)
    offset_y = offset[:, 1]

    # filter offset_y
    filt_freq = 6.0
    nyquist = 0.5 * (1.0 / np.mean(np.diff(times)))
    wn = filt_freq / nyquist
    sos = butter(2, wn, btype='low', output='sos')
    offset_y_filt = sosfiltfilt(sos, offset_y, padlen=50)

    offset_y_filt = offset_y_filt[:, None]
    updated_marker_xyz = marker_xyz.copy()
    updated_marker_xyz[:, 1::3] += offset_y_filt

    # Rename markers if requested (after optimization, markers were already filtered before)
    if rename_markers_on_save:
        output_marker_names = rename_markers(marker_names, marker_mapping)
    else:
        output_marker_names = marker_names

    out_basename = f"MarkerData_optfeet_{trial_name}"
    write_trc_file(
        time=times,
        mrkdata=updated_marker_xyz,
        mrknames=output_marker_names,
        directory=save_dir,
        file=out_basename,
    )

    out_path = os.path.join(save_dir, out_basename + '.trc')
    return out_path

def get_trc_frame_rate(trc_path):
    """
    Extract the frame rate from a TRC file by reading only the header.

    Returns:
        float : frame rate (Hz)
    """
    with open(trc_path, 'r') as f:
        lines = f.readlines()

    # Line 2 of TRC header contains DataRate, CameraRate, NumFrames...
    # Example:
    # DataRate	CameraRate	NumFrames	NumMarkers	Units	OrigDataRate	OrigDataStartFrame	OrigNumFrames
    # 60.00       60.00       186         181         mm      60.00           1                   186

    # Look for the line that contains numeric values for DataRate
    for line in lines:
        parts = line.strip().split()
        # Look for float-like strings followed by more numbers
        if len(parts) >= 2:
            try:
                data_rate = float(parts[0])  # this is typically DataRate
                camera_rate = float(parts[1])
                return camera_rate  # both are the same, either is fine
            except ValueError:
                continue

    raise ValueError(f"Could not extract frame rate from TRC file: {trc_path}")

def refine_foot_kinematics_for_session(
    session_folder, 
    trial_prefix, 
    gait_style, 
    trimming_start,
    trimming_end,
    rename_markers_on_save=True,
    filter_markers_on_save=True,
    marker_mapping=None,
):
    """
    Run kinematics refinement for better foot contact.
    
    Parameters
    ----------
    session_folder : Path or str
        Path to the session folder.
    trial_prefix : str
        Prefix to filter trial names.
    gait_style : str
        Gait style ('overground' or 'treadmill').
    rename_markers_on_save : bool
        If True, rename markers using marker_mapping after optimization.
    filter_markers_on_save : bool
        If True, filter markers to only include those in marker_mapping BEFORE optimization.
    marker_mapping : dict, optional
        Dictionary mapping old marker names to new names.
    """
    
    session_folder = Path(session_folder)
    
    marker_trc_dir = session_folder / "MarkerData"
    refined_trc_dir = session_folder / "ForGaitDynamics"
    refined_trc_dir.mkdir(parents=True, exist_ok=True)

    for fname in os.listdir(marker_trc_dir):
        if (
            fname.endswith(".trc") 
            and "Optimized" not in fname
            and (not trial_prefix or fname.startswith(trial_prefix))
        ):
            trc_path = marker_trc_dir / fname
            print(f"  Optimizing foot positions for {fname}")
            refine_foot_kinematics_trc(
                trc_path=str(trc_path),
                session_dir=str(session_folder),
                save_dir=str(refined_trc_dir),
                gait_style=gait_style,
                trimming_start=trimming_start,
                trimming_end=trimming_end,
                frame_rate=get_trc_frame_rate(trc_path),
                rename_markers_on_save=rename_markers_on_save,
                filter_markers_on_save=filter_markers_on_save,
                marker_mapping=marker_mapping,
            )