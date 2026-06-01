# -*- coding: utf-8 -*-
"""
Foot Position Kinematic Optimizer

@author: Emily Miller
"""
import os
import re
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
# NEW: OpenCap monocular marker mapping
# (use source names from monocular TRCs, map to the same target names used elsewhere)
# ---------------------------------------------------------------------
MONOCULAR_MARKER_MAPPING = {
    # trunk
    'C7': 'C7',
    'sternum': 'sternum',
    'L4': 'L4',
    'T6': 'T6',

    # pelvis
    'r_ASIS': 'r.ASIS',
    'l_ASIS': 'L.ASIS',
    'r_PSIS': 'r.PSIS',
    'l_PSIS': 'L.PSIS',

    # lower limb
    'r_knee': 'r_knee',
    'l_knee': 'L_knee',
    'r_mknee': 'r_mknee',
    'l_mknee': 'L_mknee',
    'r_ankle': 'r_ankle',
    'l_ankle': 'L_ankle',
    'r_mankle': 'r_mankle',
    'l_mankle': 'L_mankle',
    'r_calc': 'r_calc',
    'l_calc': 'L_calc',
    'r_toe': 'r_toe',
    'l_toe': 'L_toe',
    'r_5meta': 'r_5meta',
    'l_5meta': 'L_5meta',

    # optional aliases (keep these out of pose alias names, only true marker columns)
    'r_big_toe': 'r_toe',
    'l_big_toe': 'L_toe',

    # upper limb (if present)
    'r_shoulder': 'R_Shoulder',
    'l_shoulder': 'L_Shoulder',
    'r_elbow': 'r_lelbow',
    'l_elbow': 'L_lelbow',
    'r_melbow': 'r_melbow',
    'l_melbow': 'L_melbow',
    'r_wrist_radius': 'r_lwrist',
    'l_wrist_radius': 'L_lwrist',
    'r_wrist_ulna': 'r_mwrist',
    'l_wrist_ulna': 'L_mwrist',
}

# ---------------------------------------------------------------------
# Filename utilities (normalize and avoid collisions)
# ---------------------------------------------------------------------

def normalize_trial_stem(trial_stem: str) -> str:
    """
    Convert e.g. 'walk01_01' -> 'walk01' by stripping a trailing '_<digits>'.

    Only removes a numeric suffix at the very end.
    Examples:
      - walk01_01 -> walk01
      - walk03_3  -> walk03
      - trial_A_01 -> trial_A
      - trial_A_B  -> trial_A_B  (unchanged)
    """
    return re.sub(r"_\d+$", "", trial_stem)


def make_unique_out_basename(save_dir: str, out_basename: str) -> str:
    """
    If <save_dir>/<out_basename>.trc already exists, append _dup2, _dup3, ...
    Returns a basename (no extension) that is safe to write.
    """
    candidate = out_basename
    k = 2
    while os.path.exists(os.path.join(save_dir, candidate + ".trc")):
        candidate = f"{out_basename}_dup{k}"
        k += 1
    return candidate


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
    """
    raw_names = header['markername']
    cleaned = [name for name in raw_names if name != '']
    return cleaned


def infer_marker_mapping(marker_names, user_mapping=None):
    """
    Choose the best marker mapping for this TRC.

    Priority:
      0) If user_mapping is provided, use it as-is
      1) offsetRemoved scheme
      2) legacy *_study scheme
      3) monocular scheme (fallback)
    """
    if user_mapping is not None:
        return user_mapping

    has_offset_removed = any('offsetremoved' in n.lower() for n in marker_names)
    if has_offset_removed:
        return OFFSETREMOVED_MARKER_MAPPING

    has_study_markers = any(n.lower().endswith('_study') or '_study' in n.lower() for n in marker_names)
    if has_study_markers:
        return DEFAULT_MARKER_MAPPING

    return MONOCULAR_MARKER_MAPPING


def rename_markers(marker_names, marker_mapping=None):
    """
    Rename markers according to a mapping dictionary.
    """
    marker_mapping = infer_marker_mapping(marker_names, marker_mapping)

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
    """
    marker_mapping = infer_marker_mapping(marker_names, marker_mapping)

    desired_markers = set(marker_mapping.keys())
    indices_to_keep = []
    filtered_names = []

    for i, marker in enumerate(marker_names):
        if marker in desired_markers:
            indices_to_keep.append(i)
            filtered_names.append(marker)

    col_indices = []
    for idx in indices_to_keep:
        col_indices.extend([idx * 3, idx * 3 + 1, idx * 3 + 2])

    filtered_data = marker_data[:, col_indices]

    return filtered_names, filtered_data


def write_trc_file(time, mrkdata, mrknames, directory, file, units="mm"):
    """
    Write a TRC file that OpenSim can read reliably.

    Fixes vs your current version:
      - Metadata keys/value counts always match (8 keys, 8 values), no extra empty columns.
      - PathFileType line uses tabs consistently.
      - Marker-name line uses TWO tabs between marker names (TRC convention for X/Y/Z triplets).
      - XYZ label line uses single tabs, no trailing tab.
      - Exactly one blank line after XYZ label line.
      - Time and marker rows are written with consistent tab separation and precision.
    """
    os.makedirs(directory, exist_ok=True)

    time = np.asarray(time, dtype=float).reshape(-1)
    if time.size == 0:
        raise ValueError("time is empty, cannot write TRC.")
    if time.size < 2:
        # OpenSim generally expects at least 2 frames
        time = np.append(time, time[0] + 0.01)

    mrkdata = np.asarray(mrkdata, dtype=float)
    if mrkdata.ndim != 2:
        raise ValueError("mrkdata must be a 2D array of shape (nFrames, 3*nMarkers).")

    mk, nk = mrkdata.shape
    if mk != time.size:
        raise ValueError(f"mrkdata rows ({mk}) must match time length ({time.size}).")
    if nk % 3 != 0:
        raise ValueError(f"mrkdata must have 3*nMarkers columns, got {nk}.")

    n_markers = nk // 3
    if len(mrknames) != n_markers:
        raise ValueError(
            f"mrknames length ({len(mrknames)}) must match n_markers ({n_markers})."
        )

    units = units.strip().lower()
    if units not in ("mm", "m"):
        raise ValueError("units must be 'mm' or 'm'")

    # Robust sampling rate estimate
    dt = np.diff(time)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]
    if dt.size == 0:
        f_rate = 100.0
    else:
        f_rate = 1.0 / float(np.median(dt))

    filepath = os.path.join(directory, f"{file}.trc")
    trc_basename = os.path.basename(filepath)

    try:
        with open(filepath, "w", newline="\n") as fid:
            # Line 1
            fid.write(f"PathFileType\t4\t(X/Y/Z)\t{trc_basename}\n")

            # Line 2 (8 keys)
            fid.write(
                "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n"
            )

            # Line 3 (8 values, SINGLE tabs, no empty fields)
            fid.write(
                f"{f_rate:.2f}\t{f_rate:.2f}\t{mk}\t{n_markers}\t{units}\t{f_rate:.2f}\t1\t{mk}\n"
            )

            # Line 4: marker names with two tabs between names
            fid.write("Frame#\tTime\t")
            fid.write("\t\t".join([str(m) for m in mrknames]))
            fid.write("\n")

            # Line 5: XYZ labels (single tabs)
            fid.write("\t\t")
            fid.write("\t".join([f"X{i}\tY{i}\tZ{i}" for i in range(1, n_markers + 1)]))
            fid.write("\n")

            # Line 6: blank line
            fid.write("\n")

            # Data lines
            for i in range(mk):
                fid.write(f"{i + 1}\t{time[i]:.8f}")
                # Write all coords, single-tab separated
                fid.write("".join(f"\t{val:.6f}" for val in mrkdata[i, :]))
                fid.write("\n")

        return True

    except Exception as e:
        print(f"Error writing TRC: {e}")
        return False



def save_unmodified_trc(
    trc_path,
    save_dir,
    rename_markers_on_save=True,
    filter_markers_on_save=True,
    marker_mapping=None,
):
    """
    Load a TRC and write it out unmodified in coordinates, but with
    the same naming convention as the optimized files.
    """
    os.makedirs(save_dir, exist_ok=True)

    header, data, header_lines = TRCload(trc_path)
    data = data[:, ~np.all(np.isnan(data), axis=0)]

    frame = data[:, 0]
    times = data[:, 1]
    marker_xyz = data[:, 2:]

    marker_names = extract_marker_names(header)
    marker_names = [n for n in marker_names if n not in ['Frame#', 'Time']]

    if filter_markers_on_save:
        marker_names, marker_xyz = filter_markers_by_mapping_keys(
            marker_names,
            marker_xyz,
            marker_mapping
        )

    if rename_markers_on_save:
        output_marker_names = rename_markers(marker_names, marker_mapping)
    else:
        output_marker_names = marker_names

    trial_stem_raw = os.path.splitext(os.path.basename(trc_path))[0]
    trial_stem_norm = normalize_trial_stem(trial_stem_raw)

    out_basename = f"MarkerData_optfeet_{trial_stem_norm}"
    out_basename = make_unique_out_basename(save_dir, out_basename)

    write_trc_file(
        time=times,
        mrkdata=marker_xyz,
        mrknames=output_marker_names,
        directory=save_dir,
        file=out_basename,
    )

    return os.path.join(save_dir, out_basename + '.trc')


# ---------------------------------------------------------------------
# Foot marker utilities
# ---------------------------------------------------------------------

def select_foot_marker_names(marker_names):
    """
    Select foot marker names with priority:
    1) *_study_offsetRemoved
    2) *_study
    3) monocular/base name

    Returns a list in the order:
    [L_toe_variant, L_calc_variant, r_toe_variant, r_calc_variant]
    """
    priority_map = {
        'L_toe':  ['L_toe_study_offsetRemoved', 'L_toe_study', 'L_toe', 'l_toe'],
        'L_calc': ['L_calc_study_offsetRemoved', 'L_calc_study', 'L_calc', 'l_calc'],
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
    """
    n_samples = marker_positions.shape[0]
    foot_marker_matrix = np.zeros((1, 4, 3, n_samples))

    for i, foot_marker in enumerate(foot_marker_names):
        marker_index = markernames.index(foot_marker)
        foot_marker_matrix[0, i, 0, :] = marker_positions[:, marker_index * 3]
        foot_marker_matrix[0, i, 1, :] = marker_positions[:, marker_index * 3 + 1]
        foot_marker_matrix[0, i, 2, :] = marker_positions[:, marker_index * 3 + 2]

    return foot_marker_matrix


def make_contact_mask(coordinates, mask_ips, mask_cont, toe_threshold, heel_threshold):
    """
    Build a contact mask from vertical coordinates.

    coordinates: shape (T, 4) for [L_toe, L_heel, R_toe, R_heel]
    """
    thresholds = []

    cond0 = coordinates[mask_cont == 1, 0]
    row0_min = np.nanmin(cond0)
    thr0 = row0_min + toe_threshold
    thresholds.append(thr0)

    cond1 = coordinates[mask_cont == 1, 1]
    row1_min = np.nanmin(cond1)
    thr1 = row1_min + heel_threshold
    thresholds.append(thr1)

    cond2 = coordinates[mask_ips == 1, 2]
    row2_min = np.nanmin(cond2)
    thr2 = row2_min + toe_threshold
    thresholds.append(thr2)

    cond3 = coordinates[mask_ips == 1, 3]
    row3_min = np.nanmin(cond3)
    thr3 = row3_min + heel_threshold
    thresholds.append(thr3)

    thresholds = np.array(thresholds)[None, :]

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
        """
        self.marker_positions = marker_positions
        self.marker_names = marker_names
        self.foot_marker_names = foot_marker_names
        self.foot_names = foot_marker_names
        self.n_frames = self.marker_positions.shape[0]
        self.contact = contact

        self.foot_name_to_index = {}
        for foot_marker in foot_marker_names:
            if foot_marker in marker_names:
                index = marker_names.index(foot_marker)
                self.foot_name_to_index[foot_marker] = index
            else:
                raise ValueError(f"Marker {foot_marker} not found in the marker names list.")

        self.device = device
        self.frame_rate = frame_rate
        self.iterations = 10000
        self.conv_tol = 1e-10
        self.loss_frequency_init = 1.0
        self.foot_position_loss_init = 1.0
        self.offset_deriv_loss_init = 1.0
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
            if index < reshaped_marker_positions.shape[1]:
                _ = reshaped_marker_positions[:, index, 2]
            else:
                print(f"Warning: Index {index} for {foot_marker} is out of bounds.")

        self.print_loss_terms = print_loss_terms

        self.design_vars = []
        self.offset = torch.zeros(
            (1, self.n_frames, 1, 3),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )
        self.design_vars.append(self.offset)

        self.feet_original = feet_original
        self.feet = self.feet_original

        if 'contact_position' in self.weights and self.weights['contact_position'] > 0:
            self.contact_mask = self.debounced_threshold(self.contact)
            padded_mask = torch.cat(
                [
                    torch.zeros(1, self.contact_mask.shape[1], dtype=torch.bool),
                    self.contact_mask,
                    torch.zeros(1, self.contact_mask.shape[1], dtype=torch.bool),
                ]
            )
            self.contact_starts = (padded_mask[:-1] == False) & (padded_mask[1:] == True)
            self.contact_ends = (padded_mask[:-1] == True) & (padded_mask[1:] == False)
            self.contact_position_loss_init = self.loss_contact_position().clone().detach()

        if 'contact_velocity' in self.weights and self.weights['contact_velocity'] > 0:
            self.contact_velocity_loss_init = self.loss_contact_velocity().clone().detach()

        if 'flat_floor' in self.weights and self.weights['flat_floor'] > 0:
            self.flat_floor_loss_init = self.loss_flat_floor().clone().detach()

        if 'offset_deriv' in self.weights and self.weights['offset_deriv'] > 0:
            self.offset_deriv_loss_init = self.loss_offset_deriv().clone().detach()

    def add_offset(self):
        self.feet = self.feet_original + self.offset
        return

    def loss_contact_position(self, scale=1):
        position_var_loss = 0
        key3d_feet = self.feet

        for n in range(len(key3d_feet)):
            start_indices = torch.where(self.contact_starts[:, n])[0]
            end_indices = torch.where(self.contact_ends[:, n])[0]
            variances = [
                torch.var(key3d_feet[:, start:end, n, :], axis=1).sum()
                for start, end in zip(start_indices, end_indices)
                if end > start
            ]
            position_var_loss += torch.sum(torch.stack(variances)) if variances else torch.tensor(0.0)

        return position_var_loss / scale

    def loss_contact_velocity(self, scale=1):
        key3d_feet = self.feet
        speed_feet = self.compute_speed(key3d_feet, self.frame_rate)
        contact_mask_expanded = self.contact.unsqueeze(0).unsqueeze(2)
        contact_loss = ((contact_mask_expanded * speed_feet) ** 2).sum()
        return contact_loss / scale

    def loss_offset_deriv(self, scale=1, diff_n=1):
        func_offset = self.offset.detach()
        func_offset = func_offset[:, :, :, 1]
        func_offset = func_offset.to(torch.float32)
        dt = 1.0 / self.frame_rate
        offset_diff = torch.diff(func_offset, dim=1, n=diff_n)
        if offset_diff.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        offset_velocity = offset_diff / dt**diff_n
        offset_velocity = torch.cat([offset_velocity[:, 0:1, :], offset_velocity], dim=1)
        average_velocity = torch.norm(offset_velocity, dim=-1, keepdim=True).sum()
        average_velocity = average_velocity.detach().to(torch.float64)
        return average_velocity

    def debounced_threshold(self, v_mask, high_thresh=.5, low_thresh=.5, min_stretch_len=3):
        """
        Apply a debounced threshold to a TxN matrix.
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
        key_3d: (B, T, N, 3)
        """
        if isinstance(key_3d, np.ndarray):
            key_3d = torch.tensor(key_3d, dtype=torch.float32)

        key_3d = key_3d.to(torch.float32)
        dt = 1.0 / frame_rate
        position_diff = torch.diff(key_3d, dim=1, n=diff_n)
        velocity = position_diff / dt**diff_n
        velocity = torch.cat([velocity[:, 0:1, :], velocity], dim=1)
        average_velocity = torch.norm(velocity, dim=-1, keepdim=True)
        return average_velocity

    def loss_flat_floor(self, scale=1):
        key3d_feet_y = self.feet[:, :, :, 1].squeeze()
        masked_feet_y = key3d_feet_y[self.contact_mask]
        loss = torch.var(masked_feet_y)
        return loss / scale

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
        optimizer = torch.optim.LBFGS(
            self.design_vars,
            lr=2,
            tolerance_change=self.conv_tol,
            line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            self.add_offset()
            self.loss = self.objective_function()
            self.loss.backward()
            return self.loss

        objective_values = torch.zeros(self.iterations)
        for i in range(self.iterations):
            optimizer.step(closure)
            objective_values[i] = self.loss.clone().detach().cpu()
            self.last_loss = self.loss.clone().detach().cpu()

        output = {
            'offset': self.offset.detach(),
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

    Notes on naming:
      - trial_stem_raw is used for gait processing lookups (process_gait_data)
      - trial_stem_norm is used for output file naming (deduplicated / normalized)
    """
    os.makedirs(save_dir, exist_ok=True)

    trc_basename = os.path.basename(trc_path)
    trial_stem_raw = os.path.splitext(trc_basename)[0]
    trial_stem_norm = normalize_trial_stem(trial_stem_raw)

    # --------------------------------------------------------------
    # 1) Iterative gait detection with small trimming increments
    # --------------------------------------------------------------
    step = 0.1        # seconds added on each side per attempt
    max_extra = 0.5   # max extra trimming on each side
    extra_values = np.arange(0.0, max_extra + 1e-9, step)

    gait_ok = False
    last_error = None
    used_trim_start = trimming_start
    used_trim_end = trimming_end

    for extra in extra_values:
        ts = trimming_start + extra
        te = trimming_end + extra
        try:
            mask_ips, mask_cont, gait_events, foot_positions, time = process_gait_data(
                session_dir=session_dir,
                trial_name=trial_stem_raw,  # keep RAW here
                leg=side,
                lowpass_cutoff_frequency=lowpass_cutoff_frequency,
                n_gait_cycles=n_gait_cycles,
                gait_style=gait_style,
                trimming_start=ts,
                trimming_end=te,
            )
            used_trim_start = ts
            used_trim_end = te
            gait_ok = True
            if extra > 0:
                print(
                    f"  Gait events fixed for {trial_stem_raw} "
                    f"with trimming_start={used_trim_start:.2f}s, "
                    f"trimming_end={used_trim_end:.2f}s"
                )
            break

        except (ValueError, IndexError) as e:
            msg = str(e)
            last_error = e

            if isinstance(e, ValueError) and "The ordering of gait events is not correct" in msg:
                print(
                    f"  Gait events not in correct order for {trial_stem_raw} "
                    f"with trimming_start={ts:.2f}s, trimming_end={te:.2f}s. "
                    "Trying a bit more trimming."
                )
                continue

            if isinstance(e, IndexError):
                print(
                    f"  Gait processing IndexError for {trial_stem_raw} "
                    f"with trimming_start={ts:.2f}s, trimming_end={te:.2f}s: {e}. "
                    "Trying a bit more trimming."
                )
                continue

            raise

    if not gait_ok:
        print(
            f"  Failed to obtain valid gait events for {trial_stem_raw} "
            "even after iterative trimming. Writing unmodified TRC."
        )
        header, data, header_lines = TRCload(trc_path)
        data = data[:, ~np.all(np.isnan(data), axis=0)]

        frame = data[:, 0]
        times = data[:, 1]
        marker_xyz = data[:, 2:]

        marker_names = extract_marker_names(header)
        marker_names = [n for n in marker_names if n not in ['Frame#', 'Time']]

        if filter_markers_on_save:
            marker_names, marker_xyz = filter_markers_by_mapping_keys(
                marker_names,
                marker_xyz,
                marker_mapping
            )

        if rename_markers_on_save:
            output_marker_names = rename_markers(marker_names, marker_mapping)
        else:
            output_marker_names = marker_names

        out_basename = f"MarkerData_optfeet_{trial_stem_norm}"
        out_basename = make_unique_out_basename(save_dir, out_basename)

        write_trc_file(
            time=times,
            mrkdata=marker_xyz,
            mrknames=output_marker_names,
            directory=save_dir,
            file=out_basename,
        )
        return os.path.join(save_dir, out_basename + '.trc')

    # --------------------------------------------------------------
    # 2) Load TRC and apply the same final trimming
    # --------------------------------------------------------------
    header, data, header_lines = TRCload(trc_path)
    data = data[:, ~np.all(np.isnan(data), axis=0)]

    frame = data[:, 0]
    times = data[:, 1]
    marker_xyz = data[:, 2:]

    t = np.round(times, 6)
    duration = t[-1] - t[0]

    if used_trim_start < 0 or used_trim_end < 0:
        raise ValueError(
            f"trimming_start and trimming_end must be >= 0, got "
            f"{used_trim_start}, {used_trim_end}"
        )

    if used_trim_start + used_trim_end >= duration:
        raise ValueError(
            f"Requested trimming_start {used_trim_start}s and trimming_end {used_trim_end}s "
            f"remove >= full duration {duration:.4f}s"
        )

    if used_trim_start > 0:
        t_start = t[0] + used_trim_start
    else:
        t_start = t[0]

    if used_trim_end > 0:
        t_end = t[-1] - used_trim_end
    else:
        t_end = t[-1]

    if t_start >= t_end:
        raise ValueError(
            f"After applying trimming_start={used_trim_start} and trimming_end={used_trim_end}, "
            f"t_start={t_start:.6f} >= t_end={t_end:.6f}"
        )

    idx_start = np.where(t >= np.round(t_start, 6))[0][0]
    idx_end = np.where(t <= np.round(t_end, 6))[0][-1] + 1

    if idx_end <= idx_start:
        raise ValueError(
            f"Invalid trimming: idx_start={idx_start}, idx_end={idx_end}. "
            f"t_start={t_start:.6f}, t_end={t_end:.6f}"
        )

    times = times[idx_start:idx_end]
    marker_xyz = marker_xyz[idx_start:idx_end, :]
    frame = frame[idx_start:idx_end]

    marker_names = extract_marker_names(header)
    marker_names = [n for n in marker_names if n not in ['Frame#', 'Time']]

    if filter_markers_on_save:
        marker_names, marker_xyz = filter_markers_by_mapping_keys(
            marker_names,
            marker_xyz,
            marker_mapping
        )

    n_frames = marker_xyz.shape[0]

    # Clip masks to match the trimmed TRC frame count.  The gait-analysis
    # trimming and TRC trimming can differ by one frame due to floating-point
    # rounding on the time boundaries.
    mask_ips  = mask_ips[:n_frames]
    mask_cont = mask_cont[:n_frames]

    if n_frames < 10:
        print(f"  Too few frames ({n_frames}) in {trc_path}, skipping foot optimization.")
        if rename_markers_on_save:
            output_marker_names = rename_markers(marker_names, marker_mapping)
        else:
            output_marker_names = marker_names

        out_basename = f"MarkerData_optfeet_{trial_stem_norm}"
        out_basename = make_unique_out_basename(save_dir, out_basename)

        write_trc_file(
            time=times,
            mrkdata=marker_xyz,
            mrknames=output_marker_names,
            directory=save_dir,
            file=out_basename,
        )
        return os.path.join(save_dir, out_basename + '.trc')

    if foot_marker_names is None:
        foot_marker_names = select_foot_marker_names(marker_names)

    foot_matrix = create_foot_marker_matrix(foot_marker_names, marker_names, marker_xyz)
    y_coords = foot_matrix[0, :, 1, :].T  # (T,4)
    contact_mask_np = make_contact_mask(y_coords, mask_ips, mask_cont, toe_threshold, heel_threshold)

    contact_mask = torch.tensor(contact_mask_np)
    feet_tensor = torch.tensor(foot_matrix).permute(0, 3, 1, 2)

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

    try:
        output = optimizer.optimize()
    except IndexError as e:
        print(f"  FootPositionOptimizer failed for {trc_path} with IndexError: {e}")
        print("  Skipping optimization and using original marker positions.")
        if rename_markers_on_save:
            output_marker_names = rename_markers(marker_names, marker_mapping)
        else:
            output_marker_names = marker_names

        out_basename = f"MarkerData_optfeet_{trial_stem_norm}"
        out_basename = make_unique_out_basename(save_dir, out_basename)

        write_trc_file(
            time=times,
            mrkdata=marker_xyz,
            mrknames=output_marker_names,
            directory=save_dir,
            file=out_basename,
        )
        return os.path.join(save_dir, out_basename + '.trc')

    offset = output['offset'].detach().cpu().numpy()   # (1,T,1,3)
    offset = offset.reshape(-1, 3)                     # (T,3)
    offset_y = offset[:, 1]

    filt_freq = 6.0
    nyquist = 0.5 * (1.0 / np.mean(np.diff(times)))
    wn = filt_freq / nyquist
    sos = butter(2, wn, btype='low', output='sos')
    try:
        offset_y_filt = sosfiltfilt(sos, offset_y, padlen=50)
    except ValueError as e:
        print(f"  Warning: {e}. Retrying with a shorter pad length.")
        n = len(offset_y)
        if n > 3:
            new_padlen = max(1, n - 1)
            print(f"  Using padlen={new_padlen} for {trc_path}")
            offset_y_filt = sosfiltfilt(sos, offset_y, padlen=new_padlen)
        else:
            print(f"  Signal too short to filter for {trc_path}. Copying original values.")
            offset_y_filt = offset_y.copy()

    offset_y_filt = offset_y_filt[:, None]
    updated_marker_xyz = marker_xyz.copy()
    updated_marker_xyz[:, 1::3] += offset_y_filt

    if rename_markers_on_save:
        output_marker_names = rename_markers(marker_names, marker_mapping)
    else:
        output_marker_names = marker_names

    out_basename = f"MarkerData_optfeet_{trial_stem_norm}"
    out_basename = make_unique_out_basename(save_dir, out_basename)

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
    """
    with open(trc_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                data_rate = float(parts[0])
                camera_rate = float(parts[1])
                return camera_rate
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
    do_not_refine=False,
):
    """
    Run kinematics refinement for better foot contact.

    If do_not_refine is True (or 1), TRCs are simply copied to the
    ForGaitDynamics folder with the MarkerData_optfeet_<trial>.trc
    naming convention, without any gait processing or optimization.
    """
    session_folder = Path(session_folder)

    marker_trc_dir = session_folder / "MarkerData"
    refined_trc_dir = session_folder / "ForGaitDynamics"
    refined_trc_dir.mkdir(parents=True, exist_ok=True)

    for fname in os.listdir(marker_trc_dir):
        if (
            fname.endswith(".trc")
            and "optimized" not in fname.lower()
            and (not trial_prefix or fname.startswith(trial_prefix))
        ):
            trc_path = marker_trc_dir / fname

            if do_not_refine:
                print(f"  Skipping refinement for {fname}, writing unmodified TRC.")
                save_unmodified_trc(
                    trc_path=str(trc_path),
                    save_dir=str(refined_trc_dir),
                    rename_markers_on_save=rename_markers_on_save,
                    filter_markers_on_save=filter_markers_on_save,
                    marker_mapping=marker_mapping,
                )
                continue

            print(f"  Optimizing foot positions for {fname}")
            try:
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
            except (ValueError, IndexError) as e:
                print(f"  Skipping {fname} due to gait processing error: {e}")
                continue
