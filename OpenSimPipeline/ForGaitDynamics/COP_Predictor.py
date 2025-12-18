# cop_predictor.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.interpolate import interp1d, PchipInterpolator



def movmean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.size < 2:
        return x.copy()
    s = pd.Series(x)
    return s.rolling(window=window, min_periods=1, center=True).mean().to_numpy()

class MLP_AR(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def TRCload(filename):
    """
    Minimal TRC loader that matches what you used when training.
    If you prefer, copy your full TRCload implementation from COP_autoregressor.py here.
    """
    import re

    with open(filename, "r") as f:
        lines = f.readlines()

    header_idx = None
    for i, ln in enumerate(lines):
        if "Frame#" in ln and "Time" in ln:
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("Could not find TRC header row.")

    header_line = lines[header_idx].strip()
    tokens = re.split(r"\s+|\t+", header_line)
    marker_names = tokens[2:]

    # data start = header_idx + 2
    data_start = header_idx + 2
    df = pd.read_csv(
        filename,
        sep=r"\s+|\t+",
        engine="python",
        header=None,
        skiprows=data_start,
    )
    data = df.to_numpy()

    class Header:
        pass
    header = Header()
    header.markername = marker_names
    dataArray = None
    return header, data, dataArray

def load_mot_to_struct(path):
    """Parse .mot file into dict of numpy arrays."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    start_idx = 0
    for i, ln in enumerate(lines):
        if ln.strip().lower() == "endheader":
            start_idx = i + 1
            break

    df = pd.read_csv(
        path,
        sep=r"\s+|\t+",
        engine="python",
        header=0,
        skiprows=start_idx,
    )
    G = {col: df[col].to_numpy() for col in df.columns}
    if "time" not in G:
        G["time"] = df.iloc[:, 0].to_numpy()
    return G

def safe_interp1(x, y, xi):
    x = np.asarray(x).ravel()
    y = np.asarray(y)
    xi = np.asarray(xi).ravel()
    if y.ndim == 1:
        f = interp1d(x, y, kind="linear", fill_value="extrapolate", bounds_error=False)
        return f(xi)
    yi = np.empty((xi.size, y.shape[1]))
    for j in range(y.shape[1]):
        f = interp1d(x, y[:, j], kind="linear", fill_value="extrapolate", bounds_error=False)
        yi[:, j] = f(xi)
    return yi

def mlp_ar_rollout_sequence(model, X_std_step, ar_window, seed_value):
    model.eval()
    N, F = X_std_step.shape
    yhat = np.zeros(N, dtype=np.float32)
    hist = np.full(ar_window, float(seed_value), dtype=np.float32)
    with torch.no_grad():
        for i in range(N):
            x_i = X_std_step[i, :].astype(np.float32)
            x_full = np.concatenate([x_i, hist], axis=0)
            x_tensor = torch.from_numpy(x_full[None, :])
            y_i = model(x_tensor).cpu().numpy()[0].item()
            yhat[i] = y_i
            hist = np.roll(hist, -1)
            hist[-1] = y_i
    return yhat

def find_complete_stance(mask_in):
    mask_in = np.asarray(mask_in).astype(bool).ravel()
    d = np.diff(np.r_[False, mask_in, False].astype(int))
    run_starts = np.where(d == 1)[0]
    run_ends   = np.where(d == -1)[0] - 1
    lengths = run_ends - run_starts + 1
    idx = np.argmax(lengths) if lengths.size else 0
    longest = np.zeros_like(mask_in, dtype=bool)
    if lengths.size:
        longest[run_starts[idx]:run_ends[idx] + 1] = True
    return longest

def interpolate_cop_with_holdout(cop_array):
    cop = np.array(cop_array).copy()
    valid = np.isfinite(cop)
    if not np.any(valid):
        return cop
    indices = np.arange(len(cop))
    valid_indices = indices[valid]
    valid_values = cop[valid]
    cop_interp = np.interp(indices, valid_indices, valid_values)
    first_valid_idx = valid_indices[0]
    last_valid_idx = valid_indices[-1]
    if first_valid_idx > 0:
        cop_interp[:first_valid_idx] = valid_values[0]
    if last_valid_idx < len(cop) - 1:
        cop_interp[last_valid_idx + 1 :] = valid_values[-1]
    return cop_interp

def write_grf_mot_with_new_cop(original_grf_path, output_path, COP_pred_R_m, COP_pred_L_m, t_GRF):
    G = load_mot_to_struct(original_grf_path)
    numFrames = len(t_GRF)
    with open(output_path, "w") as out_file:
        out_file.write("nColumns=19\n")
        out_file.write(f"nRows={numFrames}\n")
        out_file.write("DataType=double\n")
        out_file.write("version=3\n")
        out_file.write("OpenSimVersion=4.1\n")
        out_file.write("endheader\n")
        out_file.write("time")
        for prefix in ["R", "L"]:
            out_file.write(f"\t{prefix}_ground_force_vx")
            out_file.write(f"\t{prefix}_ground_force_vy")
            out_file.write(f"\t{prefix}_ground_force_vz")
            out_file.write(f"\t{prefix}_ground_force_px")
            out_file.write(f"\t{prefix}_ground_force_py")
            out_file.write(f"\t{prefix}_ground_force_pz")
            out_file.write(f"\t{prefix}_ground_torque_x")
            out_file.write(f"\t{prefix}_ground_torque_y")
            out_file.write(f"\t{prefix}_ground_torque_z")
        out_file.write("\n")
        for i in range(numFrames):
            out_file.write(f"{t_GRF[i]:.5f}")
            out_file.write(f"\t{G['R_ground_force_vx'][i]}")
            out_file.write(f"\t{G['R_ground_force_vy'][i]}")
            out_file.write(f"\t{G['R_ground_force_vz'][i]}")
            out_file.write(f"\t{COP_pred_R_m[i]}")
            out_file.write(f"\t{G['R_ground_force_py'][i]}")
            out_file.write(f"\t{G['R_ground_force_pz'][i]}")
            out_file.write(f"\t{G.get('R_ground_torque_x', [0]*numFrames)[i]}")
            out_file.write(f"\t{G.get('R_ground_torque_y', [0]*numFrames)[i]}")
            out_file.write(f"\t{G.get('R_ground_torque_z', [0]*numFrames)[i]}")
            out_file.write(f"\t{G['L_ground_force_vx'][i]}")
            out_file.write(f"\t{G['L_ground_force_vy'][i]}")
            out_file.write(f"\t{G['L_ground_force_vz'][i]}")
            out_file.write(f"\t{COP_pred_L_m[i]}")
            out_file.write(f"\t{G['L_ground_force_py'][i]}")
            out_file.write(f"\t{G['L_ground_force_pz'][i]}")
            out_file.write(f"\t{G.get('L_ground_torque_x', [0]*numFrames)[i]}")
            out_file.write(f"\t{G.get('L_ground_torque_y', [0]*numFrames)[i]}")
            out_file.write(f"\t{G.get('L_ground_torque_z', [0]*numFrames)[i]}")
            out_file.write("\n")
  #  print(f"Wrote new GRF file to: {output_path}")


# Load trained model from artifacts


def load_cop_model(artifact_path):
    ckpt = torch.load(artifact_path, map_location="cpu")
    model_type = ckpt["model_type"]
    if model_type.lower() != "mlp_ar":
        raise ValueError(f"Expected mlp_ar model, got {model_type}")
    mu_X = ckpt["mu_X"]
    std_X = ckpt["std_X"]
    AR_WINDOW = int(ckpt["AR_WINDOW"])
    MLP_HIDDEN = int(ckpt["MLP_HIDDEN"])
    HEEL_X_AVG = ckpt["HEEL_X_AVG"]
    HEEL_Y_AVG = ckpt["HEEL_Y_AVG"]
    in_dim = mu_X.size + AR_WINDOW
    model = MLP_AR(in_dim, MLP_HIDDEN)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, mu_X, std_X, AR_WINDOW, HEEL_X_AVG, HEEL_Y_AVG


# 3. Standalone COP prediction for a single trial


def detect_stance_from_GRF(Fy, threshold_N=10.0):
    Fy = np.asarray(Fy)
    return Fy > threshold_N

def compute_stance_percent_from_mask(mask):
    """
    Assign 0–100 % stance for every contiguous run of True in the mask.
    Each stance bout is normalized independently.
    """
    mask = np.asarray(mask, bool)
    N = mask.size
    sp = np.full(N, np.nan)

    # find all runs of True
    d = np.diff(np.r_[False, mask, False].astype(int))
    run_starts = np.where(d == 1)[0]
    run_ends   = np.where(d == -1)[0] - 1

    for s, e in zip(run_starts, run_ends):
        length = e - s + 1
        if length >= 2:
            sp[s:e+1] = np.linspace(0.0, 100.0, length)
        else:
            sp[s:e+1] = 0.0  # degenerate 1-frame stance

    return sp


def predict_cop_for_trial(
    grf_path,
    trc_path,
    ik_path = None,
    trial_name=str,
    artifact_path=str,
    output_grf_path=None,
    grf_threshold_N=10.0,
):

    model, mu_X, std_X, AR_WINDOW, HEEL_X_AVG, HEEL_Y_AVG = load_cop_model(artifact_path)

    G = load_mot_to_struct(grf_path)
    t_GRF = G["time"].ravel()
    Nt = t_GRF.size
    Fy_R = np.asarray(G["R_ground_force_vy"])
    Fy_L = np.asarray(G["L_ground_force_vy"])

    OC_header, OC_mrkdata, _ = TRCload(trc_path)
    t_trc = OC_mrkdata[:, 1]
    OC_numeric = OC_mrkdata[:, 2:] * 1000.0
    OC_mrknames = [m for m in OC_header.markername if m]

    def mrk_block(name):
        idx = OC_mrknames.index(name)
        c0 = idx * 3
        return OC_numeric[:, c0 : c0 + 3]

    heel_R = mrk_block("r_calc")
    toe_R  = mrk_block("r_toe")
    heel_L = mrk_block("L_calc")
    toe_L  = mrk_block("L_toe")

    maskR = detect_stance_from_GRF(Fy_R, threshold_N=grf_threshold_N)
    maskL = detect_stance_from_GRF(Fy_L, threshold_N=grf_threshold_N)

    stance_percents_r = compute_stance_percent_from_mask(maskR)
    stance_percents_l = compute_stance_percent_from_mask(maskL)

    COP_pred_R = np.full(Nt, np.nan)
    COP_pred_L = np.full(Nt, np.nan)

    winsR = np.argwhere(np.diff(np.r_[False, maskR, False]) != 0).reshape(-1, 2)
    winsR[:, 1] -= 1
    winsL = np.argwhere(np.diff(np.r_[False, maskL, False]) != 0).reshape(-1, 2)
    winsL[:, 1] -= 1

    def process_leg(
        wins,
        Fy,
        heelM,
        toeM,
        stanceP,
        HEEL_X_AVG,
        HEEL_Y_AVG,
        is_right=True,
    ):
        # Unwrap markers if passed as (array,) from TRCload
        if isinstance(heelM, tuple):
            heelM = heelM[0]
        if isinstance(toeM, tuple):
            toeM = toeM[0]
    
        COP_out = np.full_like(t_GRF, np.nan, dtype=float)
    
        for i1, i2 in wins:
            tt = t_GRF[i1 : i2 + 1]
            if i2 < i1 or tt.size < 3:
                continue
    
            # Stance percent for this window
            stance_win = stanceP[i1 : i2 + 1]
            valid_stance = np.isfinite(stance_win)
            if np.count_nonzero(valid_stance) < 2:
                # nothing usable in this window, skip it
                continue
            stance_phase = stance_win / 100.0
    
            # Markers, interpolated to GRF time
            heel_i = safe_interp1(t_trc, heelM, tt) / 1_000_000.0
            toe_i  = safe_interp1(t_trc, toeM,  tt) / 1_000_000.0
    
            # Reference heel trajectory
            ref_stance = np.linspace(0, 1, 100)
            f_hx = interp1d(ref_stance, np.asarray(HEEL_X_AVG), kind="linear",
                            fill_value="extrapolate", bounds_error=False)
            f_hy = interp1d(ref_stance, np.asarray(HEEL_Y_AVG), kind="linear",
                            fill_value="extrapolate", bounds_error=False)
            HEEL_refX = f_hx(stance_phase)
            HEEL_refY = f_hy(stance_phase)
    
            shiftX = np.nanmean(heel_i[:, 0] - HEEL_refX)
            shiftY = np.nanmean(heel_i[:, 1] - HEEL_refY)
    
            heel_f = heel_i.copy()
            toe_f  = toe_i.copy()
            heel_f[:, 0] -= shiftX
            heel_f[:, 1] -= shiftY
            toe_f[:, 0]  -= shiftX
            toe_f[:, 1]  -= shiftY
    
            fake_hs = heel_f[0, :].copy()
            fake_hs[0] = np.asarray(HEEL_X_AVG)[0]
            fake_hs[1] = np.asarray(HEEL_Y_AVG)[0]
    
            heel2 = heel_f - fake_hs[None, :]
            toe2  = toe_f  - fake_hs[None, :]
            v = toe2 - heel2
            theta = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
    
            heel_s = heel_f[:, 0]
            toe_s  = toe_f[:, 0]
            midfoot_s = 0.5 * (heel_s + toe_s)
            N = tt.size
            foot_length = np.nanmean(toe_s - heel_s)
            foot_length_vec = np.full(N, foot_length)
    
            Fy_win = Fy[i1 : i2 + 1]
            max_abs = np.max(np.abs(Fy_win)) if np.any(np.isfinite(Fy_win)) else 0.0
            Fy_norm = (Fy_win / max_abs) if max_abs > 0 else np.zeros_like(Fy_win)
    
            grf_vel = np.r_[0.0, np.diff(Fy_norm)]
            heel_vel = np.r_[0.0, np.diff(heel_s)]
            toe_vel  = np.r_[0.0, np.diff(toe_s)]
            midfoot_vel = np.r_[0.0, np.diff(midfoot_s)]
    
            X_step = np.column_stack([
                stance_phase,
                Fy_norm,
                grf_vel,
                foot_length_vec,
                midfoot_s,
                heel_s,
                toe_s,
                heel_vel,
                toe_vel,
                midfoot_vel,
                theta,
            ])
            X_step[~np.isfinite(X_step)] = 0.0
    
            mu = np.asarray(mu_X).ravel()
            sd = np.asarray(std_X).ravel()
            if X_step.shape[1] > mu.size:
                X_step = X_step[:, :mu.size]
            elif X_step.shape[1] < mu.size:
                X_step = np.column_stack([X_step, np.zeros((N, mu.size - X_step.shape[1]))])
    
            X_std = (X_step - mu) / sd
            seed_val = float(heel_s[0])
            yhat = mlp_ar_rollout_sequence(model, X_std, AR_WINDOW, seed_val)
    
            x = stance_phase
            valid = np.isfinite(x)
            x_valid = x[valid]
            yhat_valid = yhat[valid]
    
            # Final guard before PCHIP
            if x_valid.size < 2:
                continue
    
            x_jit = x_valid + 1e-12 * np.arange(x_valid.size)
            pp = PchipInterpolator(x_jit, yhat_valid)
            yhat_smooth = pp(x_jit)
            yhat_corr = yhat_smooth + shiftX
            yhat_smooth2 = movmean(yhat_corr, 5)
    
            # Write into output
            COP_out[i1 : i2 + 1] = yhat_smooth2
    
        return COP_out


    COP_pred_R = process_leg(winsR, Fy_R, heel_R, toe_R, stance_percents_r, HEEL_X_AVG, HEEL_Y_AVG, True)
    COP_pred_L = process_leg(winsL, Fy_L, heel_L, toe_L, stance_percents_l, HEEL_X_AVG, HEEL_Y_AVG, False)

    COP_pred_R_m = interpolate_cop_with_holdout(COP_pred_R)
    COP_pred_L_m = interpolate_cop_with_holdout(COP_pred_L)

    if output_grf_path is None:
        force_dir = os.path.dirname(grf_path)
        output_grf_path = os.path.join(force_dir, f"{trial_name}_Optimized_forces.mot")

    write_grf_mot_with_new_cop(grf_path, output_grf_path, COP_pred_R_m, COP_pred_L_m, t_GRF)
    
    if os.path.exists(grf_path):
        os.remove(grf_path)



  

