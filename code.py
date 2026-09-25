import streamlit as st
import ezc3d
import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import tempfile
from math import sqrt
import seaborn as sns

st.set_page_config(page_title="Comparateur Scores de marche - Faps, eFaps, eGVI, GDI, GPS", layout="wide")
st.title("🦿 Comparateur de Scores de marche (Session 1 vs Session 2)")

# ==========================================
# FONCTIONS DE CALCUL (Refactorisées)
# ==========================================

def calculate_faps(trials, static_file, ambulatory_aids, assistive_device):
    try:
        statique = ezc3d.c3d(static_file)
        labelsStat = statique['parameters']['POINT']['LABELS']['value']

        def get_static_point(label):
            if label not in labelsStat: return None
            idx = labelsStat.index(label)
            pt = statique['data']['points'][:3, idx, :]
            mask = ~np.isnan(pt[0, :])
            return pt[:, mask][:, 0] if np.any(mask) else None

        p1 = get_static_point('LPSI')
        p2 = get_static_point('LANK')
        if p1 is None or p2 is None:
            return None

        leg_length = np.linalg.norm(p1 - p2) / 1000.0 
    except Exception:
        return None

    results = {'sl_r': [], 'sl_l': [], 'st_r': [], 'st_l': [], 'dbs': []}

    for trial_path in trials:
        try:
            acq = ezc3d.c3d(trial_path)
            labels = acq['parameters']['POINT']['LABELS']['value']
            freq = acq['header']['points']['frame_rate']
            data = acq['data']['points']

            def get_clean_marker(label):
                if label not in labels: return None
                idx = labels.index(label)
                m = data[:3, idx, :].copy()
                for i in range(3):
                    mask = np.isnan(m[i, :])
                    if np.any(mask) and not np.all(mask):
                        m[i, mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), m[i, ~mask])
                return m

            r_he = get_clean_marker('RHEE')
            l_he = get_clean_marker('LHEE')
            if r_he is None or l_he is None: continue

            hs_r, _ = find_peaks(-r_he[2, :], distance=int(freq*0.4), prominence=2)
            hs_l, _ = find_peaks(-l_he[2, :], distance=int(freq*0.4), prominence=2)

            for t0 in hs_l:
                next_r = hs_r[hs_r > t0]
                if len(next_r) > 0:
                    t1 = next_r[0]
                    results['st_r'].append((t1 - t0) / freq)
                    results['sl_r'].append(np.abs(r_he[0, t1] - l_he[0, t0]) / 1000.0)

            for t0 in hs_r:
                next_l = hs_l[hs_l > t0]
                if len(next_l) > 0:
                    t1 = next_l[0]
                    results['st_l'].append((t1 - t0) / freq)
                    results['sl_l'].append(np.abs(l_he[0, t1] - r_he[0, t0]) / 1000.0)

            results['dbs'].append(np.abs(np.mean(r_he[1, :]) - np.mean(l_he[1, :])) / 10.0)
        except Exception:
            continue

    if not results['sl_r'] or not results['sl_l']:
        return None

    avg_sl_r = np.mean(results['sl_r'])
    avg_sl_l = np.mean(results['sl_l'])
    avg_st_r = np.mean(results['st_r'])
    avg_st_l = np.mean(results['st_l'])
    avg_dbs = np.mean(results['dbs'])

    gsl_r = avg_sl_r / leg_length
    gsl_l = avg_sl_l / leg_length
    gv_r = gsl_r / avg_st_r
    gv_l = gsl_l / avg_st_l

    def get_step_function_penalty(gv_val, gsl_val, st_val):
        p_v = 0 if 1.1 <= gv_val <= 1.5 else (min(abs(gv_val - 1.1), abs(gv_val - 1.5)) / 0.4) * 7.33
        p_sl = 0 if 0.69 <= gsl_val <= 0.86 else (min(abs(gsl_val - 0.69), abs(gsl_val - 0.86)) / 0.2) * 7.33
        p_st = 0 if 0.50 <= st_val <= 0.63 else (min(abs(st_val - 0.50), abs(st_val - 0.63)) / 0.2) * 7.33
        return min(p_v + p_sl + p_st, 22)

    deduction_A = get_step_function_penalty(gv_l, gsl_l, avg_st_l)
    deduction_B = get_step_function_penalty(gv_r, gsl_r, avg_st_r)

    diff_asy = np.abs(gsl_r - gsl_l)
    deduction_C = 0 if diff_asy < 0.03 else min(((diff_asy - 0.03) / 0.15) * 8, 8)

    if 5 <= avg_dbs <= 10:
        deduction_D = 0
    else:
        dbs_diff = min(abs(avg_dbs - 5), abs(avg_dbs - 10))
        deduction_D = min((dbs_diff / 8) * 8, 8) 

    deduction_E = 5 if ambulatory_aids > 0 else 0
    deduction_F = 5 if assistive_device > 0 else 0

    total_deductions = deduction_A + deduction_B + deduction_C + deduction_D + deduction_E + deduction_F
    score_faps = 100 - total_deductions
    score_min = 30 if (ambulatory_aids>0 or assistive_device>0) else 40
    
    return max(score_min, score_faps)


def calculate_efaps(trials, static_file, ambulatory_aids, assistive_device):
    try:
        statique = ezc3d.c3d(static_file)
        labelsStat = statique['parameters']['POINT']['LABELS']['value']

        def get_static_point(label):
            if label not in labelsStat: return None
            idx = labelsStat.index(label)
            pt = statique['data']['points'][:3, idx, :]
            mask = ~np.isnan(pt[0, :])
            return pt[:, mask][:, 0] if np.any(mask) else None

        p1 = get_static_point('LPSI')
        p2 = get_static_point('LANK')
        if p1 is None or p2 is None: return None
        leg_length = np.linalg.norm(p1 - p2) / 1000.0 
    except Exception:
        return None

    results = {'sl_r': [], 'sl_l': [], 'st_r': [], 'st_l': [], 'dbs': []}

    for trial_path in trials:
        try:
            acq = ezc3d.c3d(trial_path)
            labels = acq['parameters']['POINT']['LABELS']['value']
            freq = acq['header']['points']['frame_rate']
            data = acq['data']['points']

            def get_clean_marker(label):
                if label not in labels: return None
                idx = labels.index(label)
                m = data[:3, idx, :].copy()
                for i in range(3):
                    mask = np.isnan(m[i, :])
                    if np.any(mask) and not np.all(mask):
                        m[i, mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), m[i, ~mask])
                return m

            r_he = get_clean_marker('RHEE')
            l_he = get_clean_marker('LHEE')
            if r_he is None or l_he is None: continue

            hs_r, _ = find_peaks(-r_he[2, :], distance=int(freq*0.4), prominence=2)
            hs_l, _ = find_peaks(-l_he[2, :], distance=int(freq*0.4), prominence=2)

            for t0 in hs_l:
                next_r = hs_r[hs_r > t0]
                if len(next_r) > 0:
                    t1 = next_r[0]
                    results['st_r'].append((t1 - t0) / freq)
                    results['sl_r'].append(np.abs(r_he[0, t1] - l_he[0, t0]) / 1000.0)

            for t0 in hs_r:
                next_l = hs_l[hs_l > t0]
                if len(next_l) > 0:
                    t1 = next_l[0]
                    results['st_l'].append((t1 - t0) / freq)
                    results['sl_l'].append(np.abs(l_he[0, t1] - r_he[0, t0]) / 1000.0)

            results['dbs'].append(np.abs(np.mean(r_he[1, :]) - np.mean(l_he[1, :])) / 10.0)
        except Exception:
            continue

    if not results['sl_r'] or not results['sl_l']: return None

    avg_sl_r = np.mean(results['sl_r'])
    avg_sl_l = np.mean(results['sl_l'])
    avg_st_r = np.mean(results['st_r'])
    avg_st_l = np.mean(results['st_l'])
    avg_dbs = np.mean(results['dbs'])

    gsl_r = avg_sl_r / leg_length
    gsl_l = avg_sl_l / leg_length
    v_r = avg_sl_r / avg_st_r
    v_l = avg_sl_l / avg_st_l

    froude_denom = np.sqrt(9.81 * leg_length)
    gv_r = v_r / froude_denom
    gv_l = v_l / froude_denom
    mval = 1.3 / (np.sqrt(9.81 * 0.85))

    def get_step_function_penalty(gv_val, gsl_val, st_val):
        p_v = np.abs(gv_val - mval) / 0.082
        p_sl = 0 if 0.69 <= gsl_val <= 0.86 else (min(abs(gsl_val - 0.69), abs(gsl_val - 0.86)) / 0.2) * 7.33
        p_st = 0 if 0.50 <= st_val <= 0.63 else (min(abs(st_val - 0.50), abs(st_val - 0.63)) / 0.2) * 7.33
        return min(p_v + p_sl + p_st, 22)

    deduction_A = get_step_function_penalty(gv_l, gsl_l, avg_st_l)
    deduction_B = get_step_function_penalty(gv_r, gsl_r, avg_st_r)

    diff_asy = np.abs(gsl_r - gsl_l)
    deduction_C = 0 if diff_asy < 0.03 else min(((diff_asy - 0.03) / 0.15) * 8, 8)

    if 5 <= avg_dbs <= 10:
        deduction_D = 0
    else:
        dbs_diff = min(abs(avg_dbs - 5), abs(avg_dbs - 10))
        deduction_D = min((dbs_diff / 8) * 8, 8) 

    deduction_E = ambulatory_aids
    deduction_F = assistive_device

    total_deductions = deduction_A + deduction_B + deduction_C + deduction_D + deduction_E + deduction_F
    score_faps = 100 - total_deductions
    score_min = 30 if (ambulatory_aids>0 or assistive_device>0) else 40
    
    return max(score_min, score_faps)


def calculate_egvi(trials, static_file):
    try:
        acq_stat = ezc3d.c3d(static_file)
        pts_stat = acq_stat['data']['points']
        lbl_stat = acq_stat['parameters']['POINT']['LABELS']['value']

        iLPSI, iRPSI = lbl_stat.index('LPSI'), lbl_stat.index('RPSI')
        iLANK, iRANK = lbl_stat.index('LANK'), lbl_stat.index('RANK')
        iLASI, iRASI = lbl_stat.index('LASI'), lbl_stat.index('RASI')

        LgJambeR = np.mean(np.linalg.norm(pts_stat[:, iRANK, :] - pts_stat[:, iRPSI, :], axis=0))
        LgJambeL = np.mean(np.linalg.norm(pts_stat[:, iLANK, :] - pts_stat[:, iLPSI, :], axis=0))
        LgJambe_Moy_m = ((LgJambeL + LgJambeR) / 2) / 1000
    except Exception:
        return None, None, None

    global_diffs_left = {'StepLen': [], 'StepTime': [], 'StanceTime': [], 'SingleSup': [], 'Velocity': []}
    global_diffs_right = {'StepLen': [], 'StepTime': [], 'StanceTime': [], 'SingleSup': [], 'Velocity': []}

    def find_toe_offs_from_toes(z_toe_data, cycle_tuples, threshold_clearance=20.0):
        detected_tos = []
        for (start_frame, end_frame) in cycle_tuples:
            z_segment = z_toe_data[start_frame:end_frame]
            if len(z_segment) == 0: continue
            idx_min = np.argmin(z_segment)
            min_val = z_segment[idx_min]
            post_min_segment = z_segment[idx_min:]
            candidates = np.where(post_min_segment > (min_val + threshold_clearance))[0]
            if len(candidates) > 0:
                detected_tos.append(start_frame + idx_min + candidates[0])
        return detected_tos

    def calculate_diffs(raw_values, normalization_mean=None):
        if len(raw_values) < 2: return []
        arr = np.array(raw_values)
        if normalization_mean is None: normalization_mean = np.mean(arr)
        normalized_p = [(val) / normalization_mean * 100 for val in arr]
        return [abs(normalized_p[i+1] - normalized_p[i]) for i in range(len(normalized_p) - 1)]

    for fichier in trials:
        try:
            acq1 = ezc3d.c3d(fichier)
            labels = acq1['parameters']['POINT']['LABELS']['value']
            freq = acq1['header']['points']['frame_rate']
            points = acq1['data']['points']
            axis_ap = 0

            lhee_valid_cycles, rhee_valid_cycles = [], []
            lhee_cycle_start_indices, rhee_cycle_start_indices = [], []

            if "LHEE" in labels:
                idx_lhee = labels.index("LHEE")
                peaks, _ = find_peaks(-points[2, idx_lhee, :], distance=int(freq * 0.8), prominence=1)
                lhee_valid_cycles = [(s, e) for s, e in zip(peaks[:-1], peaks[1:]) if (e - s) >= int(0.5 * freq)]
                lhee_cycle_start_indices = peaks[:-1]

            if "RHEE" in labels:
                idx_rhee = labels.index("RHEE")
                peaks, _ = find_peaks(-points[2, idx_rhee, :], distance=int(freq * 0.8), prominence=1)
                rhee_valid_cycles = [(s, e) for s, e in zip(peaks[:-1], peaks[1:]) if (e - s) >= int(0.5 * freq)]
                rhee_cycle_start_indices = peaks[:-1]

            step_lengths_left, step_lengths_right = [], []
            all_events = sorted([(f, 'Left') for f in lhee_cycle_start_indices] + [(f, 'Right') for f in rhee_cycle_start_indices], key=lambda x: x[0])

            if "LHEE" in labels and "RHEE" in labels:
                idx_lhee, idx_rhee = labels.index("LHEE"), labels.index("RHEE")
                for i in range(1, len(all_events)):
                    current_frame, current_side = all_events[i]
                    prev_frame, prev_side = all_events[i-1]
                    if current_side != prev_side:
                        step_len = abs(points[axis_ap, idx_lhee, current_frame] - points[axis_ap, idx_rhee, current_frame])
                        if current_side == 'Left': step_lengths_left.append(step_len/LgJambe_Moy_m*100)
                        else: step_lengths_right.append(step_len/LgJambe_Moy_m*100)

            step_times_left, step_times_right = [], []
            for i in range(1, len(all_events)):
                current_frame, current_side = all_events[i]
                prev_frame, prev_side = all_events[i-1]
                if current_side != prev_side:
                    step_time_seconds = (current_frame - prev_frame) / freq
                    if current_side == 'Left': step_times_left.append(step_time_seconds)
                    else: step_times_right.append(step_time_seconds)

            lhee_toe_offs, rhee_toe_offs = [], []
            if "LTOE" in labels and len(lhee_valid_cycles) > 0:
                lhee_toe_offs = find_toe_offs_from_toes(points[2, labels.index("LTOE"), :], lhee_valid_cycles)
            if "RTOE" in labels and len(rhee_valid_cycles) > 0:
                rhee_toe_offs = find_toe_offs_from_toes(points[2, labels.index("RTOE"), :], rhee_valid_cycles)

            sst_left, sst_right = [], []
            stance_time_left, stance_time_right = [], []

            for (start, end) in lhee_valid_cycles:
                cycle_dur_frames = end - start
                next_tos = [to for to in lhee_toe_offs if start < to < end]
                if next_tos: stance_time_left.append(((next_tos[0] - start) / cycle_dur_frames) * 100)

            for (start, end) in rhee_valid_cycles:
                cycle_dur_frames = end - start
                next_tos = [to for to in rhee_toe_offs if start < to < end]
                if next_tos: sst_left.append(((end - next_tos[0]) / cycle_dur_frames) * 100)

            for (start, end) in rhee_valid_cycles:
                cycle_dur_frames = end - start
                next_tos = [to for to in rhee_toe_offs if start < to < end]
                if next_tos: stance_time_right.append(((next_tos[0] - start) / cycle_dur_frames) * 100)

            for (start, end) in lhee_valid_cycles:
                cycle_dur_frames = end - start
                next_tos = [to for to in lhee_toe_offs if start < to < end]
                if next_tos: sst_right.append(((end - next_tos[0]) / cycle_dur_frames) * 100)

            def get_velocity(cycles, marker_idx):
                vels = []
                for (start, end) in cycles:
                    dur = (end - start) / freq
                    dist = abs(points[axis_ap, marker_idx, end] - points[axis_ap, marker_idx, start]) / 10
                    if dur > 0: vels.append((dist/dur)/(sqrt(9.81*LgJambe_Moy_m)))
                return vels

            velocity_left, velocity_right = [], []
            if len(lhee_valid_cycles) > 0 and "LHEE" in labels:
                velocity_left = get_velocity(lhee_valid_cycles, labels.index("LHEE"))
            if len(rhee_valid_cycles) > 0 and "RHEE" in labels:
                velocity_right = get_velocity(rhee_valid_cycles, labels.index("RHEE"))

            global_diffs_left['StepLen'].extend(calculate_diffs(step_lengths_left))
            global_diffs_left['StepTime'].extend(calculate_diffs(step_times_left))
            global_diffs_left['StanceTime'].extend(calculate_diffs(stance_time_left))
            global_diffs_left['SingleSup'].extend(calculate_diffs(sst_left))
            global_diffs_left['Velocity'].extend(calculate_diffs(velocity_left))

            global_diffs_right['StepLen'].extend(calculate_diffs(step_lengths_right))
            global_diffs_right['StepTime'].extend(calculate_diffs(step_times_right))
            global_diffs_right['StanceTime'].extend(calculate_diffs(stance_time_right))
            global_diffs_right['SingleSup'].extend(calculate_diffs(sst_right))
            global_diffs_right['Velocity'].extend(calculate_diffs(velocity_right))
        except Exception:
            continue

    def calc_egvi_math(donnees_sujet):
        coeffs = np.array([0.80, 0.93, 0.92, 0.90, 0.89, 0.73, 0.82, 0.85, 0.86, 0.90])
        mean_control_s_alpha = 20.37541132570365
        mean_ln_d_control = 1.3865728033714733
        sd_ln_d_control = 0.6193340454665202
        if len(donnees_sujet) != 10: return 0.0
        donnees_propres = [0.0 if np.isnan(x) else x for x in donnees_sujet]
        s_alpha_sujet = np.dot(coeffs, donnees_propres)
        ln_d = math.log(1 + abs(s_alpha_sujet - mean_control_s_alpha))
        z_score = (ln_d - mean_ln_d_control) / sd_ln_d_control
        return 100 + (10 * z_score)

    def get_stats(diff_list):
        arr = np.array(diff_list)
        if len(arr) == 0: return 0.0, 0.0
        return np.mean(arr), np.std(arr, ddof=1)

    m_sl_r, sd_sl_r = get_stats(global_diffs_right['StepLen'])
    m_st_r, sd_st_r = get_stats(global_diffs_right['StepTime'])
    m_sta_r, sd_sta_r = get_stats(global_diffs_right['StanceTime'])
    m_ss_r, sd_ss_r = get_stats(global_diffs_right['SingleSup'])
    m_vel_r, sd_vel_r = get_stats(global_diffs_right['Velocity'])
    valeurs_D = [m_sl_r, m_st_r, m_sta_r, m_ss_r, m_vel_r, sd_sl_r, sd_st_r, sd_sta_r, sd_ss_r, sd_vel_r]

    m_sl_l, sd_sl_l = get_stats(global_diffs_left['StepLen'])
    m_st_l, sd_st_l = get_stats(global_diffs_left['StepTime'])
    m_sta_l, sd_sta_l = get_stats(global_diffs_left['StanceTime'])
    m_ss_l, sd_ss_l = get_stats(global_diffs_left['SingleSup'])
    m_vel_l, sd_vel_l = get_stats(global_diffs_left['Velocity'])
    valeurs_G = [m_sl_l, m_st_l, m_sta_l, m_ss_l, m_vel_l, sd_sl_l, sd_st_l, sd_sta_l, sd_ss_l, sd_vel_l]

    egvi_D = calc_egvi_math(valeurs_D)
    egvi_G = calc_egvi_math(valeurs_G)
    
    return egvi_G, egvi_D, (egvi_G + egvi_D) / 2


class MasterGaitAnalyzer:
    def __init__(self, healthy_matrix_path, n_points_per_curve=51):
        self.healthy_matrix_path = healthy_matrix_path
        self.n_points_per_curve = n_points_per_curve
        self.gdi_channels_config = [
            ('PelvisAngles', 0), ('PelvisAngles', 1), ('PelvisAngles', 2),
            ('HipAngles', 0), ('HipAngles', 1), ('HipAngles', 2),
            ('KneeAngles', 0), ('AnkleAngles', 0), ('FootProgressAngles', 2)
        ]
        self.gvs_labels = [
            'Pelvic Tilt', 'Pelvic Obliquity', 'Pelvic Rotation',
            'Hip Flexion', 'Hip Adduction', 'Hip Rotation',
            'Knee Flexion', 'Ankle Dorsiflexion', 'Foot Progression'
        ]
        self._load_and_fit_healthy_reference()

    def _load_and_fit_healthy_reference(self):
        if not os.path.exists(self.healthy_matrix_path):
            st.warning(f"Matrice saine '{self.healthy_matrix_path}' introuvable. Les scores GDI/GPS ne peuvent pas être calculés.")
            self.matrix_loaded = False
            return
        
        if self.healthy_matrix_path.endswith('.npy'):
            self.healthy_matrix = np.load(self.healthy_matrix_path)
        else:
            self.healthy_matrix = np.loadtxt(self.healthy_matrix_path, delimiter=',')

        self.mean_healthy_vector = np.mean(self.healthy_matrix, axis=1, keepdims=True)
        self.healthy_mean_curves = self.mean_healthy_vector.reshape(len(self.gdi_channels_config), self.n_points_per_curve)
        
        centered_healthy = self.healthy_matrix - self.mean_healthy_vector
        U, _, _ = np.linalg.svd(centered_healthy, full_matrices=False)
        self.feature_base = U[:, :15]
        projected_healthy = np.dot(self.feature_base.T, centered_healthy)
        raw_distances_healthy = np.sqrt(np.sum(projected_healthy**2, axis=0))
        self.mean_raw_healthy_gdi = np.mean(raw_distances_healthy)
        self.std_raw_healthy_gdi = np.std(raw_distances_healthy)
        self.matrix_loaded = True

    def _resample_curve(self, curve_data):
        if np.isnan(curve_data).any():
            curve_data = np.nan_to_num(curve_data, nan=np.nanmean(curve_data))
        x_old = np.linspace(0, 100, len(curve_data))
        x_new = np.linspace(0, 100, self.n_points_per_curve)
        return interp1d(x_old, curve_data, kind='cubic', fill_value="extrapolate")(x_new)

    def _extract_and_trim_channel(self, points_data, point_labels, target_label, axis_idx):
        if target_label not in point_labels: return np.zeros(10)
        idx = point_labels.index(target_label)
        raw_curve = points_data[axis_idx, idx, :]
        non_zero_indices = np.nonzero(raw_curve)[0]
        if len(non_zero_indices) == 0: return np.zeros(10)
        return raw_curve[:non_zero_indices[-1] + 1]

    def extract_kinematics_from_c3d(self, c3d_filepath):
        c3d = ezc3d.c3d(c3d_filepath)
        point_labels = [label.strip() for label in c3d['parameters']['POINT']['LABELS']['value']]
        points_data = c3d['data']['points']
        curves_L = np.zeros((len(self.gdi_channels_config), self.n_points_per_curve))
        curves_R = np.zeros((len(self.gdi_channels_config), self.n_points_per_curve))

        for i, (base_label, axis_idx) in enumerate(self.gdi_channels_config):
            label_L = "LPelvisAngles" if base_label == 'PelvisAngles' else f"L{base_label}"
            label_R = "LPelvisAngles" if base_label == 'PelvisAngles' else f"R{base_label}"
            raw_L = self._extract_and_trim_channel(points_data, point_labels, label_L, axis_idx)
            raw_R = self._extract_and_trim_channel(points_data, point_labels, label_R, axis_idx)
            curves_L[i, :] = self._resample_curve(raw_L)
            curves_R[i, :] = self._resample_curve(raw_R)
        return curves_L, curves_R

    def compute_gdi_trial(self, patient_vector_col):
        centered = patient_vector_col - self.mean_healthy_vector
        projected = np.dot(self.feature_base.T, centered)
        z_score = (np.sqrt(np.sum(projected**2)) - self.mean_raw_healthy_gdi) / self.std_raw_healthy_gdi
        return 100.0 - (10.0 * z_score)

    def compute_gps_trial(self, patient_curves_matrix):
        gvs = np.sqrt(np.mean((patient_curves_matrix - self.healthy_mean_curves)**2, axis=1))
        return np.sqrt(np.mean(gvs**2)), gvs

    def run_full_analysis(self, c3d_files_list):
        if not getattr(self, 'matrix_loaded', False): return None
        scores = {'GDI_L': [], 'GDI_R': [], 'GPS_L': [], 'GPS_R': [], 'GVS_L': [], 'GVS_R': []}
        for filepath in c3d_files_list:
            curves_L, curves_R = self.extract_kinematics_from_c3d(filepath)
            scores['GDI_L'].append(self.compute_gdi_trial(curves_L.reshape(-1, 1)))
            scores['GDI_R'].append(self.compute_gdi_trial(curves_R.reshape(-1, 1)))
            gps_L, gvs_L = self.compute_gps_trial(curves_L)
            gps_R, gvs_R = self.compute_gps_trial(curves_R)
            scores['GPS_L'].append(gps_L); scores['GPS_R'].append(gps_R)
            scores['GVS_L'].append(gvs_L); scores['GVS_R'].append(gvs_R)

        return {
            'GDI_L': np.mean(scores['GDI_L']), 'GDI_R': np.mean(scores['GDI_R']),
            'GDI_Total': (np.mean(scores['GDI_L']) + np.mean(scores['GDI_R'])) / 2,
            'GPS_L': np.mean(scores['GPS_L']), 'GPS_R': np.mean(scores['GPS_R']),
            'GPS_Total': (np.mean(scores['GPS_L']) + np.mean(scores['GPS_R'])) / 2,
            'GVS_L': np.mean(np.array(scores['GVS_L']), axis=0),
            'GVS_R': np.mean(np.array(scores['GVS_R']), axis=0),
            'Labels': self.gvs_labels
        }


# ==========================================
# INTERFACE UTILISATEUR
# ==========================================

def save_temp_file(uploaded_file):
    if uploaded_file is None: return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name

st.info("Veuillez importer 1 fichier statique et exactement 3 fichiers dynamiques pour chaque session afin d'effectuer la comparaison.")

col1, col2 = st.columns(2)

with col1:
    st.header("🔵 Session 1")
    s1_stat_file = st.file_uploader("Fichier statique S1", type="c3d", key="s1_stat")
    s1_dyn_files = st.file_uploader("Fichiers dynamiques S1 (3 requis)", type="c3d", accept_multiple_files=True, key="s1_dyn")
    s1_amb = st.selectbox("Aide ambulatoire S1 (0 à 5):", [0,1,2,3,4,5], key="s1_a")
    s1_ast = st.selectbox("Dispositif d'assistance S1 (0 à 5):", [0,1,2,3,4,5], key="s1_ast")

with col2:
    st.header("🟠 Session 2")
    s2_stat_file = st.file_uploader("Fichier statique S2", type="c3d", key="s2_stat")
    s2_dyn_files = st.file_uploader("Fichiers dynamiques S2 (3 requis)", type="c3d", accept_multiple_files=True, key="s2_dyn")
    s2_amb = st.selectbox("Aide ambulatoire S2 (0 à 5):", [0,1,2,3,4,5], key="s2_a")
    s2_ast = st.selectbox("Dispositif d'assistance S2 (0 à 5):", [0,1,2,3,4,5], key="s2_ast")


if st.button("Lancer la comparaison des deux sessions", type="primary", use_container_width=True):
    if not (s1_stat_file and len(s1_dyn_files) == 3 and s2_stat_file and len(s2_dyn_files) == 3):
        st.error("⚠️ Vous devez fournir exactement 1 fichier statique et 3 fichiers dynamiques pour CHAQUE session.")
    else:
        with st.spinner("Analyse et calcul des scores en cours..."):
            
            # Sauvegarde des fichiers S1
            s1_stat_path = save_temp_file(s1_stat_file)
            s1_dyn_paths = [save_temp_file(f) for f in s1_dyn_files]
            
            # Sauvegarde des fichiers S2
            s2_stat_path = save_temp_file(s2_stat_file)
            s2_dyn_paths = [save_temp_file(f) for f in s2_dyn_files]

            # Calculs Session 1
            s1_faps = calculate_faps(s1_dyn_paths, s1_stat_path, s1_amb, s1_ast)
            s1_efaps = calculate_efaps(s1_dyn_paths, s1_stat_path, s1_amb, s1_ast)
            s1_egvi_g, s1_egvi_d, s1_egvi_tot = calculate_egvi(s1_dyn_paths, s1_stat_path)
            
            # Calculs Session 2
            s2_faps = calculate_faps(s2_dyn_paths, s2_stat_path, s2_amb, s2_ast)
            s2_efaps = calculate_efaps(s2_dyn_paths, s2_stat_path, s2_amb, s2_ast)
            s2_egvi_g, s2_egvi_d, s2_egvi_tot = calculate_egvi(s2_dyn_paths, s2_stat_path)

            # Analyse Cinématique (GDI / GPS)
            matrice_saine = "matrice_temoins_459.npy"
            analyzer = MasterGaitAnalyzer(matrice_saine)
            res_s1 = analyzer.run_full_analysis(s1_dyn_paths)
            res_s2 = analyzer.run_full_analysis(s2_dyn_paths)

        # Affichage des résultats
        st.divider()
        st.header("📊 Comparaison Numérique")

        # Métriques FAPS, eFAPS, eGVI
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score FAPS (Total)", f"{s2_faps:.1f}" if s2_faps else "N/A", 
                  f"{(s2_faps - s1_faps):.1f}" if (s1_faps and s2_faps) else None)
        c2.metric("Score eFAPS (Total)", f"{s2_efaps:.1f}" if s2_efaps else "N/A", 
                  f"{(s2_efaps - s1_efaps):.1f}" if (s1_efaps and s2_efaps) else None)
        c3.metric("Score eGVI (Global)", f"{s2_egvi_tot:.1f}" if s2_egvi_tot else "N/A", 
                  f"{(s2_egvi_tot - s1_egvi_tot):.1f}" if (s1_egvi_tot and s2_egvi_tot) else None)
        
        if res_s1 and res_s2:
            c4.metric("GDI (Moyenne L/R)", f"{res_s2['GDI_Total']:.1f}", 
                      f"{(res_s2['GDI_Total'] - res_s1['GDI_Total']):.1f}")
            
            st.divider()
            st.header("📈 Comparaison Graphique")
            sns.set_theme(style='whitegrid', palette='colorblind')

            # Bar Chart Global Scores
            fig_global, ax_g = plt.subplots(figsize=(10, 5))
            labels = ['FAPS', 'eFAPS', 'eGVI', 'GDI (L)', 'GDI (R)']
            vals_s1 = [s1_faps or 0, s1_efaps or 0, s1_egvi_tot or 0, res_s1['GDI_L'], res_s1['GDI_R']]
            vals_s2 = [s2_faps or 0, s2_efaps or 0, s2_egvi_tot or 0, res_s2['GDI_L'], res_s2['GDI_R']]
            
            x = np.arange(len(labels))
            width = 0.35
            ax_g.bar(x - width/2, vals_s1, width, label='🔵 Session 1', color='#1f77b4')
            ax_g.bar(x + width/2, vals_s2, width, label='🟠 Session 2', color='#ff7f0e')
            ax_g.set_xticks(x)
            ax_g.set_xticklabels(labels, fontweight='bold')
            ax_g.set_title("Scores Globaux (Plus haut = meilleur, ~100 = sain)", fontweight='bold')
            ax_g.legend()
            st.pyplot(fig_global)

            # MAP Chart (Movement Analysis Profile) Comparatif
            st.subheader("Gait Variable Score (GVS) - Profil d'Analyse du Mouvement")
            
            def plot_comparison_map(gvs1, gvs2, title, gps1, gps2):
                y = np.arange(len(analyzer.gvs_labels))
                height = 0.38
                fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
                
                # Inverser pour l'affichage visuel classique (haut vers bas)
                val1 = gvs1[::-1]
                val2 = gvs2[::-1]
                labels_inv = analyzer.gvs_labels[::-1]

                ax.barh(y + height/2, val1, height, label=f'🔵 Session 1 (GPS: {gps1:.1f}°)', color='#1f77b4')
                ax.barh(y - height/2, val2, height, label=f'🟠 Session 2 (GPS: {gps2:.1f}°)', color='#ff7f0e')
                
                ax.axvline(5.4, color='gray', linestyle='--', linewidth=1.5, label='Réf. Saine (5.4°)')
                ax.set_yticks(y)
                ax.set_yticklabels(labels_inv, fontsize=10, fontweight='bold')
                ax.set_xlabel('GVS (°)', fontsize=11, fontweight='bold')
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.legend(loc='lower right')
                sns.despine(left=True)
                plt.tight_layout()
                return fig

            col_map1, col_map2 = st.columns(2)
            with col_map1:
                st.pyplot(plot_comparison_map(res_s1['GVS_L'], res_s2['GVS_L'], "Évolution Côté GAUCHE", res_s1['GPS_L'], res_s2['GPS_L']))
            with col_map2:
                st.pyplot(plot_comparison_map(res_s1['GVS_R'], res_s2['GVS_R'], "Évolution Côté DROIT", res_s1['GPS_R'], res_s2['GPS_R']))
