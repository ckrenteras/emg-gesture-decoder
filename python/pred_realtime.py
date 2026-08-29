import serial
import csv
import os
import json
import pandas as pd
from scipy.stats import median_abs_deviation
from scipy.signal import butter, sosfilt
from train_offline_models import (
    adc_to_volts, CALIBRATION_Z_CLIP,
    SAMPLE_RATE, LOW_PASS_FREQ, HIGH_PASS_FREQ,
)
import numpy as np

import tkinter as tk


SERIAL_PORT = "/dev/cu.usbmodem202636001"
BAUD_RATE = 115200
MODEL_DIR = os.path.join('..', 'models', 'v3')
FEATURE_COL_PATH = os.path.join(MODEL_DIR, 'feature_columns.json')
CLASS_MAPPING_PATH = os.path.join(MODEL_DIR, 'class_mapping.json')
COVARIANCE_PATH = os.path.join(MODEL_DIR, 'lda_shared_covariance.npy')
TRAIN_CENTROIDS_PATH = os.path.join(MODEL_DIR, 'train_centroids.json')
CLASS_PRIORS_PATH = os.path.join(MODEL_DIR, 'class_priors.json')
WINDOW_SIZE = 400
STEP_SIZE = 200
NUM_WINDOWS_FOR_VOTE = 10 # for now
BUFFER_SIZE = WINDOW_SIZE + (NUM_WINDOWS_FOR_VOTE - 1) * STEP_SIZE  # for now
CALIBRATION_SAMPLES = 10400 # first 5 seconds, drop first window
GESTURE_CALIBRATION_SAMPLES = 5200  # ~2.5s per active gesture, drop first window
PRED_HISTORY_SIZE = 1
FILTER_ORDER = 3
WAMP_THRESHOLD = 0.003


GESTURE_DISPLAY_MAPPING = {
    "open_hand": "Open hand",
    "rest": "Rest",
    "pinch": "Pinch",
    "chaka": "Chaka"
}

# to avoid transient at each step filter
# one sample at a time with the filter's state (zi) carried forward
SOS_LOWPASS = butter(N=FILTER_ORDER, Wn=LOW_PASS_FREQ, fs=SAMPLE_RATE, btype='lowpass', output='sos')
SOS_HIGHPASS = butter(N=FILTER_ORDER, Wn=HIGH_PASS_FREQ, fs=SAMPLE_RATE, btype='highpass', output='sos')
zi_lowpass = np.zeros((SOS_LOWPASS.shape[0], 2))
zi_highpass = np.zeros((SOS_HIGHPASS.shape[0], 2))

def filter_sample(adc_value):
    global zi_lowpass, zi_highpass
    voltage = adc_to_volts(np.array([adc_value], dtype=float))
    low, zi_lowpass = sosfilt(SOS_LOWPASS, voltage, zi=zi_lowpass)
    high, zi_highpass = sosfilt(SOS_HIGHPASS, low, zi=zi_highpass)
    return float(high[0])

# connect to port
print(f'Connecting to Teensy at port: {SERIAL_PORT}')
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

# recalibration
with open(FEATURE_COL_PATH, 'r') as file:
    feature_cols = json.load(file)
with open(CLASS_MAPPING_PATH, 'r') as file:
    class_mapping = json.load(file)
with open(TRAIN_CENTROIDS_PATH, 'r') as file:
    train_centroids = {k: np.array(v) for k, v in json.load(file).items()}
with open(CLASS_PRIORS_PATH, 'r') as file:
    class_priors = json.load(file)
shared_covariance = np.load(COVARIANCE_PATH)
shared_cov_inv = np.linalg.pinv(shared_covariance)

REST_CLASS = next(k for k, v in class_mapping.items() if v == 'rest')
ACTIVE_CLASSES = [k for k in class_mapping if k != REST_CLASS]

# ======== baseline calibration =========

def get_rest_stats(feature_df, amplitude_cols=feature_cols):
    rest_median = feature_df[amplitude_cols].median()
    rest_mad = feature_df[amplitude_cols].apply(
        lambda s: median_abs_deviation(s, scale='normal', nan_policy='omit')
    )
    return rest_median, rest_mad

def apply_baseline_calibration(feature_df, rest_median, rest_mad,
                               amplitude_cols=feature_cols, eps=1e-8, clip=CALIBRATION_Z_CLIP):
    calibrated_df = feature_df.copy()
    z = (calibrated_df[amplitude_cols] - rest_median) / (rest_mad + eps)
    calibrated_df[amplitude_cols] = z.clip(-clip, clip)
    return calibrated_df

# get pred based off of realtime data

def get_features(emg_data, feature_cols=feature_cols):
    """emg_data must already be filtered (see filter_sample), this only windows
    and computes amplitude/shape features, it does no filtering itself."""
    df = pd.DataFrame({'emg': np.asarray(emg_data)})
    windowed_rms_col = df["emg"].pow(2).rolling(window=WINDOW_SIZE, step=STEP_SIZE).mean().pow(0.5)
    windowed_wfl_col = df["emg"].rolling(window=WINDOW_SIZE, step=STEP_SIZE).apply(lambda x: np.abs(np.diff(x)).sum())
    windowed_stdev_col = df["emg"].rolling(window=WINDOW_SIZE, step=STEP_SIZE).std()
    windowed_mav_col = df["emg"].rolling(window=WINDOW_SIZE, step=STEP_SIZE).apply(lambda x: np.abs(x).mean())
    windowed_min_col = df["emg"].rolling(window=WINDOW_SIZE, step=STEP_SIZE).apply(lambda x: np.abs(x).min())
    windowed_max_col = df["emg"].rolling(window=WINDOW_SIZE, step=STEP_SIZE).apply(lambda x: np.abs(x).max())
    def crossings(s):
        return (s.shift(1) * s < 0)
    windowed_zc_col = crossings(df['emg']).rolling(window=WINDOW_SIZE, step=STEP_SIZE).sum()
    freqs = np.fft.rfftfreq(WINDOW_SIZE, d=1/SAMPLE_RATE)
    def mean_freq(x):
        """frequency weighted avg"""
        power = np.abs(np.fft.rfft(x)) ** 2
        return np.sum(freqs * power) / np.sum(power)
    windowed_mean_freq_col = df["emg"].rolling(window=WINDOW_SIZE, step=STEP_SIZE).apply(mean_freq, raw=True)
    def median_freq(x):
        """emg median freq MDF"""
        power = np.abs(np.fft.rfft(x)) ** 2
        cum_power = np.cumsum(power)
        return freqs[np.searchsorted(cum_power, cum_power[-1] / 2)]
    windowed_median_freq_col = df["emg"].rolling(window=WINDOW_SIZE, step=STEP_SIZE).apply(median_freq, raw=True)
    def ssc(x):
        diffs = np.diff(x)
        return np.sum(diffs[:-1] * diffs[1:] < 0)
    windowed_ssc_col = df["emg"].rolling(window=WINDOW_SIZE, step=STEP_SIZE).apply(ssc, raw=True)
    def wamp(x, threshold=WAMP_THRESHOLD):
        return np.sum(np.abs(np.diff(x)) > threshold)
    windowed_wamp_col = df["emg"].rolling(window=WINDOW_SIZE, step=STEP_SIZE).apply(wamp, raw=True)
    feature_df = pd.DataFrame({"RMS": windowed_rms_col,
                               "waveform_len": windowed_wfl_col,
                               "MAV": windowed_mav_col,
                               "max_abs": windowed_max_col,
                               "min_abs": windowed_min_col,
                               "zero_crossings": windowed_zc_col,
                               "std": windowed_stdev_col,
                               "mean_freq": windowed_mean_freq_col,
                               "median_freq": windowed_median_freq_col,
                               "ssc": windowed_ssc_col,
                               "wamp": windowed_wamp_col})
    feature_df = feature_df.dropna()
    return feature_df

# ======== session-recalibrated LDA discriminant ========

def lda_discriminant_scores(features, live_centroids, cov_inv=shared_cov_inv, priors=class_priors):
    """Standard LDA discriminant score per class, using the shared covariance learned at
    training time but per-class means captured live at the start of THIS session."""
    classes = sorted(live_centroids.keys(), key=int)
    scores = np.zeros((len(features), len(classes)))
    for i, c in enumerate(classes):
        mu = live_centroids[c]
        w = cov_inv @ mu
        b = -0.5 * mu @ cov_inv @ mu + np.log(priors[c])
        scores[:, i] = features @ w + b
    return scores, classes

def get_pred(features, live_centroids):
    if len(features) == 0:
        return None, None, None

    scores, classes = lda_discriminant_scores(features, live_centroids)
    mean_scores = np.mean(scores, axis=0)
    # softmax over discriminant scores for a probability-like confidence value
    exp_scores = np.exp(mean_scores - mean_scores.max())
    probs = exp_scores / exp_scores.sum()

    best_idx = np.argmax(mean_scores)
    pred = int(classes[best_idx])
    prob = probs[best_idx]

    return pred, prob, probs


def class_to_gesture(class_num, class_mapping=class_mapping):
    if str(class_num) not in class_mapping:
        raise ValueError(f'Unexpected class_num: {class_num}')

    return class_mapping[str(class_num)]

def gesture_to_display_name(gesture, gesture_display_mapping=GESTURE_DISPLAY_MAPPING):
    if str(gesture) not in gesture_display_mapping:
        raise ValueError(f'Unexpected gesture: {gesture}')
    return gesture_display_mapping[str(gesture)]


def read_filtered_sample():
    """Blocks until one valid filtered sample is available from the serial port."""
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line or line == 'adc':
            continue
        try:
            adc_value = int(line)
        except ValueError:
            print(f"Skipping malformed line: {line}")
            continue
        return filter_sample(adc_value)


def main():
    root = tk.Tk()
    root.title("Prediction Display")
    root.geometry("1200x800")
    root.configure(bg='black')
    label = tk.Label(root, text="Calibrating...", font=('Helvatica', 90, "bold"),
                     relief="flat", bg="black")
    label.pack(expand=True)
    label_bottom = tk.Label(root, font=('Helvatica', 15), relief="flat", bg="black")
    root.update()

    try:
        # ---- stage 1: rest calibration (unchanged) ----
        print("Calibrating: rest...")
        label.config(text="Calibrating: Rest")
        root.update()
        rest_raw = []
        while len(rest_raw) < CALIBRATION_SAMPLES:
            rest_raw.append(read_filtered_sample())
            if len(rest_raw) % 500 == 0:
                root.update()
        rest_features = get_features(rest_raw[400:])
        rest_median, rest_mad = get_rest_stats(rest_features)
        ser.reset_input_buffer()
        print("Rest calibration complete.")

        # ---- stage 2: live per-gesture calibration ----
        # captures a brief live sample of each active gesture and uses it, together with
        # the covariance/priors learned at training time, in place of the frozen
        # training-set class means -- this is what fixes the session-to-session drift
        # that a frozen model can't adapt to.
        live_centroids = {REST_CLASS: train_centroids[REST_CLASS]}  # rest is centered at 0 post-calibration by construction; keep training value as a stable anchor
        for class_key in ACTIVE_CLASSES:
            gesture = class_to_gesture(class_key)
            display_name = gesture_to_display_name(gesture)
            print(f"Calibrating: {gesture}...")
            label.config(text=f"Calibrating: {display_name}\nHold the gesture")
            root.update()
            gesture_raw = []
            while len(gesture_raw) < GESTURE_CALIBRATION_SAMPLES:
                gesture_raw.append(read_filtered_sample())
                if len(gesture_raw) % 500 == 0:
                    root.update()
            gesture_features = get_features(gesture_raw[400:])
            gesture_cal = apply_baseline_calibration(gesture_features, rest_median, rest_mad)
            if len(gesture_cal) > 0:
                live_centroids[class_key] = gesture_cal[feature_cols].mean(axis=0).to_numpy()
            else:
                print(f"WARNING: no usable calibration windows for {gesture}, falling back to training centroid.")
                live_centroids[class_key] = train_centroids[class_key]
            ser.reset_input_buffer()

        label.config(text="Calibration Complete")
        root.update()
        print("Full calibration complete. Starting predictions.")

        # ---- stage 3: live prediction using session-recalibrated LDA ----
        running_data = []
        while True:
            root.update()
            running_data.append(read_filtered_sample())
            if len(running_data) >= BUFFER_SIZE:
                features = get_features(running_data)
                cal_df = apply_baseline_calibration(features, rest_median, rest_mad)
                feature_arr = cal_df[feature_cols].to_numpy()
                pred, prob, probs = get_pred(feature_arr, live_centroids)
                if pred is not None:
                    pred_gesture = class_to_gesture(pred)
                    display_gesture = gesture_to_display_name(pred_gesture)
                    label.config(text=f"{display_gesture}")
                    prob_as_percent = prob * 100
                    label_bottom.config(text=f"Confidence: {prob_as_percent:.2f}%")
                    label_bottom.pack(side="bottom", pady=40)
                    print(f"Prediction: {pred_gesture} | Confidence: {prob}")
                running_data = running_data[STEP_SIZE:]

    except KeyboardInterrupt:
        print('\n Logging stopped')
    finally:
        ser.close()
if __name__ == "__main__":
    main()
