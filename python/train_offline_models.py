import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from scipy.signal import butter, sosfilt
from scipy.stats import median_abs_deviation
import sklearn.metrics as metrics
import pickle
import json

DATA_PATH = csv_path = os.path.join("..", "data", 'my_data', 'subject_one', "combined_subject_one.csv")
RESULTS_PATH = os.path.join('.', 'results', 'my_data')
MODEL_PATH = os.path.join("..", "models", "v4",)

# ======= pre processing consts =====
LOW_PASS_FREQ = 500
HIGH_PASS_FREQ = 20
SAMPLE_RATE = 2000 # Hz
WINDOW = 400
STEP = 200
WAMP_THRESHOLD = 0.003


# ====== dataset consts =======
# note, sessions [1-23] were 1 session, 24-33 a second session
# significant variation from session to session
TRIALS = range(1, 33)
CLASSES = [0, 1, 3, 6] # rest, open_hand, pinch, chaka, easy to distinguish
BITS = 10
VREF = 3.7

#=======  model constants =======
FEATURE_COLS = ["RMS", "waveform_len", "MAV", "max_abs", "min_abs", "std", 'zero_crossings',
                'mean_freq', 'median_freq', 'ssc']
CALIBRATION_Z_CLIP = 32
CLASS_MAPPING = {
    0: "rest",
    1: "open_hand",
    3: "pinch",
    6: "chaka"
}

# ====== training consts ===========
TEST_TRIALS = [32, 30, 19, 18, 20, 23, 29] # 26% of data
RECAL_CALIBRATION_WINDOWS = 20 #2.1s

#======== pre processing ============

def butterworth_filter(data, order=3, cutoff=HIGH_PASS_FREQ, fs=SAMPLE_RATE, filter_type='highpass'):
    if filter_type not in ['lowpass', 'highpass', 'bandpass', 'bandstop']:
        raise ValueError("Invalid filter type requested")
    sos = butter(N=order, Wn=cutoff, fs=SAMPLE_RATE, btype=filter_type, output='sos')
    filtered_data = sosfilt(sos, data)
    return filtered_data
    
def filter_emg(df, cutoff=HIGH_PASS_FREQ, fs=SAMPLE_RATE, butter_order=3, filter_type='highpass'):
    df['emg_filtered'] = df.groupby('trial')['emg'].transform(
        lambda s: butterworth_filter(s.to_numpy(), order=butter_order, cutoff=cutoff, fs=fs, filter_type=filter_type)
    )
    pre_processed_df = pd.DataFrame()
    pre_processed_df['sample'] = df['sample']
    pre_processed_df['time_ms'] = df['time_ms']
    pre_processed_df['emg'] = df['emg_filtered']
    pre_processed_df['gesture'] = df['gesture']
    pre_processed_df['class'] = df['class']
    pre_processed_df['subject'] = df['subject']
    pre_processed_df['trial'] = df['trial']
    return pre_processed_df

# ===== feature extraction =========

def adc_to_volts(adc, vref=VREF, bits=BITS):
     return (adc * vref) / (2 ** bits - 1)

def get_feature_df(df, read_trials=range(1, 9), classes=range(0, 5), filter=True, 
                   order=3, fs=SAMPLE_RATE, hpf=HIGH_PASS_FREQ, lpf=LOW_PASS_FREQ):
    df = df.copy()
    df['emg'] = df['adc'].apply(lambda x: adc_to_volts(x))
    if filter:
         low_passed_df = filter_emg(df, cutoff=lpf, fs=fs, butter_order=order, filter_type='lowpass')
         df = filter_emg(low_passed_df, cutoff=hpf, fs=fs, butter_order=order, filter_type='highpass')
    df['trial_mask'] = df['trial'].apply(lambda x: x in read_trials)
    df = df[df['trial_mask']]
    df['class_mask'] = df["class"].apply(lambda x: x in classes)
    df = df[df['class_mask']]

    windowed_class = df["class"].rolling(window=WINDOW, step=STEP).apply(lambda w: w.iloc[0])
    windowed_subject = df["subject"].rolling(window=WINDOW, step=STEP).apply(lambda w: w.iloc[0])
    windowed_trial = df["trial"].rolling(window=WINDOW, step=STEP).apply(lambda w: w.iloc[0])
    windowed_rms_col = df["emg"].pow(2).rolling(window=WINDOW, step=STEP).mean().pow(0.5)
    windowed_wfl_col = df["emg"].rolling(window=WINDOW, step=STEP).apply(lambda x: np.abs(np.diff(x)).sum())
    windowed_stdev_col = df["emg"].rolling(window=WINDOW, step=STEP).std()
    windowed_mav_col = df["emg"].rolling(window=WINDOW, step=STEP).apply(lambda x: np.abs(x).mean())
    windowed_min_col = df["emg"].rolling(window=WINDOW, step=STEP).apply(lambda x: np.abs(x).min())
    windowed_max_col = df["emg"].rolling(window=WINDOW, step=STEP).apply(lambda x: np.abs(x).max())
    freqs = np.fft.rfftfreq(WINDOW, d=1/SAMPLE_RATE)
    def mean_freq(x):
        """frequency weighted avg"""
        power = np.abs(np.fft.rfft(x)) ** 2
        return np.sum(freqs * power) / np.sum(power)
    windowed_mean_freq_col = df["emg"].rolling(window=WINDOW, step=STEP).apply(mean_freq, raw=True)
    def median_freq(x):
        """emg median freq MDF"""
        power = np.abs(np.fft.rfft(x)) ** 2
        cum_power = np.cumsum(power)
        return freqs[np.searchsorted(cum_power, cum_power[-1] / 2)]
    windowed_median_freq_col = df["emg"].rolling(window=WINDOW, step=STEP).apply(median_freq, raw=True)
    def crossings(s): 
        return (s.shift(1) * s < 0)
    windowed_zc_col = crossings(df['emg']).rolling(window=WINDOW, step=STEP).sum()
    
    rolling_class = df["class"].rolling(window=WINDOW, step=STEP)
    windowed_mask_mixed_col = rolling_class.min() == rolling_class.max()
    rolling_trial = df["trial"].rolling(window=WINDOW, step=STEP)
    windowed_mask_mixed_trial = rolling_trial.min() == rolling_trial.max()
    def ssc(x):
        diffs = np.diff(x)
        return np.sum(diffs[:-1] * diffs[1:] < 0)
    windowed_ssc_col = df["emg"].rolling(window=WINDOW, step=STEP).apply(ssc, raw=True)
    def wamp(x, threshold=WAMP_THRESHOLD):
        return np.sum(np.abs(np.diff(x)) > threshold)
    windowed_wamp_col = df["emg"].rolling(window=WINDOW, step=STEP).apply(wamp, raw=True)
    feature_df = pd.DataFrame({"RMS": windowed_rms_col,
                               "waveform_len": windowed_wfl_col,
                               "MAV": windowed_mav_col,
                               "max_abs": windowed_max_col,
                               "min_abs": windowed_min_col,
                               "std": windowed_stdev_col,
                               "zero_crossings": windowed_zc_col,
                               'mask_mixed_col': windowed_mask_mixed_col,
                               "class": windowed_class,
                               'subject': windowed_subject,
                               'trial': windowed_trial,
                               'mask_mixed_trial': windowed_mask_mixed_trial,
                               'mean_freq': windowed_mean_freq_col,
                               'median_freq': windowed_median_freq_col,
                               "ssc": windowed_ssc_col,
                               "wamp": windowed_wamp_col})
    feature_df = feature_df[feature_df['mask_mixed_col']]
    feature_df = feature_df[feature_df['mask_mixed_trial']]

    # drop each trials first window because Butterworth filter has
    # 0 init cond. first window is filter's startup transient
    feature_df = feature_df[feature_df.groupby('trial').cumcount() != 0]
    return feature_df

def get_train_test_df(data_path=DATA_PATH, train_trials=range(1, 9), 
                      test_trials=range(9, 11), classes=range(0, 5)):
    for trial in test_trials:
        if trial in train_trials:
            raise ValueError("WARNING: Data leakage. Overlapping trials in test and train split")
        if trial > max(TRIALS) or trial < 1:
            raise ValueError("Ivalid test trial requested")
    for trial in train_trials:
        if trial > max(TRIALS) or trial < 1:
            raise ValueError("Ivalid train trial requested")
    
    df = pd.read_csv(data_path)
    train_df = get_feature_df(df, read_trials=train_trials, classes=classes)
    test_df = get_feature_df(df, read_trials=test_trials, classes=classes)
    return train_df, test_df

# ============== baseline calibration ============

def get_subject_rest_stats(feature_df, amplitude_cols, rest_class=0, max_samples=10000):
    rest_df = feature_df[feature_df['class'] == rest_class]
    if max_samples is not None:
        rest_df = rest_df.groupby('trial', group_keys=False).head(max_samples)
    rest_median = rest_df.groupby('trial')[amplitude_cols].median()
    rest_mad = rest_df.groupby('trial')[amplitude_cols].agg(
        lambda s: median_abs_deviation(s, scale='normal', nan_policy='omit')
    )
    return rest_median, rest_mad

def apply_baseline_calibration(feature_df, amplitude_cols=FEATURE_COLS, rest_class=0,
                               max_samples=5000, eps=1e-8, clip=CALIBRATION_Z_CLIP):
    rest_median, rest_mad = get_subject_rest_stats(feature_df, amplitude_cols, rest_class=rest_class, max_samples=max_samples)
    calibrated_df = feature_df.copy()
    for trial in rest_median.index:
        mask = calibrated_df['trial'] == trial
        z = (
            (calibrated_df.loc[mask, amplitude_cols] - rest_median.loc[trial])
            / (rest_mad.loc[trial] + eps)
        )
        calibrated_df.loc[mask, amplitude_cols] = z.clip(-clip, clip)
    return calibrated_df

# ======== train + eval ===============

def get_df_features_labels(df, feature_cols=FEATURE_COLS):
    features = df[feature_cols].to_numpy()
    labels = df["class"].to_numpy()
    return features, labels

def train_lda(df, feature_cols=FEATURE_COLS, store_covariance=True):
    features, labels = get_df_features_labels(df, feature_cols)
    lda = LinearDiscriminantAnalysis(store_covariance=store_covariance)
    lda.fit(features, labels)
    return lda

def lda_discriminant_scores(features, centroids, cov_inv, priors, classes):
    """from scratch implementation of lda disc. f'n so we can dynamically adjust centroids
    per trials"""
    scores = np.zeros((len(features), len(classes)))
    for i, c in enumerate(classes):
        mu = centroids[c]
        w = cov_inv @ mu
        b = -0.5 * mu.T @ cov_inv @ mu + np.log(priors[i])
        scores[:, i] = features @ w + b
    return scores

def evaluate_lda_recalibrated(model, df, feature_cols=FEATURE_COLS,
                              classes=CLASSES, calibration_windows=RECAL_CALIBRATION_WINDOWS):
    
    cov_inv = np.linalg.pinv(model.covariance_)
    priors = model.priors_
    classes_sorted = sorted(classes)
    all_true, all_pred = [], []
    skipped = []
    for trial, trial_df in df.groupby('trial'):
        centroids = {}
        eval_idx_parts = []
        for c in classes_sorted:
            class_rows = trial_df[trial_df['class'] == c]
            if len(class_rows) <= calibration_windows:
                print(f"skipping class {c} in trial {trial}")
                skipped.append(trial)
                centroids = None
                break
            centroids[c] = class_rows.iloc[:calibration_windows][feature_cols].mean(axis=0).to_numpy()
            eval_idx_parts.append(class_rows.index[calibration_windows:])
        if centroids is None:
            continue
        eval_rows = trial_df.loc[np.concatenate(eval_idx_parts)]
        features = eval_rows[feature_cols].to_numpy()
        scores = lda_discriminant_scores(features, centroids, cov_inv, priors, classes_sorted)
        preds = np.array(classes_sorted)[np.argmax(scores, axis=1)]
        all_true.extend(eval_rows['class'].tolist())
        all_pred.extend(preds.tolist())

    if skipped:
        print(f"WARNING: skipped trials with too few windows to recalibrate: {sorted(set(skipped))}")

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    results = {
        'preds': all_pred,
        'score': float((all_true == all_pred).mean()),
        'f1': metrics.f1_score(all_true, all_pred, average='macro'),
        'bacc': metrics.balanced_accuracy_score(all_true, all_pred),
        'confusion_matrix': metrics.confusion_matrix(all_true, all_pred, labels=classes_sorted)
    }
    return results
# =========== save results =============

def save_performance_metrics(train_results, test_results, classes, classifier_type='log_reg', results_dir=RESULTS_PATH):
    os.makedirs(results_dir, exist_ok=True)
    tag = ""
    for c in classes:
        tag += str(c)
    train_clean = train_results.copy()
    test_clean = test_results.copy()
    del train_clean['preds']
    del test_clean['preds']
    train_confusion_matrix = train_clean.pop('confusion_matrix', None)
    test_confusion_matrix = test_clean.pop('confusion_matrix', None)
 
    train_confusion_path = os.path.join(results_dir, classifier_type + '_train_confusion_matrix_' + tag + '.npy')
    test_confusion_path = os.path.join(results_dir, classifier_type + '_test_confusion_matrix_' + tag + '.npy')
    results_path = os.path.join(results_dir, classifier_type + f'_sumary_results_{tag}.csv')
    np.save(train_confusion_path, train_confusion_matrix)
    np.save(test_confusion_path, test_confusion_matrix)

    results = [train_clean, test_clean]
    results_df = pd.DataFrame(results, index=['train', 'test'])

    results_df.to_csv(results_path, index=True, index_label='split')

# ========= runner ========

def main():
    # test trials must have all 4 classes present from the 17-23 range
    # (open_hand isn't in 1-8, pinch/chaka 9-16, chaka missing trial 18)
    train_trials = [t for t in TRIALS if t not in TEST_TRIALS]
    train_feature, test_feature = get_train_test_df( data_path=DATA_PATH,
                                                    train_trials=train_trials,
                                                    test_trials=TEST_TRIALS,
                                                    classes=CLASSES)

    train_cal = apply_baseline_calibration(train_feature)
    test_cal = apply_baseline_calibration(test_feature)


    lda = train_lda(train_cal, store_covariance=True)
    train_centroids = lda.means_.tolist()
    train_priors = lda.priors_.tolist()
    covariance_path = os.path.join(MODEL_PATH, "lda_shared_covariance.npy")
    centroids_path = os.path.join(MODEL_PATH, "train_centroids.json")
    priors_path = os.path.join(MODEL_PATH, "class_priors.json")
    np.save(covariance_path, lda.covariance_)
    with open(centroids_path, "w") as file:
        json.dump(train_centroids, file)
    with open(priors_path, "w") as file:
        json.dump(train_priors, file)

    lda_train_results = evaluate_lda_recalibrated(lda, train_cal)
    lda_test_results = evaluate_lda_recalibrated(lda, test_cal)
    save_performance_metrics(lda_train_results, lda_test_results, CLASSES, classifier_type='lda_recal')

if __name__ == "__main__":
    main()