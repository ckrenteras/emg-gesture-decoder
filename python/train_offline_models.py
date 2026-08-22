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

# ======= pre processing consts =====
LOW_PASS_FREQ = 500
HIGH_PASS_FREQ = 20
SAMPLE_RATE = 2000 # Hz
WINDOW = 400
STEP = 200

# ====== dataset consts =======
TRIALS = range(1, 26) 
CLASSES = [0, 1, 2, 4]
BITS = 10
VREF = 3.7

#=======  model constants =======
FEATURE_COLS = ["RMS", "waveform_len", "MAV", "max_abs", "min_abs", "std", 'zero_crossings']
AMPLITUDE_FEATURE_COLS = ["RMS", "MAV", "max_abs", "min_abs", "std", "waveform_len", "zero_crossings"]
CALIBRATION_Z_CLIP = 32  
CLASS_MAPPING = {
    0: "rest",
    1: "open_hand",
    2: "fist",
    4: "index"
}

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
    def crossings(s): 
        return (s.shift(1) * s < 0)
    windowed_zc_col = crossings(df['emg']).rolling(window=WINDOW, step=STEP).sum()
    
    rolling_class = df["class"].rolling(window=WINDOW, step=STEP)
    windowed_mask_mixed_col = rolling_class.min() == rolling_class.max()
    rolling_trial = df["trial"].rolling(window=WINDOW, step=STEP)
    windowed_mask_mixed_trial = rolling_trial.min() == rolling_trial.max()
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
                               'mask_mixed_trial': windowed_mask_mixed_trial})
    feature_df = feature_df[feature_df['mask_mixed_col']]
    feature_df = feature_df[feature_df['mask_mixed_trial']]

    # drop each trial's very first window because Butterworth filter has
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

def apply_baseline_calibration(feature_df, amplitude_cols=AMPLITUDE_FEATURE_COLS, rest_class=0,
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

def evaluate_model(model, df, feature_cols=FEATURE_COLS):
    features, labels = get_df_features_labels(df, feature_cols)
    preds = model.predict(features)
    score = model.score(features, labels)
    return preds, score

def train_log_reg(df, feature_cols=FEATURE_COLS):
    features, labels = get_df_features_labels(df, feature_cols)
    reg = LogisticRegression(max_iter=10000)
    reg.fit(features, labels)
    return reg

def train_rf(df, feature_cols=FEATURE_COLS):
     features, labels = get_df_features_labels(df, feature_cols)
     rf = RandomForestClassifier(n_estimators=100, max_depth=10)
     rf.fit(features, labels)
     return rf

def train_lda(df, feature_cols=FEATURE_COLS):
    features, labels = get_df_features_labels(df, feature_cols)
    lda = LinearDiscriminantAnalysis()
    lda.fit(features, labels)
    return lda

def evaluate_model_detailed(model, df, feature_cols=FEATURE_COLS, f1_average='macro'):
    preds, score = evaluate_model(model, df, feature_cols=feature_cols)
    labels = df['class'].to_numpy()

    f1 = metrics.f1_score(labels, preds, average=f1_average)
    bacc = metrics.balanced_accuracy_score(labels, preds)
    confusion_matrix = metrics.confusion_matrix(labels, preds)

    results = {
        'preds': preds,
        'score': score,
        'f1': f1,
        'bacc': bacc,
        'confusion_matrix': confusion_matrix
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
    if (classifier_type == 'rf'):
            train_confusion_path = os.path.join(results_dir, 'rf_train_confusion_matrix_' + tag + '.npy')
            test_confusion_path = os.path.join(results_dir, 'rf_test_confusion_matrix_' + tag + '.npy')
            results_path = os.path.join(results_dir, f'rf_sumary_results_{tag}.csv')
    else: 
            train_confusion_path = os.path.join(results_dir, 'train_confusion_matrix_' + tag + '.npy')
            test_confusion_path = os.path.join(results_dir, 'test_confusion_matrix_' + tag + '.npy')
            results_path = os.path.join(results_dir, f'sumary_results_{tag}.csv')
    np.save(train_confusion_path, train_confusion_matrix)
    np.save(test_confusion_path, test_confusion_matrix)

    results = [train_clean, test_clean]
    results_df = pd.DataFrame(results, index=['train', 'test'])

    results_df.to_csv(results_path, index=True, index_label='split')

# ========= runner ========

def main():
    test_trials = [10, 14, 17, 21, 24]
    train_trials = [t for t in TRIALS if t not in test_trials]
    train_feature, test_feature = get_train_test_df( data_path=DATA_PATH,
                                                    train_trials=train_trials,
                                                    test_trials=test_trials,
                                                    classes=CLASSES)

    train_cal = apply_baseline_calibration(train_feature)
    test_cal = apply_baseline_calibration(test_feature)

    reg = train_log_reg(train_cal)
    train_results = evaluate_model_detailed(reg, train_cal)
    test_results = evaluate_model_detailed(reg, test_cal)
    save_performance_metrics(train_results, test_results, CLASSES)


    model_path = os.path.join("..", "models", "v1", "log_red_v1.pkl")
    v1_path = os.path.dirname(model_path)
    class_mapping_path = os.path.join(v1_path, "class_mapping.json")
    features_path = os.path.join(v1_path, "feature_columns.json")
    amplitude_features_path = os.path.join(v1_path, "amp_feature_columns.json")
    os.makedirs(v1_path, exist_ok=True)
    with open(model_path, "wb") as file:
        pickle.dump(reg, file)
    with open(class_mapping_path, "w") as file:
            json.dump(CLASS_MAPPING, file)
    with open(features_path, "w") as file:
                json.dump(FEATURE_COLS, file)
    with open(amplitude_features_path, "w") as file:
                json.dump(AMPLITUDE_FEATURE_COLS, file)

    # ===== lil rf run ======
    rf = train_rf(train_cal)
    rf_train_results = evaluate_model_detailed(rf, train_cal)
    rf_test_results = evaluate_model_detailed(rf, test_cal)
    save_performance_metrics(rf_train_results, rf_test_results, CLASSES, classifier_type='rf')


    rf_model_path = os.path.join("..", "models", "v1", "rf_v1.pkl")
    with open(rf_model_path, "wb") as file:
        pickle.dump(rf, file)
    


if __name__ == "__main__":
    main()