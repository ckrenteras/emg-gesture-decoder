import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn.metrics as metrics
from scipy.signal import butter, sosfilt


DATA_PATH = csv_path = os.path.join("..", "data", "EMG-data.csv")
RESULTS_PATH = os.path.join('.', 'results')

# ====== dataset consts =======
SUBJECT_ROWS = [99980, 103636, 111012, 105175, 100859, 99670, 102134, 100225]
TOTAL_ROWS = sum(SUBJECT_ROWS)
CHANNELS = ["channel1", "channel2", "channel3", "channel4"]
SAMPLE_RATE = 1000


#=======  model constants =======
FEATURE_COLS = ["RMS", "waveform_len", "MAV", "max_abs", "min_abs", "std", "zero_crossings"]
FEATURE_COLS_MC = [f"{col}_{channel}" for channel in CHANNELS for col in FEATURE_COLS]
CHANNELS_FILTERED = [c + '_filtered' for c in CHANNELS]
CHANNELS_NORMALIZED = [c + '_normalized' for c in CHANNELS]

AMPLITUDE_FEATURE_COLS = ["RMS", "MAV", "max_abs", "min_abs", "std"]
AMPLITUDE_FEATURE_COLS_MC = [f"{col}_{channel}" for channel in CHANNELS for col in AMPLITUDE_FEATURE_COLS]


# ======= pre processing consts =====
LOW_PASS_FREQ = 500
HIGH_PASS_FREQ = 30

# ===== training consts =============
SUBJECT_SEVEN_START = sum(SUBJECT_ROWS[0:6])
LAST_TWO_NUM_ROWS = SUBJECT_ROWS[-1] + SUBJECT_ROWS[-2]

SUBJECT_SEVEN_START = sum(SUBJECT_ROWS[0:6])
LAST_TWO_NUM_ROWS = SUBJECT_ROWS[-1] + SUBJECT_ROWS[-2]

#======== pre processing ============

def butterworth_filter(data, order=3, cutoff=HIGH_PASS_FREQ, fs=SAMPLE_RATE, filter_type='highpass'):
    if filter_type not in ['lowpass', 'highpass', 'bandpass', 'bandstop']:
        raise ValueError("Invalid filter type requested")
    sos = butter(N=order, Wn=cutoff, fs=SAMPLE_RATE, btype=filter_type, output='sos')
    filtered_data = sosfilt(sos, data)
    return filtered_data
    
def filter_emg(csv_path=DATA_PATH, cutoff=HIGH_PASS_FREQ, fs=SAMPLE_RATE, butter_order=3, filter_type='highpass'):
    df = pd.read_csv(csv_path)
    df[CHANNELS_FILTERED] = df[CHANNELS].apply(lambda x: butterworth_filter(x, order=butter_order, cutoff=cutoff, fs=fs, filter_type=filter_type))
    pre_processed_df = pd.DataFrame()
    pre_processed_df[CHANNELS] = df[CHANNELS_FILTERED]
    pre_processed_df['class'] = df['class']
    pre_processed_df['subject'] = df['subject']
    return pre_processed_df

# ============== feature extraction =============

def get_feature_df(df, read_channel='channel3', start_row=0, num_read=SUBJECT_ROWS[0]):
    """Given a path to EMG df with channel, gesture, class, subject cols, returns feature windowed data from specified 
    range, exlcuding specified channels omits windows in which all rows do not have the same class"""
    df = df.iloc[start_row : start_row + num_read].copy()
    dropped_channels = []
    for i in range(len(CHANNELS)):
        if read_channel != CHANNELS[i]:
            dropped_channels.append(CHANNELS[i])
    df = df.drop(columns=dropped_channels)

    windowed_class = df["class"].rolling(window=200, step=100).apply(lambda w: w.iloc[0])
    windowed_class = windowed_class 
    windowed_subject = df["subject"].rolling(window=200, step=100).apply(lambda w: w.iloc[0])
    windowed_rms_col = df[read_channel].pow(2).rolling(window=200, step=100).mean().pow(0.5)
    windowed_wfl_col = df[read_channel].rolling(window=200, step=100).apply(lambda x: np.abs(np.diff(x)).sum())
    windowed_stdev_col = df[read_channel].rolling(window=200, step=100).std()
    windowed_mav_col = df[read_channel].rolling(window=200, step=100).apply(lambda x: np.abs(x).mean())
    windowed_min_col = df[read_channel].rolling(window=200, step=100).apply(lambda x: np.abs(x).min())
    def crossings(s): 
        return (s.shift(1) * s < 0)
    windowed_zc_col = crossings(df[read_channel]).rolling(window=200, step=100).sum()
    windowed_max_col = df[read_channel].rolling(window=200, step=100).apply(lambda x: np.abs(x).max())
    
    rolling_class = df["class"].rolling(window=200, step=100)
    windowed_mask_col = rolling_class.min() == rolling_class.max()
    feature_df = pd.DataFrame({"RMS": windowed_rms_col,
                               "waveform_len": windowed_wfl_col,
                               "MAV": windowed_mav_col,
                               "max_abs": windowed_max_col,
                               "min_abs": windowed_min_col,
                               "std": windowed_stdev_col,
                               "zero_crossings": windowed_zc_col,
                               'mask': windowed_mask_col,
                               "class": windowed_class,
                               'subject': windowed_subject})
    feature_df = feature_df[feature_df['mask']]
    return feature_df

def get_feature_df_mc(df, start_row=0, num_read=SUBJECT_ROWS[0]):
    mc_df = None
    for channel in CHANNELS:
        df_ch = get_feature_df(df, read_channel=channel, start_row=start_row, num_read=num_read)
        channel_features = df_ch[FEATURE_COLS].add_suffix(f"_{channel}")
        if mc_df is None:
            mc_df = channel_features
            mc_df["class"] = df_ch["class"]
            mc_df["subject"] = df_ch["subject"]
        else:
            mc_df = pd.concat([mc_df, channel_features], axis=1)
    return mc_df

def get_train_test_df(whole_df, read_channel='channel3', train_start=0, train_read=sum(SUBJECT_ROWS[0:5]),
                  test_start=SUBJECT_SEVEN_START, test_read=LAST_TWO_NUM_ROWS):
    if (train_start + train_read > TOTAL_ROWS):
        raise ValueError('Requested train rows read out of bounds')
    if (test_start + test_read > TOTAL_ROWS):
        raise ValueError('Requested test rows read out of bounds')
    if (train_start > test_start and train_start < test_start + test_read):
        raise ValueError('Leakage in train and test sets')
    if (train_start + train_read > test_start and train_start + train_read < test_start + test_read):
            raise ValueError('Leakage in train and test sets')

    train_df = get_feature_df(whole_df,read_channel=read_channel, start_row=train_start, num_read=train_read)
    test_df = get_feature_df(whole_df, read_channel=read_channel, start_row=test_start, num_read=test_read)
    return train_df, test_df

def get_train_test_mc(whole_df, train_start=0, train_read=sum(SUBJECT_ROWS[0:5]),
                  test_start=SUBJECT_SEVEN_START, test_read=LAST_TWO_NUM_ROWS):
    if (train_start + train_read > TOTAL_ROWS):
        raise ValueError('Requested train rows read out of bounds')
    if (test_start + test_read > TOTAL_ROWS):
        raise ValueError('Requested test rows read out of bounds')
    if (train_start > test_start and train_start < test_start + test_read):
        raise ValueError('Leakage in train and test sets')
    if (train_start + train_read > test_start and train_start + train_read < test_start + test_read):
            raise ValueError('Leakage in train and test sets')

    train_df = get_feature_df_mc(whole_df, start_row=train_start, num_read=train_read)
    test_df = get_feature_df_mc(whole_df, start_row=test_start, num_read=test_read)
    return train_df, test_df

# ============== baseline calibration ============

def get_subject_rest_stats(feature_df, amplitude_cols, rest_class=0, max_samples=5000):
    """Per-subject mean and std of each amplitude feature during rest (class 0)
    irl will be a short calibration period."""
    if max_samples is not None:
        rest_df = feature_df[feature_df['class'] == rest_class].head(max_samples)
    else:
        rest_df = feature_df[feature_df['class'] == rest_class]
    rest_mean = rest_df.groupby('subject')[amplitude_cols].mean()
    rest_std = rest_df.groupby('subject')[amplitude_cols].std()
    return rest_mean, rest_std

def apply_baseline_calibration(feature_df, amplitude_cols, rest_class=0, max_samples=5000, eps=1e-8):
    """Z-score each subject's amplitude features against that subject's own rest distribution."""
    rest_mean, rest_std = get_subject_rest_stats(feature_df, amplitude_cols, rest_class=rest_class, max_samples=max_samples)
    calibrated_df = feature_df.copy()
    for subject in rest_mean.index:
        mask = calibrated_df['subject'] == subject
        calibrated_df.loc[mask, amplitude_cols] = (
            (calibrated_df.loc[mask, amplitude_cols] - rest_mean.loc[subject])
            / (rest_std.loc[subject] + eps)
        )
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

def save_performance_metrics(train_results, test_results, channel, results_dir=RESULTS_PATH):
    os.makedirs(results_dir, exist_ok=True)
    if channel in range(0, 5):
        tag = f'ch{channel}'
    elif channel == 'mc':
        tag = channel
    else:
        raise ValueError('invalid channel')
    train_clean = train_results.copy()
    test_clean = test_results.copy()
    del train_clean['preds']
    del test_clean['preds']
    train_confusion_matrix = train_clean.pop('confusion_matrix', None)
    test_confusion_matrix = test_clean.pop('confusion_matrix', None)
    train_confusion_path = os.path.join(results_dir, 'train_confusion_matrix_' + tag + '.npy')
    test_confusion_path = os.path.join(results_dir, 'test_confusion_matrix_' + tag + '.npy')
    np.save(train_confusion_path, train_confusion_matrix)
    np.save(test_confusion_path, test_confusion_matrix)

    results = [train_clean, test_clean]
    results_df = pd.DataFrame(results, index=['train', 'test'])
    results_path = os.path.join(results_dir, f'sumary_results_{tag}.csv')
    results_df.to_csv(results_path, index=True, index_label='split')

# ========= runner ========

def main():
    filtered_df = filter_emg()
    #======== single channel ========
    # use channel 3 as it had the best performance
    # right now using about a 75%/25% train test split with the last 2 subjects serving as test set
    train_feature_df, test_feature_df = get_train_test_df(filtered_df, read_channel='channel3', train_start=0, train_read=SUBJECT_SEVEN_START,
                  test_start=SUBJECT_SEVEN_START, test_read=LAST_TWO_NUM_ROWS)

    train_features_cal = apply_baseline_calibration(train_feature_df, AMPLITUDE_FEATURE_COLS)
    test_features_cal = apply_baseline_calibration(test_feature_df, AMPLITUDE_FEATURE_COLS)

    reg_sc = train_log_reg(train_features_cal)
    train_results = evaluate_model_detailed(reg_sc, train_features_cal)
    test_results = evaluate_model_detailed(reg_sc, test_features_cal)
    save_performance_metrics(train_results, test_results, 3)

    # ====== multi-channel =======
    train_features_mc, test_features_mc = get_train_test_mc(filtered_df, 
                                                            train_start=0, 
                                                            train_read=SUBJECT_SEVEN_START,
                                                            test_start=SUBJECT_SEVEN_START, 
                                                            test_read=LAST_TWO_NUM_ROWS)
    # no calibration on multi-channel, made performance worse
    reg_mc = train_log_reg(train_features_mc, feature_cols=FEATURE_COLS_MC)
    train_results_mc = evaluate_model_detailed(reg_mc, train_features_mc, feature_cols=FEATURE_COLS_MC)
    test_results_mc = evaluate_model_detailed(reg_mc, test_features_mc, feature_cols=FEATURE_COLS_MC)
    save_performance_metrics(train_results_mc, test_results_mc, 'mc')
    print("Done!")

if __name__ == "__main__":
    main()



