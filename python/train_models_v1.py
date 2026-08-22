#!/usr/bin/env python
# coding: utf-8

# In[73]:


import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn.metrics as metrics

DATA_PATH = csv_path = os.path.join("..", "data", "EMG-data.csv")

df = pd.read_csv(DATA_PATH)
df['subject'].value_counts()

SUBJECT_ROWS = [91350, 103636, 111012, 105175, 100859, 99670, 102134]
TOTAL_ROWS = 713836
CHANNELS = ["channel1", "channel2", "channel3", "channel4"]


# In[ ]:


def get_feature_df(csv_path=DATA_PATH, read_channel='channel1', start_row=0, num_read=SUBJECT_ROWS[0]):
    """Given a path to EMG data csv, returns feature windowed data from specified range, exlcuding specified channels
    omits windows in which all rows do not have the same class"""
    df = pd.read_csv(csv_path)
    df = df.iloc[start_row : start_row + num_read].copy()
    dropped_channels = []
    for i in range(len(CHANNELS)):
        if read_channel != CHANNELS[i]:
            dropped_channels.append(CHANNELS[i])
    df = df.drop(columns=dropped_channels)

    windowed_class = df["class"].rolling(window=200, step=100).apply(lambda w: w.iloc[0])
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
    feature_df = pd.DataFrame({"RMS": windowed_rms_col, "waveform_len": windowed_wfl_col, 
                           "MAV": windowed_mav_col, "max_abs": windowed_max_col,
                           "min_abs": windowed_min_col, "std": windowed_stdev_col,
                           "zero_crossings": windowed_zc_col,
                           'mask': windowed_mask_col,
                           "class": windowed_class,
                           'subject': windowed_subject})
    feature_df = feature_df[feature_df['mask']]
    feature_df = feature_df[feature_df['class'].between(0, 4)]
    return feature_df

feature_cols = ["RMS", "waveform_len", "MAV", "max_abs", "min_abs", "std", "zero_crossings"]

def get_df_features_labels(df, feature_cols=feature_cols):
    features = df[feature_cols].to_numpy()
    labels = df["class"].to_numpy()
    return features, labels

def train_nearest_centroid(df, feature_cols=feature_cols):
    features, labels = get_df_features_labels(df, feature_cols)
    nc = NearestCentroid()
    nc.fit(features, labels)
    return nc

def evaluate_model(model, df, feature_cols=feature_cols):
    features, labels = get_df_features_labels(df, feature_cols)
    preds = model.predict(features)
    score = model.score(features, labels)
    return preds, score

def train_log_reg(df, feature_cols=feature_cols):
    features, labels = get_df_features_labels(df, feature_cols)
    reg = LogisticRegression(max_iter=10000)
    reg.fit(features, labels)
    return reg

def train_lda(df, feature_cols=feature_cols):
    features, labels = get_df_features_labels(df, feature_cols)
    lda = LinearDiscriminantAnalysis()
    lda.fit(features, labels)
    return lda


# In[75]:


test_rows = SUBJECT_ROWS[-1] + SUBJECT_ROWS[-2]
train_rows = TOTAL_ROWS - test_rows
train_df = get_feature_df(read_channel='channel1', start_row=0, num_read=train_rows)
test_df = get_feature_df(read_channel='channel1', start_row=train_rows, num_read=test_rows)

nc = train_nearest_centroid(train_df)
nc_train_preds, nc_train_score = evaluate_model(nc, train_df)
nc_test_preds, nc_test_score = evaluate_model(nc, test_df)

nc_bacc_train = metrics.balanced_accuracy_score(train_df['class'].to_numpy(), nc_train_preds)
nc_bacc_test = metrics.balanced_accuracy_score(test_df['class'].to_numpy(), nc_test_preds)
nc_confusion_train = metrics.confusion_matrix(train_df['class'].to_numpy(), nc_train_preds)
nc_confusion_test = metrics.confusion_matrix(test_df['class'].to_numpy(), nc_test_preds)
nc_f1_train = metrics.f1_score(train_df['class'].to_numpy(), nc_train_preds, average='macro')
nc_f1_test = metrics.f1_score(test_df['class'].to_numpy(), nc_test_preds, average='macro')

print("For channel1: ")
print(f"Nearest centroid train set score: {nc_train_score} \n Nearest centroid test set score:{nc_test_score}")
print(f"Train BACC: {nc_bacc_train} \n Test BACC: {nc_bacc_test}")
print(f"Train confusion matric: {nc_confusion_train} \n Test confusion matrix: {nc_confusion_test}")
print(f"Train f1 score: {nc_f1_train} \n Test f1 score: {nc_f1_test}")

reg = train_log_reg(train_df)
reg_train_preds, reg_train_score = evaluate_model(reg, train_df)
reg_test_preds, reg_test_score = evaluate_model(reg, test_df)

reg_bacc_train = metrics.balanced_accuracy_score(train_df['class'].to_numpy(), reg_train_preds)
reg_bacc_test = metrics.balanced_accuracy_score(test_df['class'].to_numpy(), reg_test_preds)
reg_confusion_train = metrics.confusion_matrix(train_df['class'].to_numpy(), reg_train_preds)
reg_confusion_test = metrics.confusion_matrix(test_df['class'].to_numpy(), reg_test_preds)
reg_f1_train = metrics.f1_score(train_df['class'].to_numpy(), reg_train_preds, average='macro')
reg_f1_test = metrics.f1_score(test_df['class'].to_numpy(), reg_test_preds, average='macro')

print(f"Logistic regression train set score: {reg_train_score} \n Logistic regression test set score:{reg_test_score}")
print(f"Train BACC: {reg_bacc_train} \n Test BACC: {reg_bacc_test}")
print(f"Train confusion matric: {reg_confusion_train} \n Test confusion matrix: {reg_confusion_test}")
print(f"Train f1 score: {reg_f1_train} \n Test f1 score: {reg_f1_test}")

lda = train_lda(train_df)
lda_train_preds, lda_train_score = evaluate_model(lda, train_df)
lda_test_preds, lda_test_score = evaluate_model(lda, test_df)

lda_bacc_train = metrics.balanced_accuracy_score(train_df['class'].to_numpy(), lda_train_preds)
lda_bacc_test = metrics.balanced_accuracy_score(test_df['class'].to_numpy(), lda_test_preds)
lda_confusion_train = metrics.confusion_matrix(train_df['class'].to_numpy(), lda_train_preds)
lda_confusion_test = metrics.confusion_matrix(test_df['class'].to_numpy(), lda_test_preds)
lda_f1_train = metrics.f1_score(train_df['class'].to_numpy(), lda_train_preds, average='macro')
lda_f1_test = metrics.f1_score(test_df['class'].to_numpy(), lda_test_preds, average='macro')

print(f"LDA train set score: {lda_train_score} \n LDA test set score:{lda_test_score}")
print(f"Train BACC: {lda_bacc_train} \n Test BACC: {lda_bacc_test}")
print(f"Train confusion matric: {lda_confusion_train} \n Test confusion matriix: {lda_confusion_test}")
print(f"Train f1 score: {lda_f1_train} \n Test f1 score: {lda_f1_test}")


# In[76]:


test_rows = SUBJECT_ROWS[-1] + SUBJECT_ROWS[-2]
train_rows = TOTAL_ROWS - test_rows
train_df = get_feature_df(read_channel='channel2', start_row=0, num_read=train_rows)
test_df = get_feature_df(read_channel='channel2', start_row=train_rows, num_read=test_rows)

nc = train_nearest_centroid(train_df)
nc_train_preds, nc_train_score = evaluate_model(nc, train_df)
nc_test_preds, nc_test_score = evaluate_model(nc, test_df)

nc_bacc_train = metrics.balanced_accuracy_score(train_df['class'].to_numpy(), nc_train_preds)
nc_bacc_test = metrics.balanced_accuracy_score(test_df['class'].to_numpy(), nc_test_preds)
nc_confusion_train = metrics.confusion_matrix(train_df['class'].to_numpy(), nc_train_preds)
nc_confusion_test = metrics.confusion_matrix(test_df['class'].to_numpy(), nc_test_preds)
nc_f1_train = metrics.f1_score(train_df['class'].to_numpy(), nc_train_preds, average='macro')
nc_f1_test = metrics.f1_score(test_df['class'].to_numpy(), nc_test_preds, average='macro')

print("For channel2: ")
print(f"Nearest centroid train set score: {nc_train_score} \n Nearest centroid test set score:{nc_test_score}")
print(f"Train BACC: {nc_bacc_train} \n Test BACC: {nc_bacc_test}")
print(f"Train confusion matric: {nc_confusion_train} \n Test confusion matrix: {nc_confusion_test}")
print(f"Train f1 score: {nc_f1_train} \n Test f1 score: {nc_f1_test}")

reg = train_log_reg(train_df)
reg_train_preds, reg_train_score = evaluate_model(reg, train_df)
reg_test_preds, reg_test_score = evaluate_model(reg, test_df)

reg_bacc_train = metrics.balanced_accuracy_score(train_df['class'].to_numpy(), reg_train_preds)
reg_bacc_test = metrics.balanced_accuracy_score(test_df['class'].to_numpy(), reg_test_preds)
reg_confusion_train = metrics.confusion_matrix(train_df['class'].to_numpy(), reg_train_preds)
reg_confusion_test = metrics.confusion_matrix(test_df['class'].to_numpy(), reg_test_preds)
reg_f1_train = metrics.f1_score(train_df['class'].to_numpy(), reg_train_preds, average='macro')
reg_f1_test = metrics.f1_score(test_df['class'].to_numpy(), reg_test_preds, average='macro')

print(f"Logistic regression train set score: {reg_train_score} \n Logistic regression test set score:{reg_test_score}")
print(f"Train BACC: {reg_bacc_train} \n Test BACC: {reg_bacc_test}")
print(f"Train confusion matric: {reg_confusion_train} \n Test confusion matrix: {reg_confusion_test}")
print(f"Train f1 score: {reg_f1_train} \n Test f1 score: {reg_f1_test}")

lda = train_lda(train_df)
lda_train_preds, lda_train_score = evaluate_model(lda, train_df)
lda_test_preds, lda_test_score = evaluate_model(lda, test_df)

lda_bacc_train = metrics.balanced_accuracy_score(train_df['class'].to_numpy(), lda_train_preds)
lda_bacc_test = metrics.balanced_accuracy_score(test_df['class'].to_numpy(), lda_test_preds)
lda_confusion_train = metrics.confusion_matrix(train_df['class'].to_numpy(), lda_train_preds)
lda_confusion_test = metrics.confusion_matrix(test_df['class'].to_numpy(), lda_test_preds)
lda_f1_train = metrics.f1_score(train_df['class'].to_numpy(), lda_train_preds, average='macro')
lda_f1_test = metrics.f1_score(test_df['class'].to_numpy(), lda_test_preds, average='macro')

print(f"LDA train set score: {lda_train_score} \n LDA test set score:{lda_test_score}")
print(f"Train BACC: {lda_bacc_train} \n Test BACC: {lda_bacc_test}")
print(f"Train confusion matric: {lda_confusion_train} \n Test confusion matriix: {lda_confusion_test}")
print(f"Train f1 score: {lda_f1_train} \n Test f1 score: {lda_f1_test}")


# In[77]:


test_rows = SUBJECT_ROWS[-1] + SUBJECT_ROWS[-2]
train_rows = TOTAL_ROWS - test_rows
train_df = get_feature_df(read_channel='channel3', start_row=0, num_read=train_rows)
test_df = get_feature_df(read_channel='channel3', start_row=train_rows, num_read=test_rows)

nc = train_nearest_centroid(train_df)
nc_train_preds, nc_train_score = evaluate_model(nc, train_df)
nc_test_preds, nc_test_score = evaluate_model(nc, test_df)

nc_bacc_train = metrics.balanced_accuracy_score(train_df['class'].to_numpy(), nc_train_preds)
nc_bacc_test = metrics.balanced_accuracy_score(test_df['class'].to_numpy(), nc_test_preds)
nc_confusion_train = metrics.confusion_matrix(train_df['class'].to_numpy(), nc_train_preds)
nc_confusion_test = metrics.confusion_matrix(test_df['class'].to_numpy(), nc_test_preds)
nc_f1_train = metrics.f1_score(train_df['class'].to_numpy(), nc_train_preds, average='macro')
nc_f1_test = metrics.f1_score(test_df['class'].to_numpy(), nc_test_preds, average='macro')

print("For channel3: ")
print(f"Nearest centroid train set score: {nc_train_score} \n Nearest centroid test set score:{nc_test_score}")
print(f"Train BACC: {nc_bacc_train} \n Test BACC: {nc_bacc_test}")
print(f"Train confusion matric: {nc_confusion_train} \n Test confusion matrix: {nc_confusion_test}")
print(f"Train f1 score: {nc_f1_train} \n Test f1 score: {nc_f1_test}")

reg = train_log_reg(train_df)
reg_train_preds, reg_train_score = evaluate_model(reg, train_df)
reg_test_preds, reg_test_score = evaluate_model(reg, test_df)

reg_bacc_train = metrics.balanced_accuracy_score(train_df['class'].to_numpy(), reg_train_preds)
reg_bacc_test = metrics.balanced_accuracy_score(test_df['class'].to_numpy(), reg_test_preds)
reg_confusion_train = metrics.confusion_matrix(train_df['class'].to_numpy(), reg_train_preds)
reg_confusion_test = metrics.confusion_matrix(test_df['class'].to_numpy(), reg_test_preds)
reg_f1_train = metrics.f1_score(train_df['class'].to_numpy(), reg_train_preds, average='macro')
reg_f1_test = metrics.f1_score(test_df['class'].to_numpy(), reg_test_preds, average='macro')

print(f"Logistic regression train set score: {reg_train_score} \n Logistic regression test set score:{reg_test_score}")
print(f"Train BACC: {reg_bacc_train} \n Test BACC: {reg_bacc_test}")
print(f"Train confusion matric: {reg_confusion_train} \n Test confusion matrix: {reg_confusion_test}")
print(f"Train f1 score: {reg_f1_train} \n Test f1 score: {reg_f1_test}")

lda = train_lda(train_df)
lda_train_preds, lda_train_score = evaluate_model(lda, train_df)
lda_test_preds, lda_test_score = evaluate_model(lda, test_df)

lda_bacc_train = metrics.balanced_accuracy_score(train_df['class'].to_numpy(), lda_train_preds)
lda_bacc_test = metrics.balanced_accuracy_score(test_df['class'].to_numpy(), lda_test_preds)
lda_confusion_train = metrics.confusion_matrix(train_df['class'].to_numpy(), lda_train_preds)
lda_confusion_test = metrics.confusion_matrix(test_df['class'].to_numpy(), lda_test_preds)
lda_f1_train = metrics.f1_score(train_df['class'].to_numpy(), lda_train_preds, average='macro')
lda_f1_test = metrics.f1_score(test_df['class'].to_numpy(), lda_test_preds, average='macro')

print(f"LDA train set score: {lda_train_score} \n LDA test set score:{lda_test_score}")
print(f"Train BACC: {lda_bacc_train} \n Test BACC: {lda_bacc_test}")
print(f"Train confusion matric: {lda_confusion_train} \n Test confusion matriix: {lda_confusion_test}")
print(f"Train f1 score: {lda_f1_train} \n Test f1 score: {lda_f1_test}")


# In[78]:


test_rows = SUBJECT_ROWS[-1] + SUBJECT_ROWS[-2]
train_rows = TOTAL_ROWS - test_rows
train_df = get_feature_df(read_channel='channel4', start_row=0, num_read=train_rows)
test_df = get_feature_df(read_channel='channel4', start_row=train_rows, num_read=test_rows)

nc = train_nearest_centroid(train_df)
nc_train_preds, nc_train_score = evaluate_model(nc, train_df)
nc_test_preds, nc_test_score = evaluate_model(nc, test_df)

nc_bacc_train = metrics.balanced_accuracy_score(train_df['class'].to_numpy(), nc_train_preds)
nc_bacc_test = metrics.balanced_accuracy_score(test_df['class'].to_numpy(), nc_test_preds)
nc_confusion_train = metrics.confusion_matrix(train_df['class'].to_numpy(), nc_train_preds)
nc_confusion_test = metrics.confusion_matrix(test_df['class'].to_numpy(), nc_test_preds)
nc_f1_train = metrics.f1_score(train_df['class'].to_numpy(), nc_train_preds, average='macro')
nc_f1_test = metrics.f1_score(test_df['class'].to_numpy(), nc_test_preds, average='macro')

print("For channel4: ")
print(f"Nearest centroid train set score: {nc_train_score} \n Nearest centroid test set score:{nc_test_score}")
print(f"Train BACC: {nc_bacc_train} \n Test BACC: {nc_bacc_test}")
print(f"Train confusion matric: {nc_confusion_train} \n Test confusion matrix: {nc_confusion_test}")
print(f"Train f1 score: {nc_f1_train} \n Test f1 score: {nc_f1_test}")

reg = train_log_reg(train_df)
reg_train_preds, reg_train_score = evaluate_model(reg, train_df)
reg_test_preds, reg_test_score = evaluate_model(reg, test_df)

reg_bacc_train = metrics.balanced_accuracy_score(train_df['class'].to_numpy(), reg_train_preds)
reg_bacc_test = metrics.balanced_accuracy_score(test_df['class'].to_numpy(), reg_test_preds)
reg_confusion_train = metrics.confusion_matrix(train_df['class'].to_numpy(), reg_train_preds)
reg_confusion_test = metrics.confusion_matrix(test_df['class'].to_numpy(), reg_test_preds)
reg_f1_train = metrics.f1_score(train_df['class'].to_numpy(), reg_train_preds, average='macro')
reg_f1_test = metrics.f1_score(test_df['class'].to_numpy(), reg_test_preds, average='macro')

print(f"Logistic regression train set score: {reg_train_score} \n Logistic regression test set score:{reg_test_score}")
print(f"Train BACC: {reg_bacc_train} \n Test BACC: {reg_bacc_test}")
print(f"Train confusion matric: {reg_confusion_train} \n Test confusion matrix: {reg_confusion_test}")
print(f"Train f1 score: {reg_f1_train} \n Test f1 score: {reg_f1_test}")

lda = train_lda(train_df)
lda_train_preds, lda_train_score = evaluate_model(lda, train_df)
lda_test_preds, lda_test_score = evaluate_model(lda, test_df)

lda_bacc_train = metrics.balanced_accuracy_score(train_df['class'].to_numpy(), lda_train_preds)
lda_bacc_test = metrics.balanced_accuracy_score(test_df['class'].to_numpy(), lda_test_preds)
lda_confusion_train = metrics.confusion_matrix(train_df['class'].to_numpy(), lda_train_preds)
lda_confusion_test = metrics.confusion_matrix(test_df['class'].to_numpy(), lda_test_preds)
lda_f1_train = metrics.f1_score(train_df['class'].to_numpy(), lda_train_preds, average='macro')
lda_f1_test = metrics.f1_score(test_df['class'].to_numpy(), lda_test_preds, average='macro')

print(f"LDA train set score: {lda_train_score} \n LDA test set score:{lda_test_score}")
print(f"Train BACC: {lda_bacc_train} \n Test BACC: {lda_bacc_test}")
print(f"Train confusion matric: {lda_confusion_train} \n Test confusion matriix: {lda_confusion_test}")
print(f"Train f1 score: {lda_f1_train} \n Test f1 score: {lda_f1_test}")


# In[ ]:


def get_feature_df_mc(csv_path=DATA_PATH, start_row=0, num_read=SUBJECT_ROWS[0]):
    mc_df = None
    for channel in CHANNELS:
        df_ch = get_feature_df(csv_path=csv_path, read_channel=channel, start_row=start_row, num_read=num_read)
        channel_features = df_ch[feature_cols].add_suffix(f"_{channel}")
        if mc_df is None:
            mc_df = channel_features
            mc_df["class"] = df_ch["class"]
            mc_df["subject"] = df_ch["subject"]
        else:
            mc_df = pd.concat([mc_df, channel_features], axis=1)
    return mc_df
feature_cols_mc = [f"{col}_{channel}" for channel in CHANNELS for col in feature_cols]


# In[82]:


test_rows = SUBJECT_ROWS[-1] + SUBJECT_ROWS[-2]
train_rows = TOTAL_ROWS - test_rows
train_df = get_feature_df_mc(start_row=0, num_read=train_rows)
test_df = get_feature_df_mc(start_row=train_rows, num_read=test_rows)

nc = train_nearest_centroid(train_df, feature_cols=feature_cols_mc)
nc_train_preds, nc_train_score = evaluate_model(nc, train_df, feature_cols=feature_cols_mc)
nc_test_preds, nc_test_score = evaluate_model(nc, test_df, feature_cols=feature_cols_mc)

nc_bacc_train = metrics.balanced_accuracy_score(train_df['class'].to_numpy(), nc_train_preds)
nc_bacc_test = metrics.balanced_accuracy_score(test_df['class'].to_numpy(), nc_test_preds)
nc_confusion_train = metrics.confusion_matrix(train_df['class'].to_numpy(), nc_train_preds)
nc_confusion_test = metrics.confusion_matrix(test_df['class'].to_numpy(), nc_test_preds)
nc_f1_train = metrics.f1_score(train_df['class'].to_numpy(), nc_train_preds, average='macro')
nc_f1_test = metrics.f1_score(test_df['class'].to_numpy(), nc_test_preds, average='macro')

print("For all channels: ")
print(f"Nearest centroid train set score: {nc_train_score} \n Nearest centroid test set score:{nc_test_score}")
print(f"Train BACC: {nc_bacc_train} \n Test BACC: {nc_bacc_test}")
print(f"Train confusion matric: {nc_confusion_train} \n Test confusion matrix: {nc_confusion_test}")
print(f"Train f1 score: {nc_f1_train} \n Test f1 score: {nc_f1_test}")

reg = train_log_reg(train_df, feature_cols=feature_cols_mc)
reg_train_preds, reg_train_score = evaluate_model(reg, train_df, feature_cols=feature_cols_mc)
reg_test_preds, reg_test_score = evaluate_model(reg, test_df, feature_cols=feature_cols_mc)

reg_bacc_train = metrics.balanced_accuracy_score(train_df['class'].to_numpy(), reg_train_preds)
reg_bacc_test = metrics.balanced_accuracy_score(test_df['class'].to_numpy(), reg_test_preds)
reg_confusion_train = metrics.confusion_matrix(train_df['class'].to_numpy(), reg_train_preds)
reg_confusion_test = metrics.confusion_matrix(test_df['class'].to_numpy(), reg_test_preds)
reg_f1_train = metrics.f1_score(train_df['class'].to_numpy(), reg_train_preds, average='macro')
reg_f1_test = metrics.f1_score(test_df['class'].to_numpy(), reg_test_preds, average='macro')

print(f"Logistic regression train set score: {reg_train_score} \n Logistic regression test set score:{reg_test_score}")
print(f"Train BACC: {reg_bacc_train} \n Test BACC: {reg_bacc_test}")
print(f"Train confusion matric: {reg_confusion_train} \n Test confusion matrix: {reg_confusion_test}")
print(f"Train f1 score: {reg_f1_train} \n Test f1 score: {reg_f1_test}")

lda = train_lda(train_df, feature_cols=feature_cols_mc)
lda_train_preds, lda_train_score = evaluate_model(lda, train_df, feature_cols=feature_cols_mc)
lda_test_preds, lda_test_score = evaluate_model(lda, test_df, feature_cols=feature_cols_mc)

lda_bacc_train = metrics.balanced_accuracy_score(train_df['class'].to_numpy(), lda_train_preds)
lda_bacc_test = metrics.balanced_accuracy_score(test_df['class'].to_numpy(), lda_test_preds)
lda_confusion_train = metrics.confusion_matrix(train_df['class'].to_numpy(), lda_train_preds)
lda_confusion_test = metrics.confusion_matrix(test_df['class'].to_numpy(), lda_test_preds)
lda_f1_train = metrics.f1_score(train_df['class'].to_numpy(), lda_train_preds, average='macro')
lda_f1_test = metrics.f1_score(test_df['class'].to_numpy(), lda_test_preds, average='macro')

print(f"LDA train set score: {lda_train_score} \n LDA test set score:{lda_test_score}")
print(f"Train BACC: {lda_bacc_train} \n Test BACC: {lda_bacc_test}")
print(f"Train confusion matric: {lda_confusion_train} \n Test confusion matriix: {lda_confusion_test}")
print(f"Train f1 score: {lda_f1_train} \n Test f1 score: {lda_f1_test}")


# In[ ]:




