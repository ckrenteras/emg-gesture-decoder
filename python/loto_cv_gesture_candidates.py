import os
import pandas as pd
import sklearn.metrics as metrics
import numpy as np

from train_offline_models import (
    FEATURE_COLS,
    get_feature_df, apply_baseline_calibration,
    train_rf, evaluate_model_detailed, get_df_features_labels,
)

DATA_DIR = os.path.join("..", "data", "my_data", "subject_one", "combined_subject_one.csv")
CLASS_CODE = {"rest": 0, "open_hand": 1, "pinch": 3, "index": 4, "rock_you": 5, "chaka": 6}
RESULTS_DIR = os.path.join("..", "results", "my_data", "gesture_candidates")


def run_loto(gesture_trials, label, data_dir=DATA_DIR, results_dir=RESULTS_DIR):
    classes = [CLASS_CODE[g] for g in gesture_trials]
    all_trials = sorted(set(t for trials in gesture_trials.values() for t in trials))
    df = pd.read_csv(data_dir)
    feature_df = get_feature_df(df, read_trials=all_trials, classes=classes)
    cal_df = apply_baseline_calibration(feature_df)

    fold_rows = []
    all_true, all_pred = [], []
    for held_out in all_trials:
        train_cal = cal_df[cal_df["trial"] != held_out]
        test_cal = cal_df[cal_df["trial"] == held_out]
        if test_cal.empty:
            continue
        model = train_rf(train_cal, feature_cols=FEATURE_COLS)
        results = evaluate_model_detailed(model, test_cal, feature_cols=FEATURE_COLS)
        fold_rows.append({"held_out_trial": held_out,
                           "score": results["score"], "f1": results["f1"], "bacc": results["bacc"]})
        features, labels = get_df_features_labels(test_cal)
        preds = model.predict(features)
        all_true.extend(labels.tolist())
        all_pred.extend(preds.tolist())

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(results_dir, index=False)

    labels_sorted = sorted(classes)
    cm = metrics.confusion_matrix(all_true, all_pred, labels=labels_sorted)
    np.save(os.join(results_dir, 'confusion_matrices.npy'), cm)



def main():
    # rest/open_hand/pinch use the same set of trials
    rest = list(range(1, 24))
    open_hand_relaxed = list(range(9, 24))
    pinch = list(range(1, 9)) + list(range(17, 24))

    run_loto({
        "rest": rest,
        "open_hand": open_hand_relaxed,
        "pinch": pinch,
        "chaka": list(range(1, 9)) + [17, 19, 20, 21, 22, 23],  # trial 18 missing
    }, "rest, open_hand(relaxed), pinch, chaka")

    run_loto({
        "rest": rest,
        "open_hand": open_hand_relaxed,
        "pinch": pinch,
        "rock_you": list(range(1, 9)) + list(range(17, 21)),
    }, "rest, open_hand(relaxed), pinch, rock_you")


if __name__ == "__main__":
    main()
