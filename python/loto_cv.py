import os
import pandas as pd

from train_offline_models import (
    DATA_PATH, CLASSES, FEATURE_COLS, TRIALS,
    get_feature_df, apply_baseline_calibration,
    train_log_reg, train_rf, evaluate_model_detailed,
)

RESULTS_PATH = os.path.join('.', 'results', 'my_data')

TRAINERS = {'log_reg': train_log_reg, 'rf': train_rf}


def run_loto_cv(data_path=DATA_PATH, trials=TRIALS, classes=CLASSES, feature_cols=FEATURE_COLS):
    trial_list = list(trials)
    df = pd.read_csv(data_path)
    # filtering + calibration are per-trial independent, so this is equivalent to
    # recomputing them separately for every fold, just done once for efficiency.
    feature_df = get_feature_df(df, read_trials=trial_list, classes=classes)
    cal_df = apply_baseline_calibration(feature_df)

    fold_rows = []
    for held_out in trial_list:
        train_cal = cal_df[cal_df['trial'] != held_out]
        test_cal = cal_df[cal_df['trial'] == held_out]
        for name, trainer in TRAINERS.items():
            model = trainer(train_cal, feature_cols=feature_cols)
            results = evaluate_model_detailed(model, test_cal, feature_cols=feature_cols)
            fold_rows.append({
                'classifier': name,
                'held_out_trial': held_out,
                'score': results['score'],
                'f1': results['f1'],
                'bacc': results['bacc'],
            })
    return pd.DataFrame(fold_rows)


def main():
    fold_df = run_loto_cv()
    os.makedirs(RESULTS_PATH, exist_ok=True)
    fold_df.to_csv(os.path.join(RESULTS_PATH, 'loto_cv_per_fold.csv'), index=False)

    summary = fold_df.groupby('classifier')[['score', 'f1', 'bacc']].agg(['mean', 'std'])
    summary.to_csv(os.path.join(RESULTS_PATH, 'loto_cv_summary.csv'))

    print(fold_df.to_string(index=False))
    print()
    print(summary)


if __name__ == "__main__":
    main()
