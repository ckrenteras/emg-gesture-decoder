"""Plot filtered EMG signal across a whole trial, colored by class, to visually
compare 'clean' trials against trials suspected of having a bad rest baseline.

Usage:
    python plot_trial_signals.py --trials 3 9
    python plot_trial_signals.py --trials 1 3 6 7 9 --out results/my_data/trial_signals.png
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_offline_models import (
    DATA_PATH, CLASS_MAPPING, WINDOW, STEP,
    adc_to_volts, filter_emg, LOW_PASS_FREQ, HIGH_PASS_FREQ, SAMPLE_RATE,
)

CLASS_COLORS = {0: "#dddddd", 1: "#a6cee3", 2: "#b2df8a", 4: "#fb9a99"}


def build_trial_signal(df, trial):
    trial_df = df[df["trial"] == trial].reset_index(drop=True).copy()
    trial_df["emg"] = trial_df["adc"].apply(adc_to_volts)
    low_passed = filter_emg(trial_df, cutoff=LOW_PASS_FREQ, fs=SAMPLE_RATE, filter_type="lowpass")
    filtered = filter_emg(low_passed, cutoff=HIGH_PASS_FREQ, fs=SAMPLE_RATE, filter_type="highpass")
    filtered["rms_envelope"] = (
        filtered["emg"].pow(2).rolling(window=WINDOW, step=1, min_periods=WINDOW).mean().pow(0.5)
    )
    filtered["rms_envelope"] = filtered["rms_envelope"].bfill()
    # NOTE: 'time_ms'/'sample' reset to 0 at the start of every class segment
    # within a trial, so they can't be used as a continuous x-axis here.
    # Row order within a trial is chronological, so build time from that instead.
    filtered["t_sec"] = np.arange(len(filtered)) / SAMPLE_RATE

    # drop the filter warm-up transient (sosfilt starts from zero initial state,
    # producing a large spike at t=0 that swamps the y-scale otherwise)
    settle_samples = int(0.5 * SAMPLE_RATE)
    filtered = filtered.iloc[settle_samples:].reset_index(drop=True)
    return filtered


def plot_trials(df, trials, out_path):
    fig, axes = plt.subplots(len(trials), 1, figsize=(14, 3 * len(trials)), sharex=False)
    if len(trials) == 1:
        axes = [axes]

    for ax, trial in zip(axes, trials):
        sig = build_trial_signal(df, trial)

        for cls, block in sig.groupby((sig["class"] != sig["class"].shift()).cumsum()):
            cls_val = block["class"].iloc[0]
            ax.axvspan(block["t_sec"].iloc[0], block["t_sec"].iloc[-1],
                       color=CLASS_COLORS.get(cls_val, "#ffffff"), alpha=0.5, lw=0)

        ax.plot(sig["t_sec"], sig["emg"], color="#333333", linewidth=0.3, alpha=0.7)
        ax.plot(sig["t_sec"], sig["rms_envelope"], color="black", linewidth=1.2)

        rest_rms = sig.loc[sig["class"] == 0, "rms_envelope"].mean()
        ax.axhline(rest_rms, color="red", linestyle="--", linewidth=1,
                   label=f"mean rest RMS = {rest_rms:.4f}")

        y_lim = np.percentile(np.abs(sig["emg"]), 99.9) * 1.3
        ax.set_ylim(-y_lim, y_lim)

        ax.set_title(f"Trial {trial}")
        ax.set_ylabel("filtered EMG (V)")
        ax.legend(loc="upper right", fontsize=8)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.5) for c in CLASS_COLORS.values()]
    labels = [CLASS_MAPPING[k] for k in CLASS_COLORS]
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, nargs="+", default=[10, 14])
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--out", default=os.path.join("results", "my_data", "trial_signal_comparison.png"))
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    plot_trials(df, args.trials, args.out)


if __name__ == "__main__":
    main()
