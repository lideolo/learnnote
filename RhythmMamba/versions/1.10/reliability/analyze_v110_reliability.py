import argparse
import csv
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import get_config
from dataset import data_loader
from evaluation.post_process import calculate_metric_per_video
from neural_methods.trainer.RhythmMambaTrainer import RhythmMambaTrainer
from main import build_data_loader, general_generator, set_random_seed


CONFIGS = {
    "v102": "configs/train_configs/intra/2UBFC-rPPG_RHYTHMMAMBA_AUGFIX_V102_TEST.yaml",
    "v105": "configs/train_configs/intra/2UBFC-rPPG_RHYTHMMAMBA_LOSSOPT_V105_TIME05_EPOCH21_TEST.yaml",
    "v110": "configs/train_configs/intra/2UBFC-rPPG_RHYTHMMAMBA_ROISGPM_WARM_LR1E6_V110_EPOCH2_TEST.yaml",
}


def make_config(config_file, tta_flip=False):
    args = SimpleNamespace(config_file=config_file, cached_path=None, preprocess=None)
    config = get_config(args)
    config.defrost()
    config.INFERENCE.TTA.HORIZONTAL_FLIP = tta_flip
    config.freeze()
    return config


def make_test_loader(config):
    if config.TEST.DATA.DATASET not in ["UBFC", "UBFC-rPPG"]:
        raise ValueError("This analysis script currently expects UBFC-rPPG configs.")
    test_data = data_loader.UBFCrPPGLoader.UBFCrPPGLoader(
        name="test",
        data_path=config.TEST.DATA.DATA_PATH,
        config_data=config.TEST.DATA,
    )
    return build_data_loader(
        dataset=test_data,
        batch_size=config.INFERENCE.BATCH_SIZE,
        shuffle=False,
        generator=general_generator,
        config=config,
        split_name="test",
    )


def reform_chunks(chunks):
    sorted_chunks = sorted(chunks.items(), key=lambda x: x[0])
    return np.reshape(torch.cat([item[1] for item in sorted_chunks], dim=0).cpu().numpy(), (-1))


def metric_rows(predictions, labels, fs, diff_flag, window_seconds=None):
    rows = []
    for subject in sorted(predictions.keys()):
        prediction = reform_chunks(predictions[subject])
        label = reform_chunks(labels[subject])
        if window_seconds is None:
            window_size = len(prediction)
        else:
            window_size = min(len(prediction), int(window_seconds * fs))
        for start in range(0, len(prediction), window_size):
            pred_window = prediction[start:start + window_size]
            label_window = label[start:start + window_size]
            if len(pred_window) < 9:
                continue
            gt_hr, pred_hr, snr = calculate_metric_per_video(
                pred_window,
                label_window,
                fs=fs,
                diff_flag=diff_flag,
                hr_method="FFT",
            )
            rows.append({
                "subject": subject,
                "start": start,
                "frames": len(pred_window),
                "gt_hr": float(gt_hr),
                "pred_hr": float(pred_hr),
                "abs_error": float(abs(pred_hr - gt_hr)),
                "mape": float(abs((pred_hr - gt_hr) / gt_hr) * 100),
                "snr": float(snr),
            })
    return rows


def summarize(rows):
    gt = np.array([row["gt_hr"] for row in rows])
    pred = np.array([row["pred_hr"] for row in rows])
    snr = np.array([row["snr"] for row in rows])
    return {
        "n": int(len(rows)),
        "mae": float(np.mean(np.abs(pred - gt))),
        "rmse": float(np.sqrt(np.mean(np.square(pred - gt)))),
        "mape": float(np.mean(np.abs((pred - gt) / gt)) * 100),
        "pearson": float(np.corrcoef(pred, gt)[0][1]),
        "snr": float(np.mean(snr)),
    }


def evaluate(config_file, out_dir, tag, tta_flip=False):
    config = make_config(config_file, tta_flip=tta_flip)
    set_random_seed(config.TRAIN.SEED)
    loader = make_test_loader(config)
    trainer = RhythmMambaTrainer(config, {"train": None, "valid": None, "test": loader})
    checkpoint = torch.load(config.INFERENCE.MODEL_PATH, map_location=trainer.device)
    trainer.model.load_state_dict(checkpoint)
    trainer.model.eval()

    predictions = {}
    labels = {}
    with torch.no_grad():
        for batch in loader:
            batch_size = batch[0].shape[0]
            data_test = batch[0].to(trainer.device, non_blocking=True)
            labels_test = batch[1].to(trainer.device, non_blocking=True)
            pred_ppg = trainer._predict_ppg(data_test)
            trainer._store_batch_predictions(
                predictions, labels, pred_ppg, labels_test, batch, batch_size, trainer.chunk_len)

    diff_flag = config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized"
    full_rows = metric_rows(predictions, labels, config.TEST.DATA.FS, diff_flag)
    win_rows = metric_rows(predictions, labels, config.TEST.DATA.FS, diff_flag, window_seconds=10)

    os.makedirs(out_dir, exist_ok=True)
    write_rows(os.path.join(out_dir, f"{tag}_per_video.csv"), full_rows)
    write_rows(os.path.join(out_dir, f"{tag}_window10s.csv"), win_rows)

    return {
        "tag": tag,
        "config": config_file,
        "model_path": config.INFERENCE.MODEL_PATH,
        "full": summarize(full_rows),
        "window10s": summarize(win_rows),
        "predictions": predictions,
        "labels": labels,
        "full_rows": full_rows,
    }


def write_rows(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compare_models(a, b):
    rows = []
    subjects = sorted(set(a["predictions"].keys()) & set(b["predictions"].keys()))
    a_rows = {row["subject"]: row for row in a["full_rows"]}
    b_rows = {row["subject"]: row for row in b["full_rows"]}
    for subject in subjects:
        pa = reform_chunks(a["predictions"][subject])
        pb = reform_chunks(b["predictions"][subject])
        length = min(len(pa), len(pb))
        pa = pa[:length]
        pb = pb[:length]
        corr = np.corrcoef(pa, pb)[0][1]
        rows.append({
            "subject": subject,
            "pred_hr_a": a_rows[subject]["pred_hr"],
            "pred_hr_b": b_rows[subject]["pred_hr"],
            "same_hr": bool(np.isclose(a_rows[subject]["pred_hr"], b_rows[subject]["pred_hr"])),
            "gt_hr_a": a_rows[subject]["gt_hr"],
            "gt_hr_b": b_rows[subject]["gt_hr"],
            "snr_a": a_rows[subject]["snr"],
            "snr_b": b_rows[subject]["snr"],
            "wave_corr": float(corr),
            "wave_l2": float(np.sqrt(np.mean(np.square(pa - pb)))),
            "wave_max_abs": float(np.max(np.abs(pa - pb))),
        })
    return rows


def write_summary(path, results, comparisons):
    with open(path, "w") as f:
        f.write("# V1.10 Reliability Analysis\n\n")
        f.write("## Aggregate Metrics\n\n")
        f.write("| tag | mode | n | MAE | RMSE | MAPE | Pearson | SNR |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for result in results:
            for mode in ["full", "window10s"]:
                m = result[mode]
                f.write(
                    f"| {result['tag']} | {mode} | {m['n']} | {m['mae']:.12f} | "
                    f"{m['rmse']:.12f} | {m['mape']:.12f} | {m['pearson']:.12f} | {m['snr']:.12f} |\n"
                )
        f.write("\n## Pairwise HR Equality\n\n")
        f.write("| pair | subjects | same_hr_subjects | mean_wave_corr | mean_l2 | mean_snr_delta |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for name, rows in comparisons.items():
            same = sum(1 for row in rows if row["same_hr"])
            mean_corr = np.mean([row["wave_corr"] for row in rows])
            mean_l2 = np.mean([row["wave_l2"] for row in rows])
            mean_snr_delta = np.mean([row["snr_b"] - row["snr_a"] for row in rows])
            f.write(
                f"| {name} | {len(rows)} | {same} | {mean_corr:.12f} | "
                f"{mean_l2:.12f} | {mean_snr_delta:.12f} |\n"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="/root/markdown文件/v110_reliability")
    parser.add_argument("--repeat_v110", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = []
    for tag, config_file in CONFIGS.items():
        results.append(evaluate(config_file, args.out_dir, tag))

    results.append(evaluate(CONFIGS["v110"], args.out_dir, "v110_flip_tta", tta_flip=True))
    for i in range(args.repeat_v110):
        results.append(evaluate(CONFIGS["v110"], args.out_dir, f"v110_repeat{i + 1}"))

    by_tag = {result["tag"]: result for result in results}
    comparisons = {
        "v102_vs_v110": compare_models(by_tag["v102"], by_tag["v110"]),
        "v105_vs_v110": compare_models(by_tag["v105"], by_tag["v110"]),
        "v110_vs_v110_flip_tta": compare_models(by_tag["v110"], by_tag["v110_flip_tta"]),
        "v110_repeat1_vs_repeat2": compare_models(by_tag["v110_repeat1"], by_tag["v110_repeat2"]),
    }
    for name, rows in comparisons.items():
        write_rows(os.path.join(args.out_dir, f"{name}.csv"), rows)
    write_summary(os.path.join(args.out_dir, "summary.md"), results, comparisons)
    print("Wrote reliability analysis to", args.out_dir)


if __name__ == "__main__":
    main()
