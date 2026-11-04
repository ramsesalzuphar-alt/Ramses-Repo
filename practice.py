"""Practise a shot unattended: find YouTube references, screen them, run Animate on the usable ones, and rank the results.

Usage: python practice.py "prompt" --subject image.png [--query "search words"] [--runs 4]

This finds better references and settings. It doesn't train the model, which is the same afterwards.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import av
import numpy as np
import torch
from PIL import Image

import generate_video as g
import reference_video

REPORTS = os.path.expanduser("~/ComfyUI/output/practice")
MAX_RUNS = 20
MAX_TEMPERATURE = 85
POSE_STRENGTHS = (1.0, 0.8)
detector = None


def gray_frames(path, limit=200):
    with av.open(path) as container:
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="gray").astype(np.int16))
            if len(frames) >= limit:
                break
    return frames


def frame_changes(frames):
    return np.array([np.abs(frames[i + 1] - frames[i]).mean() for i in range(len(frames) - 1)])


def usable_reference(path):
    """Reject clips with a scene cut, or with too little movement to copy."""
    frames = gray_frames(path)
    if len(frames) < 10:
        return False, "too short"
    changes = frame_changes(frames)
    median = float(np.median(changes))
    if median < 1.0:
        return False, "too little movement"
    if changes.max() > max(median * 5, 25):
        return False, "has a scene cut"
    return True, "ok"


def score_parts(frames, subject_rgb, last_rgb):
    """Scores from 0 to 1. Rough measures: they can't see a plastic-looking face or odd motion."""
    if frames[len(frames) // 2].std() < 5:
        return {"score": 0.0, "steadiness": 0.0, "sharpness": 0.0, "subject_match": 0.0}
    changes = frame_changes(frames)
    steadiness = 1 / max(float(changes.max() / max(np.median(changes), 0.01)), 1)
    middle = frames[len(frames) // 2].astype(np.float32)
    laplacian = 4 * middle[1:-1, 1:-1] - middle[:-2, 1:-1] - middle[2:, 1:-1] - middle[1:-1, :-2] - middle[1:-1, 2:]
    sharpness = min(float(laplacian.var()) / 300, 1.0)
    histograms = [np.histogram(rgb, bins=48, range=(0, 255))[0].astype(np.float64) for rgb in (subject_rgb, last_rgb)]
    subject_match = max(float(np.corrcoef(*histograms)[0, 1]), 0.0)
    score = 0.5 * steadiness + 0.25 * sharpness + 0.25 * subject_match
    return {
        "score": round(score, 3),
        "steadiness": round(steadiness, 3), "sharpness": round(sharpness, 3), "subject_match": round(subject_match, 3),
    }


def person_in_frames(path, samples=6):
    """Share of sampled frames with a confidently detected person, on the CPU so it can't take GPU memory."""
    global detector
    if detector is None:
        from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights, fasterrcnn_mobilenet_v3_large_fpn
        detector = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT).eval()
    with av.open(path) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    picks = [frames[round(i * (len(frames) - 1) / (samples - 1))] for i in range(samples)]
    found = 0
    with torch.no_grad():
        for rgb in picks:
            tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255
            result = detector([tensor])[0]
            # COCO label 1 is a person
            found += bool(((result["labels"] == 1) & (result["scores"] > 0.6)).any())
    return found / samples


def score_clip(path, subject):
    frames = gray_frames(path)
    with av.open(path) as container:
        for frame in container.decode(video=0):
            first = frame.to_ndarray(format="rgb24")
            break
    subject_rgb = np.array(Image.open(subject).convert("RGB"))
    parts = score_parts(frames, subject_rgb, first)
    presence = person_in_frames(path)
    # a clip that loses its subject is a failed shot, however steady it is
    return {**parts, "score": round(parts["score"] * presence**2, 3), "person_in_frame": presence, "subject_left": presence < 1}


def used_references():
    used = set()
    if os.path.isdir(REPORTS):
        for name in os.listdir(REPORTS):
            with open(os.path.join(REPORTS, name), encoding="utf-8") as f:
                used |= {run["reference_id"] for run in json.load(f)["runs"]}
    return used


def wait_until_cool():
    while True:
        result = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        if result.returncode or int(result.stdout.strip() or 0) < MAX_TEMPERATURE:
            return
        print(f"GPU is above {MAX_TEMPERATURE} C, waiting for it to cool", flush=True)
        time.sleep(60)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--query", help="search words, instead of the prompt")
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--session", default="practice")
    parser.add_argument("--full-size", action="store_true", help="remake the top pick at 1280x704 once the small clips are ranked")
    args = parser.parse_args()
    if not 1 <= args.runs <= MAX_RUNS:
        sys.exit(f"--runs must be from 1 to {MAX_RUNS}")
    if not os.path.isfile(args.subject):
        sys.exit(f"Subject image not found: {args.subject}")

    query = args.query or " ".join(args.prompt.split()[:12])
    used = used_references()
    print(f"Searching YouTube: {query}", flush=True)
    candidates = [v for v in reference_video.search(query, 30) if 5 <= (v.get("duration") or 0) <= 600 and v["id"] not in used]
    print(f"{len(candidates)} new candidates, up to {args.runs} runs", flush=True)

    runs = []
    report = os.path.join(REPORTS, time.strftime("%Y%m%d_%H%M%S.json"))
    os.makedirs(REPORTS, exist_ok=True)
    for video in candidates:
        if len(runs) >= args.runs:
            break
        url = f'https://www.youtube.com/watch?v={video["id"]}'
        try:
            # a little way in, since stock clips often open before the action starts
            reference = reference_video.get(url, 1, 3)
        except (reference_video.DownloadError, SystemExit, OSError) as error:
            print(f'Skipped {video["id"]}: {error}', flush=True)
            continue
        ok, reason = usable_reference(reference)
        if not ok:
            print(f'Skipped "{video.get("title")}": {reason}', flush=True)
            continue
        wait_until_cool()
        pose_strength = POSE_STRENGTHS[len(runs) % len(POSE_STRENGTHS)]
        print(f"=== Run {len(runs) + 1} of {args.runs}, motion strength {pose_strength}", flush=True)
        command = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_video.py"), args.prompt, "--session", args.session,
                   "--animate", args.subject, "--video", reference, "--pose-strength", str(pose_strength), "--seconds", "2", "--width", "704", "--height", "480", "--tiled", "--crf", "12"]
        if subprocess.run(command).returncode:
            print("That run failed, moving on", flush=True)
            continue
        clip = g.load_history()[-1]["file"]
        parts = score_clip(os.path.join(g.OUTPUT_VIDEO, clip), args.subject)
        runs.append({"clip": clip, "reference": reference, "reference_id": video["id"], "reference_title": video.get("title"), "reference_url": url, "pose_strength": pose_strength, **parts})
        # after every run, so stopping early keeps what's finished
        with open(report + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"prompt": args.prompt, "subject": args.subject, "runs": runs}, f, indent=1)
        os.replace(report + ".tmp", report)

    runs.sort(key=lambda run: -run["score"])
    print("\nRESULTS, best measured first. These measures are rough, so please watch them:")
    for run in runs:
        flag = "  (subject may leave the frame)" if run.get("subject_left") else ""
        print(f'{run["score"]:.2f}  {run["clip"]}  motion {run["pose_strength"]}  {run["reference_title"]}  {run["reference_url"]}{flag}')
    if not runs:
        print("No clips were made. Try different search words.")
    elif args.full_size:
        best = runs[0]
        wait_until_cool()
        print(f'\n=== Remaking the top pick at full size, which can take about 25 minutes: {best["clip"]}', flush=True)
        command = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_video.py"), args.prompt, "--session", args.session,
                   "--animate", args.subject, "--video", best["reference"], "--pose-strength", str(best["pose_strength"]), "--seconds", "2", "--width", "1280", "--height", "704", "--tiled", "--crf", "12", "--final"]
        sys.exit(subprocess.run(command).returncode)


if __name__ == "__main__":
    still = np.full((48, 64), 120, dtype=np.int16)
    noise = np.random.default_rng(1).integers(0, 255, (48, 64), dtype=np.int16)
    smooth = [noise + i for i in range(12)]
    assert score_parts([still] * 5, np.zeros((4, 4, 3)), np.zeros((4, 4, 3)))["score"] == 0.0
    rgb = np.random.default_rng(2).integers(0, 255, (16, 16, 3))
    jumpy = smooth[:6] + [noise[::-1] for _ in range(6)]
    assert score_parts(smooth, rgb, rgb)["steadiness"] > score_parts(jumpy, rgb, rgb)["steadiness"]
    main()
