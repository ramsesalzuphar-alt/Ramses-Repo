"""Check whether a person's movement looks natural, by comparing their body pose over time with real clips.

Usage: python motion_check.py clip.mp4 [more.mp4 ...]

Real clips are the saved YouTube references in ~/ComfyUI/reference, so the comparison improves as practice saves more.
It measures the body, not the picture: how far knees bend and feet reach, how arms swing, how fast steps come,
whether limbs keep their length, and how jittery joints are.
"""
import json
import os
import sys

import av
import numpy as np
import torch

REAL = os.path.expanduser("~/ComfyUI/reference")
CACHE = os.path.join(REAL, "motion_tracks.json")  # joint tracks, so changing a measure needs no slow re-tracking
RATE = 16  # every clip is sampled on the same time grid, so generated 48 fps and real 30 fps compare fairly
MAX_SECONDS = 3
MIN_REAL = 4
MIN_STRIDE = 1.0  # feet apart by a torso length or more: below this the person is standing, not moving
# COCO keypoints
SHOULDERS, ELBOWS, WRISTS, HIPS, KNEES, ANKLES = (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)
LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]
WORDS = {
    "knee_bend": ("knees bend too little", "knees bend too much"),
    "stride": ("steps are too short", "steps are too long"),
    "arm_swing": ("arms barely swing", "arms swing too much"),
    "cadence": ("steps come too slowly", "steps come too quickly"),
    "bounce": ("body glides without bouncing", "body bounces too much"),
    "limb_stretch": ("", "limbs stretch and shrink"),
    "jitter": ("", "joints jitter"),
}
model = None


def pose_track(path):
    """Keypoints of one person, followed from frame to frame, on a 16 fps grid, on the CPU so it can't take GPU memory. None if no one is found."""
    global model
    if model is None:
        from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn
        model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT).eval()
    points, next_time = [], 0.0
    with av.open(path) as container:
        for frame in container.decode(video=0):
            if frame.time is None or frame.time + 1e-6 < next_time:
                continue
            if frame.time > MAX_SECONDS:
                break
            next_time += 1 / RATE
            rgb = frame.to_ndarray(format="rgb24", height=480, width=round(480 * frame.width / frame.height))
            with torch.no_grad():
                result = model([torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255])[0]
            seen = result["keypoints_scores"].detach().numpy()
            # the whole body must really be visible: the model guesses hidden joints, so a legs-only shot still gets a "person"
            whole = (seen[:, 0] > 1) & (seen[:, SHOULDERS].max(1) > 1) & (seen[:, ANKLES].max(1) > 1)
            people = result["keypoints"][(result["scores"] > 0.8) & torch.from_numpy(whole)][:, :, :2].numpy()
            if not len(people):
                points.append(None)
                continue
            last = next((p for p in reversed(points) if p is not None), None)
            if last is None:  # start with the biggest person, then stay with whoever is nearest, so a crowd can't swap runners
                pick = max(people, key=lambda p: np.ptp(p[:, 1]))
            else:
                pick = min(people, key=lambda p: np.linalg.norm(p.mean(0) - last.mean(0)))
            points.append(pick)
    found = [p for p in points if p is not None]
    if len(found) < 0.8 * len(points) or len(found) < 12:
        return None
    # fill short gaps by repeating the last pose, so timing stays right
    filled, last = [], found[0]
    for p in points:
        last = p if p is not None else last
        filled.append(last)
    return np.stack(filled)


def angle(a, b, c):
    u, v = a - b, c - b
    return np.degrees(np.arccos(np.clip((u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-6), -1, 1)))


def spread(values, axis=None):
    """Range without the outliers a single bad frame of tracking gives."""
    return np.percentile(values, 90, axis=axis) - np.percentile(values, 10, axis=axis)


def features(track):
    hips = track[:, HIPS].mean(1)
    torso = np.median(np.linalg.norm(track[:, SHOULDERS].mean(1) - hips, axis=-1)) + 1e-6
    body = (track - hips[:, None]) / torso  # size and position no longer matter
    knees = np.stack([angle(body[:, h], body[:, k], body[:, a]) for h, k, a in zip(HIPS, KNEES, ANKLES)])
    feet = body[:, ANKLES[0], 0] - body[:, ANKLES[1], 0]
    wrists = body[:, WRISTS, 0] - body[:, SHOULDERS, 0]
    crossings = np.count_nonzero(np.diff(np.sign(feet - feet.mean())) != 0)
    lengths = np.stack([np.linalg.norm(body[:, i] - body[:, j], axis=-1) for i, j in LIMBS])
    return {
        "knee_bend": float(np.mean(np.percentile(180 - knees, 90, axis=1))),
        "stride": float(spread(feet)),
        "arm_swing": float(np.mean(spread(wrists, axis=0))),
        "cadence": crossings * RATE / len(track),
        "bounce": float(spread(hips[:, 1]) / torso),
        "limb_stretch": float(np.mean(spread(lengths, axis=1) / (np.median(lengths, axis=1) + 1e-6))),
        "jitter": float(np.median(np.linalg.norm(np.diff(body, 2, axis=0), axis=-1))),
    }


def real_features():
    """Features of each saved reference with one clear person; tracks are cached by file name and time."""
    try:
        with open(CACHE, encoding="utf-8") as f:
            cache = json.load(f)
    except (OSError, ValueError):
        cache = {}
    changed = False
    for name in sorted(os.listdir(REAL)) if os.path.isdir(REAL) else []:
        if not name.endswith(".mp4") or "_16fps" in name:
            continue
        key = f"{name}:{int(os.path.getmtime(os.path.join(REAL, name)))}"
        if key not in cache:
            track = pose_track(os.path.join(REAL, name))
            cache[key] = None if track is None else track.round(1).tolist()
            changed = True
    if changed:
        with open(CACHE + ".tmp", "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=1)
        os.replace(CACHE + ".tmp", CACHE)
    found = [features(np.array(track)) for track in cache.values() if track]
    return [f for f in found if legs_move(f)]


def legs_move(found):
    return found["stride"] >= MIN_STRIDE


def compare(found, real):
    """0 to 1, where 1 sits in the middle of the real clips, and what stands out most."""
    notes, distances = [], []
    for name, (too_low, too_high) in WORDS.items():
        values = np.array([r[name] for r in real])
        # median and its typical distance, so one badly tracked real clip can't widen what counts as natural
        middle = np.median(values)
        width = max(1.4826 * np.median(np.abs(values - middle)), 0.15 * abs(middle), 1e-3)
        z = (found[name] - middle) / width
        if (z < 0 and not too_low) or (z > 0 and not too_high):
            z = min(abs(z), 1.0)  # less stretch or jitter than real clips is no fault
        distances.append(abs(z))
        if abs(z) > 1.5:
            notes.append((abs(z), too_low if z < 0 else too_high))
    natural = float(np.exp(-np.mean(np.square(distances)) / 4))
    return round(natural, 3), [text for _, text in sorted(notes, reverse=True)]


def check(path):
    """{"natural": 0-1 or None, "notes": [...]}. None means no clear person, or too few real clips to compare with."""
    real = real_features()
    if len(real) < MIN_REAL:
        return {"natural": None, "notes": [f"only {len(real)} real clips with a clear person to compare with, {MIN_REAL} needed"]}
    track = pose_track(path)
    if track is None:
        return {"natural": None, "notes": ["no clear person in most frames"]}
    natural, notes = compare(features(track), real)
    return {"natural": natural, "notes": notes, "compared_with": len(real)}


if __name__ == "__main__":
    # self-check on a made-up runner: a natural stride sits in the middle, a warped one is flagged
    t = np.arange(32) / RATE
    def runner(stretch=0.0):
        pose = np.zeros((32, 17, 2))
        pose[:, [5, 6], 1], pose[:, [11, 12], 1] = -1, 0
        swing = np.sin(2 * np.pi * 1.5 * t)
        for side, sign in ((0, 1), (1, -1)):
            pose[:, KNEES[side]] = np.stack([0.3 * sign * swing, 0.9 + 0 * t], -1)
            pose[:, ANKLES[side]] = np.stack([0.6 * sign * swing, 1.5 - 0.3 * np.maximum(sign * swing, 0) * (1 + stretch * np.sin(9 * t))], -1)
            pose[:, ELBOWS[side]] = np.stack([-0.2 * sign * swing, -0.5 + 0 * t], -1)
            pose[:, WRISTS[side]] = np.stack([-0.4 * sign * swing, -0.2 + 0 * t], -1)
        return features(pose * 100 + 50)
    real = [runner() for _ in range(5)]
    real = [{k: v * (1 + 0.05 * i) for k, v in r.items()} for i, r in enumerate(real)]
    good, bad = compare(runner(), real), compare(runner(stretch=3), real)
    assert good[0] > 0.8 and not good[1] and bad[0] < 0.5 and bad[1], (good, bad)
    sys.stdout.reconfigure(encoding="utf-8")
    for clip in sys.argv[1:]:
        print(os.path.basename(clip), json.dumps(check(clip)), flush=True)
