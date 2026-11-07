"""Check that the person stays whole and looks the same the whole way through, not only that they move well.

Two references:
- the control joints: where Blender put every joint on screen in each frame of the depth video VACE followed, and
  whether the body hides it (mocap_render.py --joints). Each arm and leg segment Blender shows must be covered by the
  person's outline in the clip, so an arm that fades into the background, melts or shrinks is caught. (A pose model
  can't do this: it guesses joints it can't see.)
- the subject image: how this person looks. Outfit and skin colours that drift away from it are caught.

Usage: python body_check.py clip.mp4 joints.json subject.png [--offset 1]
--offset is how many control frames come before the clip's first frame (VACE drops its first frame, so 1).
"""
import argparse
import json
import sys

import av
import numpy as np
import torch
from PIL import Image

SEGMENTS = {"left upper arm": (5, 7), "left forearm": (7, 9), "right upper arm": (6, 8), "right forearm": (8, 10),
            "left thigh": (11, 13), "left shin": (13, 15), "right thigh": (12, 14), "right shin": (14, 16)}
COVERED = 0.7  # share of a segment's points that must lie on the person
SLACK = 0.12  # the outline is widened by this many torso lengths, as VACE doesn't follow the figure to the pixel
COLOUR_DRIFT = 0.15  # chroma distance from the subject image that counts as a change of clothes or skin (a consistent clip is about 0.07)
model = None


def person_mask(rgb):
    """Pixels that belong to people (Mask R-CNN, on the CPU so it can't take GPU memory), or None if there's nobody."""
    global model
    if model is None:
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
        model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).eval()
    with torch.no_grad():
        result = model([torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255])[0]
    people = (result["labels"] == 1) & (result["scores"] > 0.6)
    if not people.any():
        return None
    return (result["masks"][people, 0] > 0.5).any(0).numpy()


def frames(path):
    with av.open(path) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")


def to_clip(points, control_size, clip_size):
    """Control pixel positions in the clip's pixels: ComfyUI scaled the control to cover the clip size, cropping the middle."""
    (cw, ch), (w, h) = control_size, clip_size
    scale = max(w / cw, h / ch)
    return points * scale - np.array([(cw * scale - w) / 2, (ch * scale - h) / 2])


def widen(mask, pixels):
    pixels = max(1, int(pixels))
    grown = torch.nn.functional.max_pool2d(torch.from_numpy(mask).float()[None, None], 2 * pixels + 1, stride=1, padding=pixels)
    return grown[0, 0].numpy() > 0


def lost_segments(mask, joints):
    """Segments whose both ends Blender shows, but which mostly don't lie on the person in the clip."""
    if mask is None:
        return list(SEGMENTS)
    h, w = mask.shape
    lost = []
    for name, (a, b) in SEGMENTS.items():
        if not (joints[a][2] and joints[b][2]):
            continue  # hidden behind the body in the figure too
        points = np.linspace(joints[a][:2], joints[b][:2], 10).round().astype(int)
        inside = [0 <= x < w and 0 <= y < h and mask[y, x] for x, y in points]
        if np.mean(inside) < COVERED:
            lost.append(name)
    return lost


def colour(rgb, mask):
    """Average chroma (colour without brightness) of the person, ignoring near-black pixels, which have no colour."""
    pixels = rgb[mask].astype(float)
    pixels = pixels[pixels.sum(1) > 40]
    if len(pixels) < 50:
        return None
    mean = pixels.mean(0)
    return mean / mean.sum()


def spans(times, gap=0.1):
    out = []
    for t in times:
        if out and t - out[-1][1] <= gap:
            out[-1][1] = t
        else:
            out.append([t, t])
    return [f"{a:.2f}-{b:.2f} s" if b > a else f"{a:.2f} s" for a, b in out]


def check(clip, joints_path, subject, offset=1):
    with av.open(clip) as container:
        rate = float(container.streams.video[0].average_rate)
    with open(joints_path, encoding="utf-8") as f:
        control = json.load(f)
    track = control["frames"][offset:]
    picture = np.asarray(Image.open(subject).convert("RGB"))
    subject_mask = person_mask(picture)
    reference = colour(picture, subject_mask) if subject_mask is not None else None
    lost, drift, checked = {}, [], 0
    for i, rgb in enumerate(frames(clip)):
        if i >= len(track):
            break
        h, w = rgb.shape[:2]
        where = to_clip(np.array([track[i][str(j)][:2] for j in range(5, 17)]), control["size"], (w, h))
        joints = {j: (*where[j - 5], track[i][str(j)][2]) for j in range(5, 17)}
        torso = np.linalg.norm(where[[0, 1]].mean(0) - where[[6, 7]].mean(0))
        mask = person_mask(rgb)
        checked += 1
        for name in lost_segments(widen(mask, SLACK * torso) if mask is not None else None, joints):
            lost.setdefault(name, []).append(i / rate)
        if reference is not None and mask is not None:
            now = colour(rgb, mask)
            if now is not None:
                drift.append(float(np.abs(now - reference).sum()))
    bad = sorted({t for times in lost.values() for t in times})
    result = {
        "frames_checked": checked,
        "body_ok": round(1 - len(bad) / max(checked, 1), 3),
        "lost_limbs": {name: spans(times) for name, times in sorted(lost.items())},
        "outfit_drift": round(float(np.percentile(drift, 90)), 3) if drift else None,
    }
    notes = [f"{name} missing at {', '.join(s)}" for name, s in result["lost_limbs"].items()]
    if result["outfit_drift"] is not None and result["outfit_drift"] > COLOUR_DRIFT:
        notes.append("colours drift from the subject image")
    result["summary"] = "; ".join(notes) if notes else "body stays whole and matches the subject"
    return result


if __name__ == "__main__":
    # self-check: a covered arm passes, a missing one is caught, a hidden one is skipped, and control pixels map to the clip
    mask = np.zeros((100, 100), bool)
    mask[20:80, 40:60] = True
    joints = {j: (50, 30, True) for j in range(5, 17)}
    joints[9] = (50, 70, True)
    assert lost_segments(mask, joints) == []
    joints[9] = (95, 30, True)
    assert lost_segments(mask, joints) == ["left forearm"]
    joints[9] = (95, 30, False)
    assert lost_segments(mask, joints) == []
    assert np.allclose(to_clip(np.array([[427, 240]]), (854, 480), (832, 480)), [[416, 240]])
    assert np.allclose(to_clip(np.array([[0, 0]]), (854, 480), (1280, 704)), [[0, -(480 * 1280 / 854 - 704) / 2]], atol=0.01)
    assert widen(np.eye(5, dtype=bool), 1)[0, 2] and spans([0.1, 0.15, 0.9]) == ["0.10-0.15 s", "0.90 s"]
    parser = argparse.ArgumentParser()
    parser.add_argument("clip")
    parser.add_argument("joints")
    parser.add_argument("subject")
    parser.add_argument("--offset", type=int, default=1)
    if len(sys.argv) > 1:
        args = parser.parse_args()
        print(json.dumps(check(args.clip, args.joints, args.subject, args.offset), indent=1))
