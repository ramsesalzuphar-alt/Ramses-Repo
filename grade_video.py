"""Grade a generated clip against the real clip it was made from.

Usage: python grade_video.py clip.mp4 [more.mp4 ...] --reference ref.mp4

motion_match, 0 to 1: how closely each part of the picture moves like the reference, from optical flow.
  The reference against itself is 1.0, and an unrelated clip is about 0.
steadiness, 0 to 1: how cleanly each frame follows from the last once motion is allowed for, compared with the reference.
  AI video shimmers and morphs between frames, while real footage changes only where things move. 1 is as steady as the
  reference. It's a rough sign of real footage, not proof: it can't see a wrong face, a plastic look or an extra leg.
Both run on the CPU, so they don't take graphics memory from ComfyUI.
"""
import argparse
import json
import os
import sys

import av
import numpy as np
import torch
import torch.nn.functional as F

RATE = 8  # both clips are compared at the same moments; 8 divides 16, 24 and 48 fps exactly
SECONDS = 2
HEIGHT, WIDTH = 352, 512  # RAFT needs multiples of 8
GRID = 8
EDGE = 8  # warping leaves gaps at the border, so it's left out
raft = None
cache = {}


def nearest(times, seconds, rate):
    """Index of the frame closest to each moment 1/rate apart. Taking the first frame after each moment instead
    gives uneven gaps between clips of different frame rates, which made matching motion look different."""
    targets = [k / rate for k in range(int(seconds * rate + 1e-6) + 1)]
    return [min(range(len(times)), key=lambda i: abs(times[i] - t)) for t in targets if t <= times[-1] + 0.5 / rate]


def sample_frames(path, aspect, seconds=SECONDS, rate=RATE):
    """Frames at the same moments for any frame rate, centre-cropped to the given width/height ratio like the generator does."""
    with av.open(path) as container:
        decoded = []
        for frame in container.decode(video=0):
            if frame.time is None:
                continue
            if frame.time > seconds + 1 / rate:
                break
            decoded.append((frame.time, frame.to_image()))
    if not decoded:
        sys.exit(f"No frames to grade in {path}")
    frames = []
    for index in nearest([t for t, _ in decoded], seconds, rate):
        image = decoded[index][1]
        width, height = image.size
        if width / height > aspect:
            crop = round(height * aspect)
            image = image.crop(((width - crop) // 2, 0, (width - crop) // 2 + crop, height))
        else:
            crop = round(width / aspect)
            image = image.crop((0, (height - crop) // 2, width, (height - crop) // 2 + crop))
        frames.append(torch.from_numpy(np.array(image.resize((WIDTH, HEIGHT)))).permute(2, 0, 1).float() / 255)
    if len(frames) < 3:
        sys.exit(f"Too few frames to grade in {path}")
    return torch.stack(frames)


def measure(path, aspect):
    """Coarse flow fields for motion, and the leftover change after following the flow, for steadiness."""
    global raft
    key = (os.path.abspath(path), round(aspect, 3))
    if key in cache:
        return cache[key]
    if raft is None:
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
        raft = raft_small(weights=Raft_Small_Weights.DEFAULT).eval()
    frames = sample_frames(path, aspect)
    with torch.no_grad():
        flow = raft(frames[:-1] * 2 - 1, frames[1:] * 2 - 1)[-1]
    rows, cols = torch.meshgrid(torch.arange(HEIGHT), torch.arange(WIDTH), indexing="ij")
    grid = torch.stack([(cols + flow[:, 0]) / (WIDTH - 1) * 2 - 1, (rows + flow[:, 1]) / (HEIGHT - 1) * 2 - 1], -1)
    warped = F.grid_sample(frames[1:], grid, align_corners=True)
    leftover = float((warped - frames[:-1]).abs()[:, :, EDGE:-EDGE, EDGE:-EDGE].mean()) * 100
    cache[key] = (F.adaptive_avg_pool2d(flow, GRID).numpy(), leftover)
    return cache[key]


def match_fields(a, b):
    """Average similarity of two runs of flow fields: 1 when every region moves the same way by the same amount."""
    count = min(len(a), len(b))
    scores = []
    for x, y in zip(a[:count], b[:count]):
        size_x, size_y = np.linalg.norm(x), np.linalg.norm(y)
        if max(size_x, size_y) < 0.5 * GRID:
            scores.append(1.0)  # both nearly still
            continue
        direction = float((x * y).sum() / (size_x * size_y + 1e-6))
        amount = min(size_x, size_y) / max(size_x, size_y)
        scores.append(max(direction, 0.0) * amount)
    return round(float(np.mean(scores)), 3) if scores else 0.0


def steadiness(leftover, reference_leftover):
    return round(min(reference_leftover / max(leftover, 1e-6), 1.0), 3)


def grade(clip, reference):
    with av.open(clip) as container:
        stream = container.streams.video[0]
        aspect = stream.width / stream.height
    clip_flow, clip_leftover = measure(clip, aspect)
    reference_flow, reference_leftover = measure(reference, aspect)
    return {
        "motion_match": match_fields(reference_flow, clip_flow),
        "steadiness": steadiness(clip_leftover, reference_leftover),
    }


if __name__ == "__main__":
    still = np.zeros((3, 2, GRID, GRID))
    right = np.zeros((3, 2, GRID, GRID))
    right[:, 0] = 3.0
    assert match_fields(right, right) == 1.0 and match_fields(right, -right) == 0.0 and match_fields(still, still) == 1.0
    assert match_fields(right, right * 0.5) == 0.5
    assert steadiness(2.0, 1.0) == 0.5 and steadiness(0.5, 1.0) == 1.0
    thirty, twenty_four = [i / 30 for i in range(90)], [i / 24 for i in range(72)]
    assert [round(thirty[i], 2) for i in nearest(thirty, 1, 8)] == [0.0, 0.13, 0.23, 0.37, 0.5, 0.63, 0.73, 0.87, 1.0]
    assert [twenty_four[i] for i in nearest(twenty_four, 1, 8)] == [k / 8 for k in range(9)]
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("clips", nargs="+")
    parser.add_argument("--reference", required=True, help="the real clip it was made from")
    args = parser.parse_args()
    for clip in args.clips:
        print(os.path.basename(clip), json.dumps(grade(clip, args.reference)), flush=True)
