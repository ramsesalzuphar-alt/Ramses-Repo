"""One prompt to a finished VACE video, with no other input.

1. Picks the CMU motion capture take whose description best matches the action in the prompt, and downloads it.
2. Renders it in Blender as a depth video (mocap_render.py --depth).
3. Makes a quick Wan 5B clip of the prompt and takes its clearest whole-body frame as the subject's look.
4. Runs VACE 14B (generate_video.py --vace) and reviews the result: smoothness (grade_video.py) and whether the body
   stays whole and looks like the subject image (body_check.py, against Blender's own joint positions). With --tries,
   a clip that loses limbs or drifts in colour is made again with a new seed, and the best one is kept.

Wan 14B makes 16 frames a second, and filling the gaps with interpolation smears fast limbs. So the motion is rendered
at half speed, made at 16 fps, and played back at 32 fps: every frame is a real one. 81 frames is Wan's limit, so
that allows 2.5 seconds.

Usage: auto_vace.py "prompt" [--seconds 2.5] [--full] [--take 05_01] [--subject IMAGE] [--pick-only]

Prompts with no human action in the library (a car, waves) are made with plain Wan 14B instead.
The progress page (/progress) shows each step from ~/ComfyUI/output/plan.json.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

REPO = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
COMFY = os.path.expanduser("~/ComfyUI")
MOCAP = os.path.join(COMFY, "mocap")
CMU = os.path.join(MOCAP, "cmu")
HISTORY = os.path.join(COMFY, "output", "history.jsonl")
PLAN = os.path.join(COMFY, "output", "plan.json")
SLOW = 0.5  # motion speed while generating; the clip is sped back up by 1 / SLOW
MAX_FRAMES = 81
GOOD_BODY = 0.9  # share of frames with every visible limb in place for a clip to pass the body check
BLENDER = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
SOURCE = "https://raw.githubusercontent.com/meshula/cmu-mocap/master/"
# words that say nothing about the movement
STOP = set("a an the and with in on at of to for from her his their its into onto over under by while is are very "
           "realistic cinematic young old man woman person people girl boy guy lady wearing shirt dress grey gray black "
           "white red blue green brown hair ponytail sunrise sunset night day light road street beach park city".split())


def stem(word):
    for end in ("ing", "ed", "es", "s", "e"):
        if word.endswith(end) and len(word) - len(end) >= 3:
            word = word[:-len(end)]
            break
    return word[:-1] if len(word) > 3 and word[-1] == word[-2] else word


def words(text):
    return {stem(w) for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP and len(w) > 2}


def load_index():
    """(take, description, subject title) for every take in the CMU library."""
    path = os.path.join(CMU, "index.txt")
    if not os.path.isfile(path):
        os.makedirs(CMU, exist_ok=True)
        urllib.request.urlretrieve(SOURCE + "cmu-mocap-index-text.txt", path)
    takes, title = [], ""
    for line in open(path, encoding="utf-8", errors="replace"):
        heading = re.match(r"Subject #\d+ \((.*)\)", line)
        if heading:
            title = heading.group(1)
        take = re.match(r"(\d+_\d+)\t(.+)", line)
        if take:
            takes.append((take.group(1), take.group(2).strip(), title))
    return takes


def pick_motion(prompt, takes):
    """Takes ranked by how many of the prompt's action words their description has; a short, focused description and a
    higher subject number (CMU's later, cleaner sessions) break ties."""
    want = words(prompt)
    ranked = []
    for take, description, title in takes:
        action = len(want & words(description))
        if action:  # the take's own description must match: a heading alone (such as "sports") isn't enough
            score = 2 * action + len(want & words(title))
            ranked.append((-score, len(description), -int(take.split("_")[0]), take, description))
    return [(take, description) for *_, take, description in sorted(ranked)]


def fetch(take, seconds):
    """The take's BVH file, or None if it can't be downloaded or is under a second long."""
    path = os.path.join(CMU, take + ".bvh")
    if not os.path.isfile(path):
        url = SOURCE + f"data/{int(take.split('_')[0]):03d}/{take}.bvh"
        try:
            urllib.request.urlretrieve(url, path + ".part")
        except OSError:
            return None
        os.replace(path + ".part", path)
    text = open(path, encoding="utf-8", errors="replace").read()
    found = re.search(r"Frames:\s*(\d+)\s*Frame Time:\s*([\d.]+)", text)
    if "HIERARCHY" not in text or not found or int(found.group(1)) * float(found.group(2)) < 1.0:
        return None
    return path


def set_plan(steps, index=None, status=None):
    if index is not None:
        steps[index]["status"] = status
    with open(PLAN + ".tmp", "w", encoding="utf-8") as f:
        json.dump({"title": "One-prompt VACE video", "steps": steps}, f, indent=1)
    os.replace(PLAN + ".tmp", PLAN)


def generate(arguments):
    """Run generate_video.py and return the clip it logged."""
    before = len(open(HISTORY, encoding="utf-8").read().splitlines()) if os.path.isfile(HISTORY) else 0
    result = subprocess.run([PY, os.path.join(REPO, "generate_video.py"), *arguments], env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    lines = open(HISTORY, encoding="utf-8").read().splitlines() if os.path.isfile(HISTORY) else []
    if result.returncode != 0 or len(lines) <= before:
        return None
    return json.loads(lines[-1])["file"]


def subject_image(prompt, seed_tries=2):
    """A quick 5B clip of the prompt, and its frame with the biggest clear whole person, saved into ComfyUI's input."""
    import av
    import numpy as np
    import motion_check
    for _ in range(seed_tries):
        clip = generate([prompt + ", full body in view, side view", "--fast", "--seconds", "2", "--session", "auto"])
        if not clip:
            continue
        path = os.path.join(COMFY, "output", "video", clip)
        points = motion_check.pose_frames(path)
        sizes = [np.ptp(p[:, 1]) if p is not None else 0 for p in points]
        if max(sizes, default=0) == 0:
            print(f"No clear whole person in {clip}, trying again", flush=True)
            continue
        when = sizes.index(max(sizes)) / motion_check.RATE
        with av.open(path) as container:
            frame = next(f for f in container.decode(video=0) if f.time is not None and f.time + 1e-6 >= when)
        image = os.path.join(COMFY, "input", f"subject_{os.path.splitext(clip)[0]}.png")
        frame.to_image().save(image)
        return image
    return None


def speed_up(source, target, fps):
    """The same frames, played at a new rate: each frame kept once."""
    import av
    with av.open(source) as src, av.open(target, "w") as out:
        stream = out.add_stream("libx264", rate=fps)
        video = src.streams.video[0]
        stream.width, stream.height, stream.pix_fmt, stream.options = video.width, video.height, "yuv420p", {"crf": "12"}
        for frame in src.decode(video=0):
            out.mux(stream.encode(av.VideoFrame.from_ndarray(frame.to_ndarray(format="rgb24"), format="rgb24")))
        out.mux(stream.encode())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--seconds", type=float, default=2.5, help=f"up to {(MAX_FRAMES - 1) * SLOW / 16:g}: longer would pass Wan's {MAX_FRAMES}-frame limit")
    parser.add_argument("--full", action="store_true", help="1280x704 instead of 832x480: sharper, about 4 times slower")
    parser.add_argument("--take", help="use this CMU take, such as 05_01, instead of picking one")
    parser.add_argument("--subject", help="use this image for the subject's look instead of making one")
    parser.add_argument("--tries", type=int, default=1, help="make it again with a new seed, up to this many times, while the body check fails")
    parser.add_argument("--pick-only", action="store_true", help="show the motion it would use, then stop")
    args = parser.parse_args()
    if os.name == "nt":  # keep the PC from sleeping for the hours this runs; Windows drops it when the script exits
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED: Modern Standby sleeps when the screen goes off
    longest =(MAX_FRAMES - 1) * SLOW / 16
    if args.seconds > longest:
        print(f"Shortened to {longest:g} s: at 32 real frames a second, longer would pass Wan's {MAX_FRAMES}-frame limit", flush=True)
        args.seconds = longest
    made_seconds = args.seconds / SLOW
    size = ["--width", "1280", "--height", "704"] if args.full else ["--width", "832", "--height", "480"]
    started = time.time()

    takes = load_index()
    ranked = [(args.take, dict((t, d) for t, d, _ in takes).get(args.take, ""))] if args.take else pick_motion(args.prompt, takes)
    print("Motion matches: " + ("; ".join(f"{t} ({d})" for t, d in ranked[:5]) or "none"), flush=True)
    if args.pick_only:
        return
    bvh, take, description = None, None, None
    for take, description in ranked[:8]:
        bvh = fetch(take, args.seconds)
        if bvh:
            break
    if not bvh:
        print("No motion capture fits this prompt, so making it with plain Wan 14B", flush=True)
        clip = generate([args.prompt, "--model", "14b", "--seconds", str(args.seconds), "--tiled", "--crf", "12", "--session", "auto", *size])
        print(f"Best: {clip}" if clip else "Generation failed")
        sys.exit(0 if clip else 1)
    print(f"Motion: CMU {take}, \"{description}\"", flush=True)

    steps = [{"name": f"Motion: CMU {take} ({description})", "status": "done"},
             {"name": "Blender depth video", "status": "running", "eta": "about 1 min"},
             {"name": "Subject image (quick Wan 5B clip)", "status": "waiting", "eta": "about 3 min"},
             {"name": "VACE 14B video", "status": "waiting", "comfy": True, "eta": ("about 7 hours" if args.full else "about 2 hours") + (f" a try, up to {args.tries}" if args.tries > 1 else "")},
             {"name": "Body and smoothness checks", "status": "waiting", "eta": "about 10 min"}]
    set_plan(steps)
    control, joints = os.path.join(MOCAP, f"{take}_depth.mp4"), os.path.join(MOCAP, f"{take}_joints.json")
    render = subprocess.run([BLENDER, "-b", "-P", os.path.join(REPO, "mocap_render.py"), "--", bvh, control,
                             "--seconds", str(made_seconds + 0.25), "--fps", "16", "--speed", str(SLOW), "--depth", "--joints", joints], capture_output=True, text=True, errors="replace")
    if render.returncode != 0 or not os.path.isfile(control):
        set_plan(steps, 1, "failed")
        sys.exit("Blender render failed:\n" + render.stdout[-1500:] + render.stderr[-1500:])
    set_plan(steps, 1, "done")

    set_plan(steps, 2, "running")
    subject = args.subject or subject_image(args.prompt)
    if not subject:
        set_plan(steps, 2, "failed")
        sys.exit("Couldn't make a subject image with a clear whole person")
    set_plan(steps, 2, "done")
    print(f"Subject: {subject}", flush=True)

    import body_check
    best = None
    for attempt in range(1, args.tries + 1):
        set_plan(steps, 3, "running")
        made = generate([args.prompt, "--vace", control, "--subject", subject, "--seconds", str(made_seconds), "--steps", "20",
                         "--tiled", "--crf", "12", "--no-smooth", "--session", "auto", *size])
        if not made:
            set_plan(steps, 3, "failed")
            sys.exit("VACE generation failed")
        clip = os.path.splitext(made)[0] + "_32fps.mp4"
        speed_up(os.path.join(COMFY, "output", "video", made), os.path.join(COMFY, "output", "video", clip), round(16 / SLOW))
        set_plan(steps, 4, "running")
        body = body_check.check(os.path.join(COMFY, "output", "video", clip), joints, subject)
        print(f"Body (try {attempt}): " + json.dumps(body), flush=True)
        if best is None or body["body_ok"] > best[1]["body_ok"]:
            best = (clip, body)
        drift = body["outfit_drift"] is not None and body["outfit_drift"] > body_check.COLOUR_DRIFT
        if body["body_ok"] >= GOOD_BODY and not drift:
            break
        if attempt < args.tries:
            print(f"Body check failed, making it again with a new seed", flush=True)
    clip, body = best
    set_plan(steps, 3, "done")

    import grade_video
    review = grade_video.review(os.path.join(COMFY, "output", "video", clip))
    print("Review: " + json.dumps(review), flush=True)
    print("Body: " + body["summary"] + f" (whole in {body['body_ok']:.0%} of frames)", flush=True)
    if words(args.prompt) & {"run", "jog", "sprint"}:
        import motion_check  # its real-footage set is runners, so it only means something for a run
        print("Movement: " + json.dumps(motion_check.check(os.path.join(COMFY, "output", "video", clip))), flush=True)
    set_plan(steps, 4, "done")
    print(f"Took {(time.time() - started) / 60:.0f} min")
    print(f"Best: {clip}")


if __name__ == "__main__":
    assert stem("running") == "run" and stem("dancing") == stem("dance") == "danc" and stem("jumps") == "jump"
    assert words("A woman dancing salsa on the beach") == {"danc", "salsa"}
    sample = [("16_35", "run/jog", "run"), ("05_01", "walk, run, jump, dance", "dance"), ("60_02", "salsa dance", "salsa")]
    assert pick_motion("a man dancing salsa", sample)[0][0] == "60_02" and pick_motion("a car driving", sample) == []
    assert pick_motion("a sports car", [("1_1", "frisbee", "sports")]) == []
    main()
