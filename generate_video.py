"""Queue a Wan 2.2 text-to-video job on the local ComfyUI server and wait for the file.

Usage: python generate_video.py "prompt" [--seconds 2] [--width 704] [--height 480] [--steps 20]
                                  [--video input.mp4] [--strength 0.6]

--video reworks an existing clip to match the prompt, keeping its motion and framing.
--strength is how much may change: low stays close to the clip, 1.0 ignores it.

Every finished clip is logged. --history lists them, --rate wan_00006_.mp4 5 scores one,
and --best reuses the settings of the top-rated clip with smoothing on.
"""
import argparse
import ctypes
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

import av
import imageio_ffmpeg

SERVER = "http://127.0.0.1:8188"
COMFY_INPUT = os.path.expanduser("~/ComfyUI/input")
HISTORY = os.path.expanduser("~/ComfyUI/output/history.jsonl")
OUTPUT_VIDEO = os.path.expanduser("~/ComfyUI/output/video")
SESSIONS = os.path.expanduser("~/ComfyUI/output/sessions")
SETTINGS = ("model", "width", "height", "steps", "crf", "tiled", "upscale", "smooth", "fps", "fast")
# step-distilled add-on for the 5B model, from hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF on Hugging Face
TURBO_LORA = "Wan22_TI2V_5B_Turbo_lora_rank_64_fp16.safetensors"
FAST_STEPS = 4
# a reference graded against itself is 1.0 and an unrelated clip about 0. On a frisbee test, 20 steps at strength 0.6
# gave 0.86 and kept the prompt's dog; 0.3 gave 0.91 but kept the reference's dog; --fast at 0.6 gave only 0.51
MOTION_TARGET = 0.85
MOTION_STEP = 0.15
MOTION_ATTEMPTS = 3
MIN_STRENGTH = 0.25
PRESETS = {
    "draft": {"model": "5b", "width": 704, "height": 480, "steps": 20, "tiled": True},
    "balanced": {"model": "14b", "width": 704, "height": 480, "steps": 20, "tiled": True, "crf": 12.0, "fps": 48},
    "max": {"model": "14b", "width": 1280, "height": 704, "steps": 20, "tiled": True, "crf": 12.0, "fps": 48},
}
CAMERA = {
    "static": "static locked-off camera",
    "tracking": "smooth tracking shot moving alongside the subject",
    "dolly-in": "slow dolly in towards the subject",
    "dolly-out": "slow dolly out away from the subject",
    "pan": "smooth panning shot",
    "orbit": "camera slowly orbiting around the subject",
    "handheld": "handheld camera with subtle natural shake",
    "drone": "aerial drone shot",
}
ANGLE = {
    "eye-level": "eye-level angle",
    "low": "low angle looking up",
    "high": "high angle looking down",
    "overhead": "overhead top-down angle",
    "side": "side-on profile view",
    "front": "front view",
    "back": "view from behind",
}
SHOT = {
    "wide": "wide shot",
    "full-body": "full body shot, entire subject from head to feet in frame",
    "medium": "medium shot from the waist up",
    "close-up": "close-up shot",
}
BEST_PRESET = {"model": "5b", "width": 1280, "height": 704, "steps": 30, "crf": 12.0, "tiled": True, "upscale": False, "smooth": True}
FPS = 24
# 14B is made for 16 fps; playing it at 24 makes everything move 1.5x too fast
FPS_14B = 16
# the distilled Animate model is built for this many steps
ANIMATE_STEPS = 10
NEGATIVE = (
    "blurry, low quality, distorted, deformed, extra limbs, extra fingers, bad anatomy, "
    "static, still frame, watermark, text, subtitles"
)


def valid_frames(frames):
    # Wan needs 4n+1 frames
    return frames - (frames - 1) % 4


def frame_count(seconds, fps=FPS):
    return valid_frames(round(seconds * fps) + 1)


def build_workflow(prompt, width, height, seconds, steps, seed):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_ti2v_5B_fp16.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": 8.0}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": NEGATIVE}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": "wan2.2_vae.safetensors"}},
        "7": {"class_type": "Wan22ImageToVideoLatent", "inputs": {"vae": ["6", 0], "width": width, "height": height, "length": frame_count(seconds), "batch_size": 1}},
        "8": {"class_type": "KSampler", "inputs": {"model": ["2", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["7", 0], "seed": seed, "steps": steps, "cfg": 5.0, "sampler_name": "uni_pc", "scheduler": "simple", "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["6", 0]}},
        "10": {"class_type": "CreateVideo", "inputs": {"images": ["9", 0], "fps": float(FPS)}},
        "11": {"class_type": "SaveVideo", "inputs": {"video": ["10", 0], "filename_prefix": "video/wan", "format": "auto", "format.codec": "auto"}},
    }


def use_turbo(workflow):
    """Load the Turbo add-on into the 5B model. It's trained to need no guidance, so cfg is 1."""
    workflow["19"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": TURBO_LORA, "strength_model": 1.0}}
    workflow["2"]["inputs"]["model"] = ["19", 0]
    workflow["8"]["inputs"]["cfg"] = 1.0


def use_14b_model(workflow, width, height, seconds, steps, seed):
    """Swap the 5B model for Wan's two-stage 14B: high noise first, then low noise."""
    switch = steps // 2
    workflow["1"]["inputs"]["unet_name"] = "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
    workflow["6"]["inputs"]["vae_name"] = "wan_2.1_vae.safetensors"
    workflow["7"] = {"class_type": "EmptyHunyuanLatentVideo", "inputs": {"width": width, "height": height, "length": frame_count(seconds, FPS_14B), "batch_size": 1}}
    workflow["10"]["inputs"]["fps"] = float(FPS_14B)
    workflow["16"] = {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}}
    workflow["17"] = {"class_type": "ModelSamplingSD3", "inputs": {"model": ["16", 0], "shift": 8.0}}
    shared = {"positive": ["4", 0], "negative": ["5", 0], "steps": steps, "cfg": 3.5, "sampler_name": "euler", "scheduler": "simple", "noise_seed": seed}
    workflow["8"] = {"class_type": "KSamplerAdvanced", "inputs": {**shared, "model": ["2", 0], "latent_image": ["7", 0], "add_noise": "enable", "start_at_step": 0, "end_at_step": switch, "return_with_leftover_noise": "enable"}}
    workflow["18"] = {"class_type": "KSamplerAdvanced", "inputs": {**shared, "model": ["17", 0], "latent_image": ["8", 0], "add_noise": "disable", "start_at_step": switch, "end_at_step": steps, "return_with_leftover_noise": "disable"}}
    workflow["9"]["inputs"]["samples"] = ["18", 0]


def prepare_video(path, seconds, max_fps):
    """Put the clip where ComfyUI can load it; return its name, usable frame count and fps."""
    if not os.path.isfile(path):
        sys.exit(f"Video not found: {path}")
    try:
        with av.open(path) as container:
            source_fps = float(container.streams.video[0].average_rate)
    except (av.FFmpegError, IndexError) as error:
        sys.exit(f"Could not read {path} as a video: {error}")
    # only ever lower the rate: raising it would just repeat frames and look choppy
    fps = min(source_fps, max_fps)
    stem = os.path.splitext(os.path.basename(path))[0]
    filename = f"{stem}_{fps:g}fps.mp4" if source_fps > max_fps else os.path.basename(path)
    target = os.path.join(COMFY_INPUT, filename)
    if source_fps > max_fps:
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", path, "-t", str(seconds + 1), "-r", f"{fps:g}", "-an", "-c:v", "libx264", "-crf", "12", target]
        if subprocess.run(command).returncode != 0:
            sys.exit(f"Could not convert {path} to {fps:g} fps")
    elif os.path.abspath(path) != os.path.abspath(target):
        shutil.copyfile(path, target)
    wanted = round(seconds * fps) + 1
    frames = 0
    with av.open(target) as container:
        for _ in container.decode(video=0):
            frames += 1
            if frames >= wanted:
                break
    if frames < 5:
        sys.exit(f"Video is too short: {frames} frames, need at least 5")
    return filename, valid_frames(frames), fps


def apply_strength_14b(workflow, strength):
    """Start partway through the steps: a later start keeps more of the input clip."""
    steps = workflow["18"]["inputs"]["steps"]
    switch = workflow["18"]["inputs"]["start_at_step"]
    start = steps - max(1, round(steps * strength))
    if start < switch:
        workflow["8"]["inputs"]["start_at_step"] = start
    else:
        # the high noise stage only sets broad motion, which should come from the clip here
        del workflow["8"]
        workflow["18"]["inputs"].update({"latent_image": ["7", 0], "add_noise": "enable", "start_at_step": start})


def use_video_input(workflow, filename, frames, fps, width, height, strength):
    workflow["12"] = {"class_type": "LoadVideo", "inputs": {"file": filename}}
    workflow["13"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["12", 0]}}
    workflow["14"] = {"class_type": "ImageFromBatch", "inputs": {"image": ["13", 0], "batch_index": 0, "length": frames}}
    workflow["15"] = {"class_type": "ImageScale", "inputs": {"image": ["14", 0], "upscale_method": "lanczos", "width": width, "height": height, "crop": "center"}}
    workflow["7"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["15", 0], "vae": ["6", 0]}}
    if "18" in workflow:
        apply_strength_14b(workflow, strength)
    else:
        workflow["8"]["inputs"]["denoise"] = strength
    workflow["10"]["inputs"]["fps"] = fps


def build_animate_workflow(prompt, image, video, frames, fps, width, height, seed, pose_strength=1.0):
    """Wan Animate 2 distilled, following ComfyUI's template, with the cache in RAM for small GPUs."""
    scale = lambda source: {"class_type": "ImageScale", "inputs": {"image": source, "upscale_method": "area", "width": width, "height": height, "crop": "center"}}
    encode = lambda text: {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": text}}
    vision = lambda source: {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["6", 0], "image": source, "crop": "none"}}
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan_animate_2_distill_int8_convrot.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "WanAnimate2Cache", "inputs": {"model": ["1", 0], "device": "cpu", "dtype": "int8"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan"}},
        "4": encode(f"Character appearance description: {prompt}"),
        "5": encode(NEGATIVE),
        "6": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "clip_vision_h.safetensors"}},
        "7": {"class_type": "VAELoader", "inputs": {"vae_name": "Wan2_1_VAE_bf16.safetensors"}},
        "8": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["2", 0], "shift": 5.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["24", 0], "vae": ["7", 0]}},
        "10": {"class_type": "CreateVideo", "inputs": {"images": ["9", 0], "fps": fps}},
        "11": {"class_type": "SaveVideo", "inputs": {"video": ["10", 0], "filename_prefix": "video/wan", "format": "auto", "format.codec": "auto"}},
        "12": {"class_type": "LoadImage", "inputs": {"image": image}},
        "13": scale(["12", 0]),
        "14": {"class_type": "LoadVideo", "inputs": {"file": video}},
        "15": {"class_type": "GetVideoComponents", "inputs": {"video": ["14", 0]}},
        "16": {"class_type": "ImageFromBatch", "inputs": {"image": ["15", 0], "batch_index": 0, "length": frames}},
        "17": scale(["16", 0]),
        "18": vision(["13", 0]),
        "19": {"class_type": "ImageFromBatch", "inputs": {"image": ["17", 0], "batch_index": 0, "length": 1}},
        "20": vision(["19", 0]),
        "21": encode(prompt),
        "22": {"class_type": "WanAnimate2ToVideo", "inputs": {
            "positive": ["4", 0], "negative": ["5", 0], "vae": ["7", 0], "width": width, "height": height, "length": frames, "batch_size": 1,
            "video_frame_offset": 0, "pose_strength": pose_strength, "pose_start_percent": 0.0, "pose_end_percent": 1.0, "reference_image_strength": 1.0,
            "reference_image": ["13", 0], "pose_video": ["17", 0], "clip_vision_output": ["18", 0], "positive_pose": ["21", 0], "clip_vision_output_pose": ["20", 0]}},
        "23": {"class_type": "SamplerCustom", "inputs": {
            "model": ["8", 0], "add_noise": True, "noise_seed": seed, "cfg": 1.0, "positive": ["22", 0], "negative": ["22", 1],
            "sampler": ["25", 0], "sigmas": ["26", 0], "latent_image": ["22", 2]}},
        "24": {"class_type": "TrimVideoLatent", "inputs": {"samples": ["23", 0], "trim_amount": ["22", 3]}},
        "25": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "lcm"}},
        "26": {"class_type": "BasicScheduler", "inputs": {"model": ["2", 0], "scheduler": "simple", "steps": ANIMATE_STEPS, "denoise": 1.0}},
    }


def cinematography(args):
    parts = [CAMERA.get(args.camera), ANGLE.get(args.angle), SHOT.get(args.shot)]
    parts.append(f"shot on a {args.lens:g}mm lens" if args.lens else None)
    parts += [args.lighting, args.style]
    return ", ".join(part for part in parts if part)


def work_units(width, height, seconds, steps, model):
    return width * height * frame_count(seconds, FPS_14B if model == "14b" else FPS) * steps


def estimate_minutes(entries, args):
    """Scale the nearest logged text-only clip of the same model; returns None with no data."""
    work = work_units(args.width, args.height, args.seconds, args.steps, args.model)
    past = [e for e in entries if e["settings"]["model"] == args.model and not e.get("video")]
    if not past:
        return None
    units = lambda e: work_units(e["settings"]["width"], e["settings"]["height"], e["seconds"], e["settings"]["steps"], args.model)
    nearest = min(past, key=lambda e: abs(units(e) - work))
    return nearest["minutes"] * work / units(nearest)


def add_finishing(workflow, width, height, tiled, upscale, smooth, target_fps=None):
    if target_fps:
        smooth = target_fps > workflow["10"]["inputs"]["fps"]
    images = ["9", 0]
    if tiled:
        decode = workflow["9"]["inputs"]
        workflow["9"] = {"class_type": "VAEDecodeTiled", "inputs": {**decode, "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8}}
    if upscale:
        # RealESRGAN always gives 4x, which is too heavy to keep, so scale back to 2x
        workflow["30"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "RealESRGAN_x4plus.safetensors"}}
        workflow["31"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["30", 0], "image": images}}
        workflow["32"] = {"class_type": "ImageScale", "inputs": {"image": ["31", 0], "upscale_method": "lanczos", "width": width * 2, "height": height * 2, "crop": "disabled"}}
        images = ["32", 0]
    if smooth:
        workflow["33"] = {"class_type": "FrameInterpolationModelLoader", "inputs": {"model_name": "rife_v4.26.safetensors"}}
        # aim for about 48 fps: 16 fps needs tripling, 24 fps doubling
        multiplier = max(2, round((target_fps or 48) / workflow["10"]["inputs"]["fps"]))
        workflow["34"] = {"class_type": "FrameInterpolate", "inputs": {"interp_model": ["33", 0], "images": images, "multiplier": multiplier}}
        images = ["34", 0]
        workflow["10"]["inputs"]["fps"] *= multiplier
    workflow["10"]["inputs"]["images"] = images


def finish_clip(name):
    """Save a graded copy with light sharpening and film grain, which hide some of the raw AI look."""
    source = os.path.join(OUTPUT_VIDEO, name)
    final = os.path.splitext(name)[0] + "_final.mp4"
    grade = "eq=contrast=1.06:saturation=1.08,unsharp=5:5:0.6,noise=alls=7:allf=t"
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", source, "-vf", grade, "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", "-c:a", "copy", os.path.join(OUTPUT_VIDEO, final)]
    if subprocess.run(command).returncode != 0:
        sys.exit(f"Could not finish {name}")
    return final


def join_clips(names, music):
    """Cut clips into one film, matching the first clip's size and frame rate, with optional music."""
    paths = [os.path.join(OUTPUT_VIDEO, os.path.basename(n)) for n in names]
    missing = [p for p in paths if not os.path.isfile(p)]
    if len(paths) < 2 or missing:
        sys.exit(f"--join needs at least two existing clips. Missing: {missing}")
    if music and not os.path.isfile(music):
        sys.exit(f"Music not found: {music}")
    with av.open(paths[0]) as container:
        stream = container.streams.video[0]
        width, height, fps = stream.width, stream.height, float(stream.average_rate)
    fit = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={fps:g},setsar=1"
    parts = ";".join(f"[{i}:v]{fit}[v{i}]" for i in range(len(paths)))
    concat = "".join(f"[v{i}]" for i in range(len(paths))) + f"concat=n={len(paths)}:v=1:a=0[out]"
    name = time.strftime("film_%Y%m%d_%H%M%S.mp4")
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error"]
    for path in paths:
        command += ["-i", path]
    if music:
        command += ["-i", music]
    command += ["-filter_complex", f"{parts};{concat}", "-map", "[out]"]
    if music:
        command += ["-map", f"{len(paths)}:a:0", "-c:a", "aac", "-shortest"]
    command += ["-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", os.path.join(OUTPUT_VIDEO, name)]
    if subprocess.run(command).returncode != 0:
        sys.exit("Could not join the clips")
    return name, width, height, fps


def session_path(name):
    if not re.fullmatch(r"[\w -]+", name):
        sys.exit("Session names can only use letters, numbers, spaces, - and _")
    return os.path.join(SESSIONS, name + ".json")


def load_session(name):
    path = session_path(name)
    if not os.path.exists(path):
        return {"memory": {}, "clips": []}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_session(name, session):
    os.makedirs(SESSIONS, exist_ok=True)
    path = session_path(name)
    # write then swap, so a crash can't lose the session
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2)
    os.replace(path + ".tmp", path)


def load_history():
    if not os.path.exists(HISTORY):
        return []
    with open(HISTORY, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def log_run(entry):
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def best_settings(entries):
    # only clips rated 4 or 5 count, and the newest wins a tie
    # Animate clips are skipped because those settings need a subject image and clip
    rated = [e for e in reversed(entries) if e.get("rating", 0) >= 4 and e["settings"].get("model") not in ("animate", "film")]
    settings = max(rated, key=lambda e: e["rating"])["settings"] if rated else BEST_PRESET
    return {**settings, "smooth": True}


def update_entry(filename, fields):
    entries = load_history()
    matches = [e for e in entries if e["file"] == filename]
    if not matches:
        sys.exit(f"{filename} isn't in the history")
    matches[-1].update(fields)
    # write then swap, so a crash can't lose the log
    with open(HISTORY + ".tmp", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(e) + "\n" for e in entries)
    os.replace(HISTORY + ".tmp", HISTORY)


def rate_clip(filename, score):
    if score not in ("1", "2", "3", "4", "5"):
        sys.exit("Rating must be a whole number from 1 to 5")
    update_entry(filename, {"rating": int(score)})
    print(f"Rated {filename} {score}/5")


def drop_options(argv, flags, valued):
    """argv without the given on/off flags and the given options with their values, in either --x v or --x=v form."""
    kept, skip = [], False
    for item in argv:
        if skip:
            skip = False
        elif item in flags or item.split("=")[0] in valued and "=" in item:
            continue
        elif item in valued:
            skip = True
        else:
            kept.append(item)
    return kept


def match_motion(args):
    """Remake the clip from args.video at lower strengths until its motion follows the reference; grade every attempt."""
    import grade_video
    child = drop_options(sys.argv[1:], {"--match-motion", "--auto-reference"}, {"--reference-query", "--strength", "--video"})
    strengths = [round(args.strength - MOTION_STEP * n, 2) for n in range(MOTION_ATTEMPTS)]
    # grade against the copy the generator actually uses: its frame rate matches the result's, while the original's
    # differs, and on shaky footage that alone moved the score from 1.0 to about 0.65
    graded_against = os.path.join(COMFY_INPUT, prepare_video(args.video, args.seconds, FPS_14B if args.model == "14b" else FPS)[0])
    attempts = []
    for number, strength in enumerate(s for s in strengths if s >= MIN_STRENGTH):
        print(f"=== Attempt {number + 1}, strength {strength}", flush=True)
        if subprocess.run([sys.executable, os.path.abspath(__file__), *child, "--video", args.video, "--strength", str(strength)]).returncode:
            sys.exit("Generation failed, so stopping")
        clip = load_history()[-1]["file"]
        grades = grade_video.grade(os.path.join(OUTPUT_VIDEO, clip), graded_against)
        update_entry(clip, grades)
        attempts.append({"file": clip, "strength": strength, **grades})
        print(f'Motion match {grades["motion_match"]:.2f}, steadiness {grades["steadiness"]:.2f}', flush=True)
        if grades["motion_match"] >= MOTION_TARGET:
            break
    print("\nGRADES, motion match first. Steadiness compares frame-to-frame consistency with the real reference, 1 being as steady:")
    for a in attempts:
        print(f'{a["file"]}  strength {a["strength"]}  motion match {a["motion_match"]:.2f}  steadiness {a["steadiness"]:.2f}')
    best = max(attempts, key=lambda a: (a["motion_match"] >= MOTION_TARGET, a["motion_match"] + a["steadiness"]))
    note = "" if best["motion_match"] >= MOTION_TARGET else f", though no attempt reached a motion match of {MOTION_TARGET}"
    print(f'Best: {best["file"]}{note}')


def show_history():
    for e in load_history():
        s = e["settings"]
        extras = [name for name in ("tiled", "upscale", "smooth", "fast") if s.get(name)]
        print(f'{e["file"]}  {e.get("rating", "-")}/5  {e["minutes"]} min  {s["model"]} {s["width"]}x{s["height"]} {s["steps"]} steps {" ".join(extras)}  {e["prompt"][:50]}')


def request(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(SERVER + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?")
    parser.add_argument("--seconds", type=float, default=2)
    parser.add_argument("--width", type=int, default=704)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--best", action="store_true", help="use the top-rated settings, with smoothing; other options still override")
    parser.add_argument("--rate", nargs=2, metavar=("FILE", "SCORE"), help="score a logged clip from 1 to 5")
    parser.add_argument("--history", action="store_true", help="list logged clips")
    parser.add_argument("--model", choices=["5b", "14b"], default="5b", help="14b is better but much slower")
    parser.add_argument("--seed", type=int, help="repeat a clip's randomness; random by default")
    parser.add_argument("--revise", metavar="FILE", help="develop a logged clip, or 'latest' in a session: the prompt is your change, and its settings and seed are kept")
    parser.add_argument("--auto-reference", action="store_true", help="find a YouTube clip matching the prompt and use it as --video")
    parser.add_argument("--reference-query", help="search words for --auto-reference, instead of ones picked from the prompt")
    parser.add_argument("--fast", action="store_true", help=f"5b only: the Turbo add-on makes a clip in {FAST_STEPS} steps, about 3x faster")
    parser.add_argument("--match-motion", action="store_true", help=f"remake at lower strengths until the motion matches the reference, up to {MOTION_ATTEMPTS} tries, grading each")
    parser.add_argument("--tiled", action="store_true", help="decode in pieces: less memory, small risk of seams")
    parser.add_argument("--upscale", action="store_true", help="double the resolution with RealESRGAN")
    parser.add_argument("--smooth", action=argparse.BooleanOptionalAction, help="raise the frame rate to about 48 fps with RIFE; on by default for 14b")
    parser.add_argument("--fps", type=int, help="final frame rate: 14b gives 16, 32 or 48, and 5b gives 24 or 48")
    parser.add_argument("--preset", choices=PRESETS, help="draft is fastest, max is slowest; other options still override")
    parser.add_argument("--estimate", action="store_true", help="show the settings and rough time, then stop")
    parser.add_argument("--camera", choices=CAMERA, help="camera movement")
    parser.add_argument("--angle", choices=ANGLE, help="camera angle")
    parser.add_argument("--shot", choices=SHOT, help="how much of the subject is in frame")
    parser.add_argument("--lens", type=float, help="lens length in mm, such as 35 or 85")
    parser.add_argument("--lighting", help="lighting, such as 'golden hour backlight'")
    parser.add_argument("--style", help="look, such as 'documentary film' or 'anime'")
    parser.add_argument("--crf", type=float, help="H.264 quality: lower is better and larger, 18 is very good")
    parser.add_argument("--video", help="existing clip to rework")
    parser.add_argument("--strength", type=float, default=0.6)
    parser.add_argument("--tries", type=int, default=1, help="make this many versions with different seeds, 1 to 10")
    parser.add_argument("--final", action="store_true", help="also save a graded copy with light sharpening and film grain")
    parser.add_argument("--join", nargs="+", metavar="CLIP", help="cut these logged clips into one film")
    parser.add_argument("--music", help="audio file to play over --join")
    parser.add_argument("--session", help="project name: remembers your instructions and clips between generations")
    parser.add_argument("--remember", action="append", default=[], metavar="KEY=VALUE", help="save or replace an instruction, such as outfit=red t-shirt")
    parser.add_argument("--forget", action="append", default=[], metavar="KEY", help="remove a saved instruction")
    parser.add_argument("--memory", action="store_true", help="show what the session remembers, then stop")
    parser.add_argument("--pose-strength", type=float, default=1.0, help="how closely Animate copies the motion: 1.0 is normal")
    parser.add_argument("--animate", metavar="IMAGE", help="animate the subject in this image with the motion from --video, using Wan Animate")
    if "--best" in sys.argv:
        # as defaults, so any option typed explicitly still wins
        parser.set_defaults(**best_settings(load_history()))
    args = parser.parse_args()
    if args.preset:
        # parse again so the preset applies, but any option typed explicitly still wins
        parser.set_defaults(**PRESETS[args.preset])
        args = parser.parse_args()
    if args.smooth is None:
        # 14B and Animate are made at 16 fps, which looks choppy unless smoothed
        args.smooth = args.model == "14b" or bool(args.animate)
    if args.rate:
        return rate_clip(*args.rate)
    if args.history:
        return show_history()
    if args.join:
        name, width, height, fps = join_clips(args.join, args.music)
        final = finish_clip(name) if args.final else None
        log_run({"file": name, "final": final, "time": time.strftime("%Y-%m-%d %H:%M"), "minutes": 0, "prompt": "Film: " + ", ".join(args.join), "seed": None, "seconds": None, "video": None, "strength": None, "session": args.session, "animate": None, "revised_from": None, "change": None, "settings": {"model": "film", "width": width, "height": height, "steps": 0, "crf": 14, "tiled": False, "upscale": False, "smooth": False, "fps": fps}})
        print(f"Finished in 0.0 min\nSaved: {final or name}")
        return
    if not 1 <= args.tries <= 10:
        sys.exit("--tries must be from 1 to 10")
    if args.tries > 1:
        one = sys.argv[1:]
        at = one.index("--tries")
        del one[at:at + 2]
        for number in range(args.tries):
            print(f"=== Try {number + 1} of {args.tries}", flush=True)
            # a random seed each time, because --revise would otherwise repeat the same one
            code = subprocess.run([sys.executable, os.path.abspath(__file__), *one, "--seed", str(random.randrange(2**32))]).returncode
            if code:
                sys.exit(code)
        return
    session = load_session(args.session) if args.session else None
    if session is None and (args.remember or args.forget or args.memory):
        sys.exit("--remember, --forget and --memory need --session")
    if session is not None:
        for key in args.forget:
            session["memory"].pop(key, None)
        for item in args.remember:
            key, separator, value = item.partition("=")
            if not (separator and key.strip() and value.strip()):
                sys.exit(f"--remember needs KEY=VALUE, not: {item}")
            session["memory"][key.strip()] = value.strip()
        if args.memory:
            save_session(args.session, session)
            print(json.dumps(session, indent=2))
            return
    revise_file = args.revise
    if revise_file == "latest":
        if not (session and session["clips"]):
            sys.exit("--revise latest needs a --session that already has a clip")
        revise_file = session["clips"][-1]
    if not args.prompt and not revise_file and not (session and session["memory"]):
        parser.error("a prompt is required")
    change = None
    if revise_file:
        matches = [e for e in load_history() if e["file"] == revise_file]
        if not matches:
            sys.exit(f"{revise_file} isn't in the history")
        original = matches[-1]
        # parse again so the clip's settings apply, but any option typed explicitly still wins
        parser.set_defaults(**original["settings"], seconds=original["seconds"], seed=original["seed"], strength=0.5, animate=original.get("animate"))
        args = parser.parse_args()
        args.revise = revise_file
        change = args.prompt
        if session is None:
            if not change:
                parser.error("a prompt is required")
            args.prompt = f'{original["prompt"]} {change}'
        elif not session["memory"]:
            session["memory"]["scene"] = original["prompt"]
        args.video = os.path.join(OUTPUT_VIDEO, revise_file)
    if session is not None:
        if change:
            session["memory"][f'edit {len(session["memory"])}'] = change
        elif args.prompt and not revise_file:
            session["memory"]["scene"] = args.prompt
        memory = session["memory"]
        # saved like any other instruction, so later edits keep the same camera and look
        if cinematography(args):
            memory["camera and look"] = cinematography(args)
        # the scene reads best first, whatever order things were remembered in
        args.prompt = ", ".join(value for _, value in sorted(memory.items(), key=lambda item: item[0] != "scene"))
    if args.auto_reference:
        if args.video:
            sys.exit("--auto-reference can't be combined with --video or --revise")
        import reference_video
        args.video = reference_video.find_reference(args.reference_query or reference_video.search_words(args.prompt), args.seconds + 1)
    if not 0 < args.strength <= 1:
        sys.exit("--strength must be above 0 and at most 1")
    if args.fast:
        if args.model != "5b" or args.animate:
            sys.exit("--fast only works with the 5b model, and Animate is already fast")
        if not os.path.isfile(os.path.expanduser(f"~/ComfyUI/models/loras/{TURBO_LORA}")):
            sys.exit(f"--fast needs {TURBO_LORA} in ~/ComfyUI/models/loras, from hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF on Hugging Face")
        args.steps = FAST_STEPS
    if args.match_motion:
        if not args.video or args.animate or revise_file or args.session or args.tries > 1:
            sys.exit("--match-motion needs --video or --auto-reference, and can't be used with --animate, --revise, --session or --tries")
        return match_motion(args)

    # Block idle sleep while this process runs; ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

    extras = cinematography(args)
    if extras and session is None:
        args.prompt = f"{args.prompt}, {extras}"
    seed = args.seed if args.seed is not None else random.randrange(2**32)
    if args.animate:
        if not args.video:
            sys.exit("--animate needs --video for the motion")
        if not os.path.isfile(args.animate):
            sys.exit(f"Image not found: {args.animate}")
        image = os.path.basename(args.animate)
        if os.path.abspath(args.animate) != os.path.abspath(os.path.join(COMFY_INPUT, image)):
            shutil.copyfile(args.animate, os.path.join(COMFY_INPUT, image))
        filename, frames, fps = prepare_video(args.video, args.seconds, FPS_14B)
        args.model, args.steps = "animate", ANIMATE_STEPS
        workflow = build_animate_workflow(args.prompt, image, filename, frames, fps, args.width, args.height, seed, args.pose_strength)
    else:
        workflow = build_workflow(args.prompt, args.width, args.height, args.seconds, args.steps, seed)
        if args.model == "14b":
            use_14b_model(workflow, args.width, args.height, args.seconds, args.steps, seed)
        if args.fast:
            use_turbo(workflow)
        if args.video:
            filename, frames, fps = prepare_video(args.video, args.seconds, FPS_14B if args.model == "14b" else FPS)
            use_video_input(workflow, filename, frames, fps, args.width, args.height, args.strength)
    add_finishing(workflow, args.width, args.height, args.tiled, args.upscale, args.smooth, args.fps)
    final_fps = workflow["10"]["inputs"]["fps"]
    if args.fps and final_fps != args.fps:
        print(f"Note: {args.fps} fps isn't available for this clip, so it will be {final_fps:g} fps")
    minutes = estimate_minutes(load_history(), args)
    print(f"Settings: {args.model}, {args.width}x{args.height}, {args.seconds:g}s, {args.steps} steps, {final_fps:g} fps")
    print(f"Prompt: {args.prompt}".encode("ascii", "replace").decode())
    print(f"Rough time: about {minutes:.0f} min, based on your past clips" if minutes else "Rough time: no similar clips logged yet", flush=True)
    if args.estimate:
        return
    if session is not None:
        save_session(args.session, session)
    if args.crf is not None:
        workflow["11"]["inputs"].update({"format.codec": "h264", "format.codec.encoding": "re-encode", "format.codec.encoding.crf": args.crf})
    try:
        prompt_id = request("/prompt", {"prompt": workflow})["prompt_id"]
    except urllib.error.HTTPError as error:
        sys.exit(f"ComfyUI rejected the job: {error.read().decode()[:1500]}")

    started = time.time()
    print(f"Queued, seed {seed}", flush=True)
    while True:
        time.sleep(10)
        history = request(f"/history/{prompt_id}").get(prompt_id)
        if not history:
            continue
        status = history["status"]
        if status.get("status_str") == "error":
            messages = [m[1] for m in status.get("messages", []) if m[0] == "execution_error"]
            sys.exit("Generation failed: " + "; ".join(m.get("exception_message", "") for m in messages))
        if status.get("completed"):
            print(f"Finished in {(time.time() - started) / 60:.1f} min")
            print(json.dumps(history["outputs"]))
            clip = history["outputs"]["11"]["images"][0]["filename"]
            log_run({
                "file": clip,
                "final": finish_clip(clip) if args.final else None,
                "time": time.strftime("%Y-%m-%d %H:%M"),
                "minutes": round((time.time() - started) / 60, 1),
                "prompt": args.prompt,
                "seed": seed,
                "seconds": args.seconds,
                "video": args.video,
                "strength": args.strength if args.video else None,
                "session": args.session,
                "animate": args.animate,
                "revised_from": args.revise,
                "change": change,
                "settings": {name: getattr(args, name) for name in SETTINGS},
            })
            if session is not None:
                session["clips"].append(history["outputs"]["11"]["images"][0]["filename"])
                save_session(args.session, session)
            return


if __name__ == "__main__":
    assert frame_count(2) == 49 and frame_count(5) == 121 and frame_count(3, FPS_14B) == 49
    assert [valid_frames(n) for n in (5, 6, 8, 9, 49, 50)] == [5, 5, 5, 9, 49, 49]
    high = build_workflow("p", 640, 352, 2, 20, 1)
    use_14b_model(high, 640, 352, 2, 20, 1)
    apply_strength_14b(high, 0.8)
    assert high["8"]["inputs"]["start_at_step"] == 4 and high["18"]["inputs"]["add_noise"] == "disable"
    low = build_workflow("p", 640, 352, 2, 20, 1)
    use_14b_model(low, 640, 352, 2, 20, 1)
    apply_strength_14b(low, 0.3)
    assert "8" not in low and low["18"]["inputs"]["start_at_step"] == 14 and low["18"]["inputs"]["latent_image"] == ["7", 0]
    turbo = build_workflow("p", 640, 352, 2, FAST_STEPS, 1)
    use_turbo(turbo)
    assert turbo["2"]["inputs"]["model"] == ["19", 0] and turbo["19"]["inputs"]["model"] == ["1", 0] and turbo["8"]["inputs"]["cfg"] == 1.0
    assert drop_options(["p", "--match-motion", "--strength", "0.6", "--fast", "--video=a.mp4", "--seconds", "2"], {"--match-motion"}, {"--strength", "--video"}) == ["p", "--fast", "--seconds", "2"]
    assert best_settings([]) == BEST_PRESET
    assert best_settings([{"rating": 3, "settings": {"steps": 1}}]) == BEST_PRESET
    assert best_settings([{"rating": 5, "settings": {"steps": 1}}, {"rating": 5, "settings": {"steps": 2}}, {"settings": {"steps": 3}}]) == {"steps": 2, "smooth": True}
    main()
