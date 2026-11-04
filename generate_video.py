"""Queue a Wan 2.2 text-to-video job on the local ComfyUI server and wait for the file.

Usage: python generate_video.py "prompt" [--seconds 2] [--width 704] [--height 480] [--steps 20]
                                  [--video input.mp4] [--strength 0.6]

--video reworks an existing clip to match the prompt, keeping its motion and framing.
--strength is how much may change: low stays close to the clip, 1.0 ignores it.
"""
import argparse
import ctypes
import json
import os
import random
import shutil
import sys
import time
import urllib.error
import urllib.request

import av

SERVER = "http://127.0.0.1:8188"
COMFY_INPUT = os.path.expanduser("~/ComfyUI/input")
FPS = 24
NEGATIVE = (
    "blurry, low quality, distorted, deformed, extra limbs, extra fingers, bad anatomy, "
    "static, still frame, watermark, text, subtitles"
)


def valid_frames(frames):
    # Wan needs 4n+1 frames
    return frames - (frames - 1) % 4


def frame_count(seconds):
    return valid_frames(round(seconds * FPS) + 1)


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


def use_14b_model(workflow, width, height, seconds, steps, seed):
    """Swap the 5B model for Wan's two-stage 14B: high noise first, then low noise."""
    switch = steps // 2
    workflow["1"]["inputs"]["unet_name"] = "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
    workflow["6"]["inputs"]["vae_name"] = "wan_2.1_vae.safetensors"
    workflow["7"] = {"class_type": "EmptyHunyuanLatentVideo", "inputs": {"width": width, "height": height, "length": frame_count(seconds), "batch_size": 1}}
    workflow["16"] = {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}}
    workflow["17"] = {"class_type": "ModelSamplingSD3", "inputs": {"model": ["16", 0], "shift": 8.0}}
    shared = {"positive": ["4", 0], "negative": ["5", 0], "steps": steps, "cfg": 3.5, "sampler_name": "euler", "scheduler": "simple", "noise_seed": seed}
    workflow["8"] = {"class_type": "KSamplerAdvanced", "inputs": {**shared, "model": ["2", 0], "latent_image": ["7", 0], "add_noise": "enable", "start_at_step": 0, "end_at_step": switch, "return_with_leftover_noise": "enable"}}
    workflow["18"] = {"class_type": "KSamplerAdvanced", "inputs": {**shared, "model": ["17", 0], "latent_image": ["8", 0], "add_noise": "disable", "start_at_step": switch, "end_at_step": steps, "return_with_leftover_noise": "disable"}}
    workflow["9"]["inputs"]["samples"] = ["18", 0]


def prepare_video(path, seconds):
    """Copy the clip where ComfyUI can load it; return its name and usable frame count."""
    if not os.path.isfile(path):
        sys.exit(f"Video not found: {path}")
    try:
        with av.open(path) as container:
            stream = container.streams.video[0]
            # shortcut: frames are used at the clip's own rate, so 60 fps uses far more memory; resample if that fails
            fps = float(stream.average_rate)
            wanted = round(seconds * fps) + 1
            frames = 0
            for _ in container.decode(stream):
                frames += 1
                if frames >= wanted:
                    break
    except (av.FFmpegError, IndexError) as error:
        sys.exit(f"Could not read {path} as a video: {error}")
    if frames < 5:
        sys.exit(f"Video is too short: {frames} frames, need at least 5")
    target = os.path.join(COMFY_INPUT, os.path.basename(path))
    if os.path.abspath(path) != os.path.abspath(target):
        shutil.copyfile(path, target)
    return os.path.basename(path), valid_frames(frames), fps


def use_video_input(workflow, filename, frames, fps, width, height, strength):
    workflow["12"] = {"class_type": "LoadVideo", "inputs": {"file": filename}}
    workflow["13"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["12", 0]}}
    workflow["14"] = {"class_type": "ImageFromBatch", "inputs": {"image": ["13", 0], "batch_index": 0, "length": frames}}
    workflow["15"] = {"class_type": "ImageScale", "inputs": {"image": ["14", 0], "upscale_method": "lanczos", "width": width, "height": height, "crop": "center"}}
    workflow["7"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["15", 0], "vae": ["6", 0]}}
    workflow["8"]["inputs"]["denoise"] = strength
    workflow["10"]["inputs"]["fps"] = fps


def add_finishing(workflow, width, height, tiled, upscale, smooth):
    images = ["9", 0]
    if tiled:
        decode = workflow["9"]["inputs"]
        workflow["9"] = {"class_type": "VAEDecodeTiled", "inputs": {**decode, "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8}}
    if upscale:
        # RealESRGAN always gives 4x, which is too heavy to keep, so scale back to 2x
        workflow["19"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "RealESRGAN_x4plus.safetensors"}}
        workflow["20"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["19", 0], "image": images}}
        workflow["21"] = {"class_type": "ImageScale", "inputs": {"image": ["20", 0], "upscale_method": "lanczos", "width": width * 2, "height": height * 2, "crop": "disabled"}}
        images = ["21", 0]
    if smooth:
        workflow["22"] = {"class_type": "FrameInterpolationModelLoader", "inputs": {"model_name": "rife_v4.26.safetensors"}}
        workflow["23"] = {"class_type": "FrameInterpolate", "inputs": {"interp_model": ["22", 0], "images": images, "multiplier": 2}}
        images = ["23", 0]
        workflow["10"]["inputs"]["fps"] *= 2
    workflow["10"]["inputs"]["images"] = images


def request(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(SERVER + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--seconds", type=float, default=2)
    parser.add_argument("--width", type=int, default=704)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--model", choices=["5b", "14b"], default="5b", help="14b is better but much slower, and text-only")
    parser.add_argument("--tiled", action="store_true", help="decode in pieces: less memory, small risk of seams")
    parser.add_argument("--upscale", action="store_true", help="double the resolution with RealESRGAN")
    parser.add_argument("--smooth", action="store_true", help="double the frame rate with RIFE")
    parser.add_argument("--crf", type=float, help="H.264 quality: lower is better and larger, 18 is very good")
    parser.add_argument("--video", help="existing clip to rework")
    parser.add_argument("--strength", type=float, default=0.6)
    args = parser.parse_args()
    if not 0 < args.strength <= 1:
        sys.exit("--strength must be above 0 and at most 1")
    if args.model == "14b" and args.video:
        sys.exit("--video only works with --model 5b")

    # Block idle sleep while this process runs; ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

    seed = random.randrange(2**32)
    workflow = build_workflow(args.prompt, args.width, args.height, args.seconds, args.steps, seed)
    if args.model == "14b":
        use_14b_model(workflow, args.width, args.height, args.seconds, args.steps, seed)
    if args.video:
        filename, frames, fps = prepare_video(args.video, args.seconds)
        use_video_input(workflow, filename, frames, fps, args.width, args.height, args.strength)
    add_finishing(workflow, args.width, args.height, args.tiled, args.upscale, args.smooth)
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
            return


if __name__ == "__main__":
    assert frame_count(2) == 49 and frame_count(5) == 121
    assert [valid_frames(n) for n in (5, 6, 8, 9, 49, 50)] == [5, 5, 5, 9, 49, 49]
    main()
