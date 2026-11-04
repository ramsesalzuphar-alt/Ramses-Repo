# Wan Video Studio

Make videos locally with ComfyUI and open-source Wan 2.2 models, with no credits.

## Wan Studio page

Double-click `Start Wan Studio.bat` to start ComfyUI and the page, then bookmark `http://127.0.0.1:8190`. The
bookmark only works while they're running. The page has the prompt, your settings, uploads, past clips and ratings,
with a tab for each session that keeps its own clips, edits and memory. All clips shows everything.

## Keeping the subject intact

`--animate IMAGE --video CLIP` animates the subject in the image with the motion from the clip, using the distilled
Wan Animate model. It keeps the subject far steadier than the other modes, and takes about 3 minutes for 2 seconds
at 704x480. The look comes from the image, so a prompt can't change an outfit: use a new image for that. It also
needs `wan_animate_2_distill_int8_convrot.safetensors`, `clip_vision_h.safetensors` and `Wan2_1_VAE_bf16.safetensors`
from Comfy-Org's `Wan-Animate-2` on Hugging Face.

## Sessions

`--session NAME` saves your scene and clips in `~/ComfyUI/output/sessions`, so nothing you asked for gets lost.

```bash
"$HOME/ComfyUI/venv/Scripts/python.exe" generate_video.py "your scene" --session beach --remember "outfit=grey t-shirt"
```
```bash
"$HOME/ComfyUI/venv/Scripts/python.exe" generate_video.py "she runs at an easy pace" --session beach --revise latest
```

`--remember KEY=VALUE` saves or replaces an instruction, `--forget KEY` removes one, and `--memory` shows them.
`--revise latest` starts from the session's newest clip.

## Setup

ComfyUI is expected at `~/ComfyUI` with its own Python environment in `venv`. Start it first:

```bash
"$HOME/ComfyUI/venv/Scripts/python.exe" "$HOME/ComfyUI/main.py" --port 8188
```

Model files go in `~/ComfyUI/models`:

| File | Folder | Needed for |
|---|---|---|
| `wan2.2_ti2v_5B_fp16.safetensors` | `diffusion_models` | 5B model |
| `wan2.2_vae.safetensors` | `vae` | 5B model |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `text_encoders` | All models |
| `wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors` | `diffusion_models` | `--model 14b` |
| `wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors` | `diffusion_models` | `--model 14b` |
| `wan_2.1_vae.safetensors` | `vae` | `--model 14b` |
| `RealESRGAN_x4plus.safetensors` | `upscale_models` | `--upscale` |
| `rife_v4.26.safetensors` | `frame_interpolation` | `--smooth` |

The Wan files are from Comfy-Org's `Wan_2.2_ComfyUI_Repackaged` on Hugging Face, the upscaler from
`Real-ESRGAN_repackaged`, and the smoothing model from `frame_interpolation`.

## Use

```bash
"$HOME/ComfyUI/venv/Scripts/python.exe" generate_video.py "your prompt"
```

Clips are saved to `~/ComfyUI/output/video`.

| Option | What it does | Default |
|---|---|---|
| `--seconds` | Clip length | 2 |
| `--width`, `--height` | Size. 1280 by 704 is the 5B model's native size | 704 by 480 |
| `--steps` | More is slightly better and slower | 20 |
| `--video`, `--strength` | Rework a clip. Lower strength stays closer to it | None, 0.6 |
| `--auto-reference` | Find a YouTube clip matching the prompt and use it as `--video` | Off |
| `--reference-query` | Search words for `--auto-reference`, instead of the prompt | None |
| `--revise FILE` | Develop a logged clip. The prompt is your change; its settings and seed are kept, at strength 0.5 | None |
| `--seed` | Repeat a clip's randomness | Random |
| `--model` | `5b` or `14b`. 14B looks more consistent but is much slower | `5b` |
| `--preset` | `draft`, `balanced` or `max`. Other options override it | None |
| `--fps` | Final frame rate. 14B: 16, 32 or 48. 5B: 24 or 48 | 16 smoothed to 48 on 14B, 24 on 5B |
| `--camera`, `--angle`, `--shot` | Added to the prompt as text, so the model may not follow them exactly | None |
| `--lens`, `--lighting`, `--style` | Also added to the prompt as text | None |
| `--estimate` | Show the settings and a rough time, then stop | Off |
| `--tiled` | Decode in pieces: less memory, small risk of seams | Off |
| `--upscale` | Double the resolution | Off |
| `--smooth` | Double the frame rate | Off |
| `--crf` | H.264 quality. Lower is better and larger | Automatic |

## Best settings and ratings

Every finished clip is logged to `~/ComfyUI/output/history.jsonl` with its prompt, seed, settings and time.

```bash
"$HOME/ComfyUI/venv/Scripts/python.exe" generate_video.py "your prompt" --best
```
```bash
"$HOME/ComfyUI/venv/Scripts/python.exe" generate_video.py --history
```
```bash
"$HOME/ComfyUI/venv/Scripts/python.exe" generate_video.py --rate wan_00006_.mp4 5
```

`--best` reuses the settings of your top-rated clip, 4 or 5 out of 5, with the newest winning a tie. It always
turns smoothing on. With no such rating, it uses 1280x704, 30 steps, tiled decoding and `--crf 12`. Any option
you type yourself overrides it. Ratings choose settings only; the model itself doesn't learn from them.

## Timings on an RTX 3070 Laptop (8 GB)

All 2 seconds, using the 5B model.

| Settings | Time |
|---|---|
| 704x480, 20 steps | 1.7 min |
| 704x480, 20 steps, upscaled and smoothed | 3.2 min |
| 1280x704, 30 steps | 13.5 min |
| 1280x704, 30 steps, `--tiled` | 6.7 min |
| 14B, 704x480, 49 frames, 20 steps, `--tiled` | 16.4 min |
| 14B, 1280x704, 49 frames, 20 steps, `--tiled` | 76.2 min |
| 14B from a reference clip, 704x480, 33 frames, strength 0.6 | 6.5 min |
| Animate, 704x480, 33 frames, smoothed to 48 fps | 2.7 min |
| Animate, 1280x704, 33 frames, smoothed to 48 fps | 24.4 min |

`--revise` and references run, but a revision at strength 0.5 did not follow a request to change a shirt's colour.
Motion and composition carry over better than specific prompt details.

## Claude Code skill

`skills/rework-video` lets you attach a clip and give a prompt in Claude Code. To install it, copy that
folder and `generate_video.py` into `~/.claude/skills/rework-video`.
