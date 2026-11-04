# Wan Video Studio

Make videos locally with ComfyUI and open-source Wan 2.2 models, with no credits.

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
| `--video`, `--strength` | Rework a clip (5B only). Lower strength stays closer to it | None, 0.6 |
| `--model` | `5b` or `14b`. 14B is text-only and not yet tested | `5b` |
| `--tiled` | Decode in pieces: less memory, small risk of seams | Off |
| `--upscale` | Double the resolution | Off |
| `--smooth` | Double the frame rate | Off |
| `--crf` | H.264 quality. Lower is better and larger | Automatic |

Example at the highest settings tested:

```bash
"$HOME/ComfyUI/venv/Scripts/python.exe" generate_video.py "your prompt" --width 1280 --height 704 --steps 30 --crf 12
```

## Timings on an RTX 3070 Laptop (8 GB)

All 2 seconds, using the 5B model.

| Settings | Time |
|---|---|
| 704x480, 20 steps | 1.7 min |
| 704x480, 20 steps, upscaled and smoothed | 3.2 min |
| 1280x704, 30 steps | 13.5 min |

## Claude Code skill

`skills/rework-video` lets you attach a clip and give a prompt in Claude Code. To install it, copy that
folder and `generate_video.py` into `~/.claude/skills/rework-video`.
