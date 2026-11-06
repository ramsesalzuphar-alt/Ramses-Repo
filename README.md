# Wan Video Studio

Make videos locally with ComfyUI and open-source Wan 2.2 models, with no credits.

## Wan Studio page

Double-click `Start Wan Studio.bat` to start ComfyUI and the page, then bookmark `http://127.0.0.1:8190`. The
bookmark only works while they're running. The page has the prompt, your settings, uploads, past clips and ratings,
with a tab for each session that keeps its own clips, edits and memory. All clips shows everything.

Bookmark `http://127.0.0.1:8190/progress` too. It shows what's running from any session, Claude Code, the page or a
terminal: the ComfyUI job with a step bar and time left in the stage, queued jobs, generator scripts, growing downloads,
and the graphics card. It only watches, and it can't see Claude Code sessions themselves, only the jobs they start.

Each session tab also has a chat, run by a small local model (`llama3.2` through Ollama, on the CPU so it doesn't
take GPU memory). It turns your words into settings and asks about quality and frame rate. Code discards any setting
you didn't mention, and only the form's Generate button starts a video.

## Keeping the subject intact

`--animate IMAGE --video CLIP` animates the subject in the image with the motion from the clip, using the distilled
Wan Animate model. It keeps the subject far steadier than the other modes, and takes about 3 minutes for 2 seconds
at 704x480. The look comes from the image, so a prompt can't change an outfit: use a new image for that. It also
needs `wan_animate_2_distill_int8_convrot.safetensors`, `clip_vision_h.safetensors` and `Wan2_1_VAE_bf16.safetensors`
from Comfy-Org's `Wan-Animate-2` on Hugging Face.

## Practice

`practice.py "prompt" --subject image.png --runs 4` searches YouTube with words picked from the prompt, tries the clips
whose titles match those words first, skips ones with scene cuts, little movement or no single clear person, never reuses
a clip from earlier practice, makes a small Animate clip from each, and ranks them. It waits if the GPU goes above 85 C.
Reports are in `~/ComfyUI/output/practice`. The scores are rough, and the model doesn't change.

`motion_check.py clip.mp4` tracks the person's joints and compares the movement with the real clips saved in
`~/ComfyUI/reference`: knee bend, stride, arm swing, step rate, limbs keeping their length, and jitter. It gives a score
from 0 to 1 and says what stands out, such as "knees bend too little". Practice uses it in its ranking. The comparison
gets better as practice saves more real clips, and it runs on the CPU, so the first check of new clips is slow.

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
| `--auto-reference` | Pick search words from the prompt with `llama3.2` on the CPU, then use the most relevant YouTube clip with no scene cut and enough movement as `--video`, trying up to 4 | Off |
| `--reference-query` | Search words for `--auto-reference`, instead of picked ones | None |
| `--fast` | 5B only. The Turbo add-on makes the clip in 4 steps instead of 20. Needs `Wan22_TI2V_5B_Turbo_lora_rank_64_fp16.safetensors` in `models/loras`, from `hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF` on Hugging Face | Off |
| `--match-motion` | With `--video` or `--auto-reference`: make the clip at strength 0.6, then remake at 0.45 and 0.3 only while the motion match is under 0.85, grading each with `grade_video.py` and saving the grades in the history. Lower strengths copy motion more closely but can keep the reference's subject. `--fast` follows motion much less closely: 0.51 against 0.86 at 0.6 in a test | Off |
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
| From a reference clip, 704x480, 20 steps | 3.8 min |
| From a reference clip, 704x480, `--fast` (4 steps) | 1.2 min, or about 7 min when the model first loads |
| 1280x704, 30 steps | 13.5 min |
| 1280x704, 30 steps, `--tiled` | 6.7 min |
| 14B, 704x480, 49 frames, 20 steps, `--tiled` | 16.4 min |
| 14B, 1280x704, 49 frames, 20 steps, `--tiled` | 76.2 min |
| 14B from a reference clip, 704x480, 33 frames, strength 0.6 | 6.5 min |
| Animate, 704x480, 33 frames, smoothed to 48 fps | 2.7 min |
| Animate, 1280x704, 33 frames, smoothed to 48 fps | 24.4 min |
| Animate, 704x480 upscaled to 1408x960, 49 frames, 48 fps, `--final` | 6.2 min |

## Getting the best free result

Use `--animate` with a sharp subject image and a steady full-body reference, then `--upscale --final`. Add
`--tries 5` and keep the best, since results vary a lot by seed. `--join a.mp4 b.mp4 --music song.mp3` cuts your best
clips into one film. All of this is on the Wan Studio page too.

`--revise` and references run, but a revision at strength 0.5 did not follow a request to change a shirt's colour.
Motion and composition carry over better than specific prompt details.

## Claude Code skill

`skills/rework-video` lets you attach a clip and give a prompt in Claude Code. To install it, copy that
folder, `generate_video.py`, `reference_video.py` and `grade_video.py` into `~/.claude/skills/rework-video`.

`skills/auto-video` makes a video from just a prompt: `--auto-reference --match-motion` in one step. Copy its folder into
`~/.claude/skills/auto-video`; it runs the scripts installed with `rework-video`.

## Grading clips

```bash
"$HOME/ComfyUI/venv/Scripts/python.exe" grade_video.py clip.mp4 --reference ref.mp4
```

Motion match, 0 to 1, compares where each part of the picture moves, using optical flow. A reference graded against
itself is 1.0 and an unrelated clip about 0. Steadiness, 0 to 1, compares how cleanly each frame follows from the last
with the real reference; AI video shimmers and morphs between frames. It's a rough sign of real footage, not proof: it
can't see a wrong face or a plastic look. Asking `gemma4:e4b` to judge realism from frames was tried and dropped, since
it rated real YouTube clips no higher than generated ones.
