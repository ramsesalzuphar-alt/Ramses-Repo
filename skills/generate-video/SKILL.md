---
name: generate-video
description: Generate a video locally with ComfyUI and Wan, letting the user choose the size, model, frame rate, camera and other settings each time. Use when the user asks to generate or make a video with the local generator, or invokes /generate-video. NOT for Higgsfield generation (use higgsfield-generate), or for reworking a clip with no settings to choose (use rework-video).
---

# Generate video

The user gives a prompt. If it's missing, ask for it.

1. Unless the user already gave these or said to use defaults, ask with AskUserQuestion, in one call:
   - **Demand** (header "Quality"): Draft (5B, 704x480, about 2 minutes), Balanced (14B, 704x480, about 15 minutes), Max (14B, 1280x704, over an hour).
   - **Frame rate** (header "Frame rate"): 48 fps (smoothest, a little slower), 32 fps, 16 fps (14B's natural rate, can look choppy). For Draft, offer 48 and 24 instead.
   - **Camera** (header "Camera"): Tracking, Static, Handheld, Dolly in. "Other" covers `pan`, `orbit`, `dolly-out` and `drone`.
   - **Framing** (header "Framing"): Full body, Medium, Close-up, Wide.

   Only ask about length, angle, lens, lighting or style if the user brings them up.

2. Check the server: `curl -s -m 3 http://127.0.0.1:8188/system_stats`. If it doesn't answer, start it in a Terminal panel tab titled "ComfyUI server" with `& "$HOME\ComfyUI\venv\Scripts\python.exe" "$HOME\ComfyUI\main.py" --port 8188`, writing the home folder out in full, and wait for it to answer.

3. Show the settings and time with `--estimate`, then run the same command without it, using the Bash tool with `run_in_background`:

   ```bash
   "$HOME/ComfyUI/venv/Scripts/python.exe" "$HOME/.claude/skills/rework-video/generate_video.py" "<prompt>" --preset balanced --fps 48 --camera tracking --shot full-body
   ```

   | Option | Values |
   |---|---|
   | `--preset` | `draft`, `balanced`, `max` |
   | `--fps` | 14B: 16, 32, 48. 5B: 24, 48 |
   | `--camera` | `static`, `tracking`, `dolly-in`, `dolly-out`, `pan`, `orbit`, `handheld`, `drone` |
   | `--angle` | `eye-level`, `low`, `high`, `overhead`, `side`, `front`, `back` |
   | `--shot` | `wide`, `full-body`, `medium`, `close-up` |
   | `--lens` | mm, such as 35 or 85 |
   | `--lighting`, `--style` | Free text |
   | `--seconds`, `--width`, `--height`, `--steps`, `--model` | Override the preset. Width and height must be multiples of 16 |

4. The last line of output lists the result under `"11"`: the file is `$HOME/ComfyUI/output/<subfolder>/<filename>`. Send it with SendUserFile and say how long it took. If the script exits with an error, tell the user what it said.

## Sessions and subjects

Use `--session <name>` on every generation in a project. Save lasting details as `--remember key=value`, and when the user changes one, replace it under the same key so instructions don't conflict. For an edit, use `--session <name> --revise latest "<the change>"`. `--memory` shows what's saved, and `--forget key` removes something.

To keep a subject intact, use `--animate <image> --video <motion clip>`, which takes about 3 minutes at 704x480. A prompt can't change how that subject looks, so tell the user a new image is needed for that.

The user can also use the page at `http://127.0.0.1:8190`, started with `C:\wan-video-studio\Start Wan Studio.bat`.

Camera and framing are added to the prompt as text, so the model may not follow them exactly. Say so if a result ignores them. The time is a rough guide from past clips.
