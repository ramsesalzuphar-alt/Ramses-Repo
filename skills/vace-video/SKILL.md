---
name: vace-video
description: The default way to make a video from just a prompt with the local Wan install. One prompt runs the whole chain - a matching motion capture take, a Blender depth render, a subject image, Wan 2.2 VACE 14B and a review - with no credits and no YouTube. Use when the user sends a video prompt, asks to make or generate a video, or invokes /vace-video, unless they ask for something below. NOT for reworking a clip they have (use rework-video), choosing every setting (use generate-video), a YouTube reference (use auto-video), or Higgsfield (use higgsfield-generate).
---

# VACE video from one prompt

The user gives a prompt. If it's missing, ask for it. Don't ask about anything else.

1. Check ComfyUI: `curl -s -m 3 http://127.0.0.1:8188/system_stats`. If it doesn't answer, run `powershell -File C:\wan-video-studio\start_studio.ps1` and wait until it answers.

2. Keep the computer awake (`request_keep_awake`, until `session_idle`), then run with the Bash tool using `run_in_background`, and wait to be notified:

   ```bash
   cd /c/wan-video-studio && PYTHONIOENCODING=utf-8 ~/ComfyUI/venv/Scripts/python.exe auto_vace.py "<prompt>" 2>&1 | grep -v '^{"\|Warning' | tail -n 30
   ```

   It takes about 2 hours at 832x480. Only change these if the user asks:
   - `--full` for 1280x704: much sharper, about 7 hours.
   - `--seconds N`: the default and longest is 2.5 (made at 32 real frames a second).
   - `--take 61_01` for a specific CMU motion, `--subject IMAGE` for their own picture of the subject.
   - `--tries N` remakes it with a new seed while the body check fails, keeping the best; use 2 for full size.
   - `--pick-only` shows which motion it would use, in seconds, without making anything.

   Tell the user they can watch it at http://127.0.0.1:8190/progress.

3. The output names the motion under `Motion:`, then `Body (try N):` for each try, `Review:`, `Body:` (and `Movement:` for a run), then `Best: <filename>`: the file is `$HOME/ComfyUI/output/video/<filename>`. Look at a few frames, then send it with SendUserFile. Say which motion it used, how long it took, and the review in plain words: jumps (frames that pop or morph), form breaks (frames where the body stops being a clear whole person) and smoothness from 0 to 1, plus the body check: which limbs went missing and when, the share of frames with the body whole, and whether colours drifted from the subject image. If the prompt had no matching human motion, it was made with plain Wan 14B; say so. If the script exits with an error, tell the user what it said.
