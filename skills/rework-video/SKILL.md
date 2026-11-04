---
name: rework-video
description: Rework a video the user uploads to match their prompt, using the local ComfyUI and Wan 2.2 install, with no credits. Use when the user attaches or names a video file and gives a prompt, or invokes /rework-video. NOT for generating a video with no input clip, or for Higgsfield generation (use higgsfield-generate).
---

# Rework video

The user gives a video file and a prompt. If either is missing, ask for it. Don't ask about anything else; the defaults are fine.

1. Check the server: `curl -s -m 3 http://127.0.0.1:8188/system_stats`. If it doesn't answer, start it in a Terminal panel tab titled "ComfyUI server" with `& "$HOME\ComfyUI\venv\Scripts\python.exe" "$HOME\ComfyUI\main.py" --port 8188`, writing the home folder out in full, and wait for it to answer.

2. Run this with the Bash tool using `run_in_background`, and wait to be notified:

   ```bash
   "$HOME/ComfyUI/venv/Scripts/python.exe" "$HOME/.claude/skills/rework-video/generate_video.py" "<prompt>" --video "<file path>" --strength 0.6
   ```

   Only add these if the user asks:
   - `--strength`: lower to stay closer to the clip ("keep it similar"), higher to change more. Above 0, at most 1.
   - `--seconds`: how much of the clip to use. The default is 2.
   - `--width` and `--height`: the default is 704 by 480. 1280 by 704 is sharper and slower.

3. The last line of output lists the result under `"11"`: the file is `$HOME/ComfyUI/output/<subfolder>/<filename>`. Send it with SendUserFile and say how long it took. If the script exits with an error, tell the user what it said.

The input is copied into `~/ComfyUI/input` under the same name, replacing any earlier file with that name.
