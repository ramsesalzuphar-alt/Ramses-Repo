"""Find video links and save short reference clips as MP4s for the generator.

Usage: python reference_video.py search "query" [--count 5]
       python reference_video.py get URL [--start 0] [--seconds 3]
       python reference_video.py auto "prompt" [--seconds 3]
"""
import argparse
import json
import os
import re
import shutil
import sys
import urllib.error
import urllib.request

import av
import imageio_ffmpeg
import numpy as np
import yt_dlp
from yt_dlp.utils import DownloadError, download_range_func

OUTPUT = os.path.expanduser("~/ComfyUI/reference")
MAX_SECONDS = 30
MAX_DOWNLOADS = 4
MAX_WORDS = 6
OLLAMA = "http://127.0.0.1:11434/api/chat"
WORDS_MODEL = "llama3.2:latest"
WORDS_PROMPT = (
    "Turn this video prompt into 3 to 6 YouTube search words that find real footage of the same subject doing the same "
    "motion. Keep what the subject is, such as an animal's breed, because a look-alike subject lets the motion carry "
    "over. Leave out clothing, style and camera words. Examples: 'a man in a blue suit walks down a rainy street at "
    "night, film noir' gives 'man walking street rain', and 'a golden retriever runs on a beach, 4k' gives 'golden "
    "retriever running beach'."
)


def search(query, count):
    # shortcut: yt-dlp has no TikTok search, so TikTok links have to come from the user
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True}) as ydl:
        results = ydl.extract_info(f"ytsearch{count}:{query}", download=False)["entries"]
    return list(results)


# the small model often keeps these, and they steer YouTube towards edits and trailers instead of plain footage
# shortcut: a fixed list, so add words here if searches keep picking up style terms
STYLE_PHRASES = re.compile(
    r"\b(golden hour|slow motion|film grain|(tracking|wide|medium|full body|aerial|drone|establishing) shot|close-up|closeup"
    r"|cinematic|footage|film|handheld|4k|8k|hdr|photorealistic|bokeh|\d+ ?mm)\b",
    re.IGNORECASE,
)


def clean_words(text):
    """Model output is untrusted: keep only a short plain search phrase without style words, or None."""
    if not isinstance(text, str):
        return None
    # the small model sometimes rambles, and the first few words are the useful ones
    words = " ".join(STYLE_PHRASES.sub(" ", text.replace('"', " ").replace(",", " ")).split()[:MAX_WORDS])
    return words[:80] if len(words) >= 2 else None


def search_words(prompt):
    """Short search words from the local model, on the CPU so it can't take GPU memory; the prompt's start if it's not running."""
    fallback = " ".join(prompt.split()[:12])
    body = {
        "model": WORDS_MODEL,
        "stream": False,
        "format": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        "options": {"num_gpu": 0, "temperature": 0, "num_predict": 60},
        "messages": [{"role": "system", "content": WORDS_PROMPT}, {"role": "user", "content": prompt}],
    }
    request = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            words = clean_words(json.loads(json.loads(response.read())["message"]["content"]).get("query"))
    except (OSError, ValueError, KeyError, AttributeError) as error:
        print(f"Couldn't get search words from Ollama ({error}), so using the prompt's start", flush=True)
        return fallback
    if not words:
        print("Ollama gave no usable search words, so using the prompt's start", flush=True)
    return words or fallback


def gray_frames(path, limit=200):
    with av.open(path) as container:
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="gray").astype(np.int16))
            if len(frames) >= limit:
                break
    return frames


def frame_changes(frames):
    return np.array([np.abs(frames[i + 1] - frames[i]).mean() for i in range(len(frames) - 1)])


def usable_reference(path):
    """Reject clips with a scene cut, or with too little movement to copy."""
    frames = gray_frames(path)
    if len(frames) < 10:
        return False, "too short"
    changes = frame_changes(frames)
    median = float(np.median(changes))
    if median < 1.0:
        return False, "too little movement"
    if changes.max() > max(median * 5, 25):
        return False, "has a scene cut"
    return True, "ok"


def find_reference(query, seconds):
    """Save a clip from the most relevant result with no scene cut and enough movement; return its path."""
    print(f"Searching YouTube: {query}", flush=True)
    candidates = [v for v in search(query, 10) if 5 <= (v.get("duration") or 0) <= 600]
    for video in candidates[:MAX_DOWNLOADS]:
        try:
            # a little way in, since clips often open on a title card before the action starts
            path = get(f'https://www.youtube.com/watch?v={video["id"]}', 1, seconds)
        except (DownloadError, SystemExit, OSError) as error:
            print(f'Skipped "{video.get("title")}": {error}', flush=True)
            continue
        ok, reason = usable_reference(path)
        if ok:
            return path
        print(f'Skipped "{video.get("title")}": {reason}', flush=True)
    sys.exit(f"No usable reference found for: {query}. Try --reference-query with different words.")


def get(url, start, seconds):
    os.makedirs(OUTPUT, exist_ok=True)
    # partial downloads only look for a program named exactly "ffmpeg" on PATH
    scripts = os.path.dirname(sys.executable)
    ffmpeg = os.path.join(scripts, "ffmpeg.exe")
    if not os.path.isfile(ffmpeg):
        shutil.copyfile(imageio_ffmpeg.get_ffmpeg_exe(), ffmpeg)
    os.environ["PATH"] = scripts + os.pathsep + os.environ["PATH"]
    options = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        # no audio is needed, and capping at 1080p avoids huge files
        # 30 fps or less when offered, because the generator uses every frame
        "format": "bv*[height<=1080][fps<=30][vcodec^=avc1]/bv*[height<=1080][fps<=30]/bv*[height<=1080][vcodec^=avc1]/bv*[height<=1080]/b[height<=1080]",
        "merge_output_format": "mp4",
        "postprocessors": [{"key": "FFmpegVideoConvertor", "preferedformat": "mp4"}],
        "download_ranges": download_range_func(None, [(start, start + seconds)]),
        "force_keyframes_at_cuts": True,
        "outtmpl": os.path.join(OUTPUT, "%(extractor)s_%(id)s_" + f"{int(start)}s.%(ext)s"),
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        path = os.path.splitext(ydl.prepare_filename(info))[0] + ".mp4"
    if not os.path.isfile(path):
        sys.exit(f"Download finished but the MP4 is missing: {path}")
    print(f'Saved: {path}\nTitle: {info.get("title")}\nFrom: {info.get("uploader")}\nLink: {url}', flush=True)
    return path


def main():
    # titles can hold characters the Windows console encoding can't print
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    find = commands.add_parser("search")
    find.add_argument("query")
    find.add_argument("--count", type=int, default=5)
    fetch = commands.add_parser("get")
    fetch.add_argument("url")
    fetch.add_argument("--start", type=float, default=0)
    fetch.add_argument("--seconds", type=float, default=3)
    auto = commands.add_parser("auto", help="pick search words from a prompt and save the best clip")
    auto.add_argument("prompt")
    auto.add_argument("--seconds", type=float, default=3)
    args = parser.parse_args()
    try:
        if args.command == "search":
            for video in search(args.query, args.count):
                print(f'{video.get("duration") or "?"}s  {video.get("title")}  https://www.youtube.com/watch?v={video["id"]}')
        elif args.command == "auto":
            find_reference(search_words(args.prompt), args.seconds)
        else:
            if not 0 < args.seconds <= MAX_SECONDS or args.start < 0:
                sys.exit(f"--start can't be negative, and --seconds must be above 0 and at most {MAX_SECONDS}")
            get(args.url, args.start, args.seconds)
    except DownloadError as error:
        sys.exit(f"Failed: {error}")


if __name__ == "__main__":
    assert clean_words('  "woman  jogging beach" ') == "woman jogging beach"
    assert clean_words("golden retriever catching frisbee park slow motion film grain") == "golden retriever catching frisbee park"
    assert clean_words("astronaut dancing on moon wide shot 35mm film footage") == "astronaut dancing on moon"
    assert clean_words("woman jogging beach golden hour cinematic tracking shot") == "woman jogging beach"
    assert clean_words("x" * 200) == "x" * 80 and clean_words(None) is None and clean_words("") is None
    assert clean_words("woman jogging beach golden hour cinematic tracking shot outdoor running woman on beach sunset") == "woman jogging beach outdoor running woman"
    main()
