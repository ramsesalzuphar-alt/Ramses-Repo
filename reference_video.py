"""Find video links and save short reference clips as MP4s for the generator.

Usage: python reference_video.py search "query" [--count 5]
       python reference_video.py get URL [--start 0] [--seconds 3]
"""
import argparse
import os
import shutil
import sys

import imageio_ffmpeg
import yt_dlp
from yt_dlp.utils import DownloadError, download_range_func

OUTPUT = os.path.expanduser("~/ComfyUI/reference")
MAX_SECONDS = 30


def search(query, count):
    # shortcut: yt-dlp has no TikTok search, so TikTok links have to come from the user
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True}) as ydl:
        results = ydl.extract_info(f"ytsearch{count}:{query}", download=False)["entries"]
    return list(results)


def find_reference(query, seconds):
    """Save a clip from the first search result of a usable length; return its path."""
    for video in search(query, 10):
        if 5 <= (video.get("duration") or 0) <= 600:
            return get(f'https://www.youtube.com/watch?v={video["id"]}', 0, seconds)
    sys.exit(f"No usable reference found for: {query}")


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
    args = parser.parse_args()
    try:
        if args.command == "search":
            for video in search(args.query, args.count):
                print(f'{video.get("duration") or "?"}s  {video.get("title")}  https://www.youtube.com/watch?v={video["id"]}')
        else:
            if not 0 < args.seconds <= MAX_SECONDS or args.start < 0:
                sys.exit(f"--start can't be negative, and --seconds must be above 0 and at most {MAX_SECONDS}")
            get(args.url, args.start, args.seconds)
    except DownloadError as error:
        sys.exit(f"Failed: {error}")


if __name__ == "__main__":
    main()
