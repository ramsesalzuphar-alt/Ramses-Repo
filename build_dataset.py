"""Collect short, openly licensed clips of one motion from Wikimedia Commons, for training a test LoRA.

Usage: python build_dataset.py "folder" "trigger caption" --clips 30 --query "sprinting" --query "jogging"

Only files whose license allows changes are used, and license details are saved in sources.json.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request

import imageio_ffmpeg
import torch

import practice

API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "WanStudio/1.0 (local video generator)"}
MAX_DOWNLOAD = 400 * 1024**2
SEGMENT_SECONDS = 3
FPS = 16
WIDTH, HEIGHT = 480, 480
ALLOWED = re.compile(r"^(cc0|public domain|pd\b|cc[ -]by( |-)|cc[ -]by-sa)", re.I)


def allowed_license(name):
    name = name or ""
    return bool(ALLOWED.search(name)) and not re.search(r"\b(nc|nd)\b", name, re.I)


def api(**params):
    url = API + "?" + urllib.parse.urlencode({"format": "json", **params})
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as response:
        return json.load(response)


def search(query, limit=50):
    data = api(action="query", generator="search", gsrsearch=f"{query} filetype:video", gsrnamespace=6, gsrlimit=limit, prop="imageinfo", iiprop="url|size|extmetadata|mediatype")
    for page in data.get("query", {}).get("pages", {}).values():
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {})
        yield {"title": page["title"], "url": info.get("url"), "size": info.get("size", 0), "page": info.get("descriptionurl"),
               "license": meta.get("LicenseShortName", {}).get("value"), "artist": re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", ""))[:120]}


def download(url, path):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120) as response, open(path, "wb") as f:
        total = 0
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_DOWNLOAD:
                raise ValueError("too large")
            f.write(chunk)


def full_body_runner(path):
    """A confident single person, tall in the frame, and uncut, in every sampled frame."""
    ok, _ = practice.usable_reference(path)
    if not ok:
        return False
    practice.person_in_frames(path, samples=2)  # loads the detector
    with practice.av.open(path) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    for rgb in (frames[0], frames[len(frames) // 2], frames[-1]):
        with torch.no_grad():
            result = practice.detector([torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255])[0]
        people = result["boxes"][(result["labels"] == 1) & (result["scores"] > 0.7)]
        if len(people) != 1:
            return False
        _, top, _, bottom = people[0].tolist()
        if not (0.35 < (bottom - top) / rgb.shape[0] < 0.98):
            return False
    return True


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("caption")
    parser.add_argument("--clips", type=int, default=30)
    parser.add_argument("--query", action="append", required=True)
    args = parser.parse_args()
    os.makedirs(args.folder, exist_ok=True)
    work = os.path.join(args.folder, "_downloads")
    os.makedirs(work, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    sources, seen, count = [], set(), 0
    for query in args.query:
        for item in search(query):
            if count >= args.clips:
                break
            if item["title"] in seen or not item["url"] or item["size"] > MAX_DOWNLOAD or not allowed_license(item["license"]):
                continue
            seen.add(item["title"])
            source = os.path.join(work, str(len(seen)) + os.path.splitext(item["url"])[1])
            try:
                download(item["url"], source)
                with practice.av.open(source) as container:
                    duration = (container.duration or 0) / 1e6
            except (OSError, ValueError, practice.av.FFmpegError) as error:
                print(f'Skipped {item["title"]}: {error}', flush=True)
                continue
            kept = 0
            for start in range(0, int(max(duration - SEGMENT_SECONDS, 0)), max(SEGMENT_SECONDS, int(duration / 12) or 1)):
                if count >= args.clips or kept >= 3:
                    break
                clip = os.path.join(args.folder, f"clip_{count + 1:03d}.mp4")
                fit = f"fps={FPS},scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT}"
                done = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", str(start), "-i", source, "-t", str(SEGMENT_SECONDS), "-an", "-vf", fit, "-c:v", "libx264", "-crf", "14", clip])
                try:
                    good = done.returncode == 0 and full_body_runner(clip)
                except (OSError, practice.av.FFmpegError, IndexError):
                    good = False
                if not good:
                    if os.path.exists(clip):
                        os.remove(clip)
                    continue
                count += 1
                kept += 1
                with open(clip[:-4] + ".txt", "w", encoding="utf-8") as f:
                    f.write(args.caption)
                sources.append({"clip": os.path.basename(clip), "start": start, **item})
                print(f'{count}/{args.clips}  {item["title"][:70]}  {item["license"]}', flush=True)
                with open(os.path.join(args.folder, "sources.json"), "w", encoding="utf-8") as f:
                    json.dump(sources, f, indent=1)
            os.remove(source)
    print(f"Done: {count} clips in {args.folder}")


if __name__ == "__main__":
    assert allowed_license("CC BY-SA 4.0") and allowed_license("CC0") and allowed_license("Public domain")
    assert not allowed_license("CC BY-NC 4.0") and not allowed_license("CC BY-ND 2.0") and not allowed_license("Copyrighted") and not allowed_license(None)
    main()
