"""Find openly licensed images for a prompt through Openverse, and save one to use as a subject."""
import io
import json
import os
import re
import urllib.parse
import urllib.request

from PIL import Image

API = "https://api.openverse.org/v1/images/"
HEADERS = {"User-Agent": "WanStudio/1.0 (local video generator)"}
MAX_BYTES = 25 * 1024**2
MAX_SIDE = 2048
FILLER = set("a an and at the of on in to with for from by is are it its her his their make create generate video clip second seconds please show me i want some realistic cinematic shot".split())


def keywords(prompt):
    words = [w for w in re.findall(r"[a-z]+", prompt.lower()) if w not in FILLER and len(w) > 2]
    return " ".join(list(dict.fromkeys(words))[:6])


def fetch(url, limit):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=30) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("That image is too large")
    return data


def search(prompt, count=12):
    query = urllib.parse.urlencode({"q": keywords(prompt) or prompt, "page_size": count, "mature": "false", "license_type": "modification"})  # animating an image changes it, so no-derivatives licenses are out

    results = json.loads(fetch(f"{API}?{query}", 5 * 1024**2))["results"]
    return [{"id": r["id"], "title": r.get("title") or "Untitled", "creator": r.get("creator") or "Unknown", "license": f'{r.get("license", "").upper()} {r.get("license_version") or ""}'.strip(), "thumbnail": r.get("thumbnail"), "page": r.get("foreign_landing_url")} for r in results]


def save(image_id, folder):
    """Looks the address up from Openverse itself, so this can't be pointed at other sites, and re-saves it as a plain PNG."""
    if not re.fullmatch(r"[0-9a-f-]{36}", image_id):
        raise ValueError("Bad image id")
    record = json.loads(fetch(f"{API}{image_id}/", 1024**2))
    if not str(record.get("url", "")).startswith("https://"):
        raise ValueError("That image isn't available")
    image = Image.open(io.BytesIO(fetch(record["url"], MAX_BYTES))).convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"openverse_{image_id}.png")
    image.save(path)
    return path, {"title": record.get("title"), "creator": record.get("creator"), "license": record.get("license"), "page": record.get("foreign_landing_url")}


if __name__ == "__main__":
    assert keywords("Make a 5 second video of a guy doing parkour in a blue suit") == "guy doing parkour blue suit"
    for item in search("a young woman running on a beach", 3):
        print(item["license"], "|", item["title"][:50], "|", item["creator"][:24])
