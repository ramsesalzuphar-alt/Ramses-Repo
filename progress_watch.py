"""What's running across every session: generator scripts, the ComfyUI job and its steps, downloads and the GPU.

Used by the Studio page at /progress. It only watches, so it's safe to leave open.
"""
import asyncio
import json
import os
import subprocess
import threading
import time
import urllib.request

import psutil

COMFY = "http://127.0.0.1:8188"
SCRIPTS = {"generate_video.py": "Video", "practice.py": "Practice", "reference_video.py": "Reference download", "grade_video.py": "Grading", "build_dataset.py": "Dataset"}
DOWNLOAD_FOLDERS = [os.path.expanduser(p) for p in ("~/ComfyUI/models", "~/ComfyUI/reference", "~/.ollama/models/blobs")]
PARTIAL = (".part", ".partial", "-partial", ".ytdl", ".tmp")
NEGATIVE_START = "blurry, low quality"
steps = {}  # prompt_id -> {"node", "value", "max", "first", "last"}
last_sizes = {}


def watch_comfy():
    """Follow ComfyUI's live feed for step progress. Reconnects if ComfyUI restarts."""
    import aiohttp

    async def follow():
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    # ComfyUI only accepts its own origin on the feed
                    async with session.ws_connect(f"{COMFY}/ws?clientId=studio-progress", headers={"Origin": COMFY}, heartbeat=30) as ws:
                        async for message in ws:
                            if message.type != aiohttp.WSMsgType.TEXT:
                                continue
                            event = json.loads(message.data)
                            data = event.get("data") or {}
                            if event.get("type") == "progress" and data.get("prompt_id"):
                                now = time.time()
                                old = steps.get(data["prompt_id"], {})
                                same = old.get("node") == data.get("node") and old.get("value", 0) <= data["value"]
                                steps[data["prompt_id"]] = {"node": data.get("node"), "value": data["value"], "max": data["max"], "first": old["first"] if same else now, "last": now}
            except (aiohttp.ClientError, OSError, asyncio.TimeoutError, ValueError):
                pass
            await asyncio.sleep(5)

    threading.Thread(target=lambda: asyncio.run(follow()), daemon=True).start()


def get_json(path):
    with urllib.request.urlopen(COMFY + path, timeout=3) as response:
        return json.loads(response.read())


def prompt_text(workflow):
    texts = [n["inputs"]["text"] for n in workflow.values() if n.get("class_type") == "CLIPTextEncode" and isinstance(n.get("inputs", {}).get("text"), str)]
    texts = [t for t in texts if not t.startswith(NEGATIVE_START)]
    return texts[0][:140] if texts else ""


def comfy_jobs():
    try:
        queue = get_json("/queue")
        # a running job has no start time yet, so the time it was queued stands in
        started = {j["id"]: j.get("execution_start_time") or j.get("create_time") for j in get_json("/api/jobs?limit=20").get("jobs", [])}
    except (OSError, ValueError):
        return None
    jobs = []
    for state, items in (("running", queue.get("queue_running", [])), ("queued", queue.get("queue_pending", []))):
        for item in items:
            prompt_id, workflow = item[1], item[2]
            job = {"state": state, "prompt": prompt_text(workflow)}
            if state == "running":
                if started.get(prompt_id):
                    job["elapsed"] = round(time.time() - started[prompt_id] / 1000)
                step = steps.get(prompt_id)
                if step:
                    node = workflow.get(str(step["node"]), {}).get("class_type", "")
                    job.update({"step": step["value"], "steps": step["max"], "stage": node})
                    if step["value"] > 1 and step["last"] > step["first"]:
                        per_step = (step["last"] - step["first"]) / (step["value"] - 1)
                        job["stage_left"] = round(per_step * (step["max"] - step["value"]))
            jobs.append(job)
    return jobs


def scripts():
    """Generator scripts started from any session, Claude Code, the Studio page or a terminal."""
    found = []
    for process in psutil.process_iter(["cmdline", "create_time"]):
        try:
            command = process.info["cmdline"] or []
        except (psutil.Error, OSError):
            continue
        for index, part in enumerate(command):
            name = os.path.basename(part)
            if name in SCRIPTS:
                words = [w for w in command[index + 1:] if not w.startswith("-")]
                found.append({"kind": SCRIPTS[name], "detail": (words[0] if words else "")[:140], "elapsed": round(time.time() - process.info["create_time"]), "pid": process.pid})
                break
    # a script that starts copies of itself shows once, as the oldest
    found.sort(key=lambda s: -s["elapsed"])
    unique, seen = [], set()
    for item in found:
        key = (item["kind"], item["detail"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def downloads():
    """Partly downloaded files that are still growing. Their full size isn't known, so they show size and speed."""
    now, found = time.time(), []
    for folder in DOWNLOAD_FOLDERS:
        for root, _, files in os.walk(folder):
            for name in files:
                if not name.endswith(PARTIAL):
                    continue
                path = os.path.join(root, name)
                try:
                    size, modified = os.path.getsize(path), os.path.getmtime(path)
                except OSError:
                    continue
                if now - modified > 120:
                    continue  # left over from an old, stopped download
                before = last_sizes.get(path)
                speed = (size - before[0]) / (now - before[1]) if before and now > before[1] else None
                last_sizes[path] = (size, now)
                found.append({"name": name, "mb": round(size / 1e6), "mb_per_s": None if speed is None else round(speed / 1e6, 1)})
    return found


def gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        load, used, total, temperature = (float(v) for v in out.split(","))
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return {"load": load, "memory_gb": round(used / 1024, 1), "memory_total_gb": round(total / 1024, 1), "temperature": temperature}


def snapshot(studio_job):
    return {"time": time.strftime("%H:%M:%S"), "comfy": comfy_jobs(), "scripts": scripts(), "downloads": downloads(), "gpu": gpu(), "studio": studio_job}


if __name__ == "__main__":
    workflow = {"4": {"class_type": "CLIPTextEncode", "inputs": {"text": "a dog runs"}}, "5": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE_START + ", distorted"}}}
    assert prompt_text(workflow) == "a dog runs"
    assert prompt_text({"5": workflow["5"]}) == ""
    print(json.dumps(snapshot(None), indent=1))
