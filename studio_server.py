"""Local page for the generator. Run it, then open http://127.0.0.1:8190."""
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import generate_video as g

HOST, PORT = "127.0.0.1", 8190
HERE = os.path.dirname(os.path.abspath(__file__))
UPLOADS = os.path.join(g.COMFY_INPUT, "uploads")
JOB_LOG = os.path.join(os.path.expanduser("~/ComfyUI/output"), "studio_job.log")
MAX_UPLOAD = 2 * 1024**3
# form fields passed straight to generate_video.py as --name value
VALUE_FIELDS = ("preset", "model", "width", "height", "seconds", "steps", "fps", "camera", "angle", "shot", "lens", "lighting", "style", "strength", "session", "video", "animate", "revise", "reference_query", "crf")
FLAG_FIELDS = ("tiled", "upscale", "auto_reference", "estimate")
job = None


def build_command(form):
    command = [sys.executable, os.path.join(HERE, "generate_video.py")]
    if str(form.get("prompt") or "").strip():
        command.append(str(form["prompt"]).strip())
    for name in VALUE_FIELDS:
        value = str(form.get(name) or "").strip()
        if value:
            command += ["--" + name.replace("_", "-"), value]
    for name in FLAG_FIELDS:
        if form.get(name):
            command.append("--" + name.replace("_", "-"))
    for item in form.get("remember") or []:
        command += ["--remember", str(item)]
    return command


def job_running():
    return job is not None and job.poll() is None


def run_quick(*arguments):
    result = subprocess.run([sys.executable, os.path.join(HERE, "generate_video.py"), *arguments], capture_output=True, text=True, encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    return result.returncode, (result.stdout + result.stderr).strip()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def respond(self, body, status=200, content_type="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def trusted(self):
        # any website could otherwise send commands to this local server from the user's browser
        origin = self.headers.get("Origin")
        allowed = (f"{HOST}:{PORT}", f"localhost:{PORT}")
        return self.headers.get("Host") in allowed and (not origin or origin in [f"http://{a}" for a in allowed])

    def do_GET(self):
        if not self.trusted():
            return self.respond({"error": "Not allowed"}, 403)
        url = urllib.parse.urlparse(self.path)
        if url.path == "/":
            with open(os.path.join(HERE, "studio.html"), "rb") as f:
                return self.respond(f.read(), content_type="text/html; charset=utf-8")
        if url.path == "/api/clips":
            return self.respond(list(reversed(g.load_history())))
        if url.path == "/api/sessions":
            names = [f[:-5] for f in os.listdir(g.SESSIONS) if f.endswith(".json")] if os.path.isdir(g.SESSIONS) else []
            return self.respond(sorted(names))
        if url.path == "/api/session":
            name = urllib.parse.parse_qs(url.query).get("name", [""])[0]
            if not re.fullmatch(r"[\w -]+", name):
                return self.respond({"error": "Bad session name"}, 400)
            return self.respond(g.load_session(name))
        if url.path == "/api/job":
            text = ""
            if os.path.exists(JOB_LOG):
                with open(JOB_LOG, encoding="utf-8", errors="replace") as f:
                    text = f.read()[-6000:]
            return self.respond({"running": job_running(), "code": None if job is None else job.poll(), "log": text})
        if url.path.startswith("/video/"):
            return self.send_video(urllib.parse.unquote(url.path[7:]))
        return self.respond({"error": "Not found"}, 404)

    def send_video(self, name):
        # basename only, so a request can't reach files outside the output folder
        path = os.path.join(g.OUTPUT_VIDEO, os.path.basename(name))
        if not path.endswith(".mp4") or not os.path.isfile(path):
            return self.respond({"error": "Not found"}, 404)
        size = os.path.getsize(path)
        start, end = 0, size - 1
        match = re.fullmatch(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
        if match:
            start = int(match[1])
            end = min(int(match[2]) if match[2] else end, end)
        if start > end:
            return self.respond({"error": "Bad range"}, 416)
        self.send_response(206 if match else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if match:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            self.wfile.write(f.read(end - start + 1))

    def read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def do_POST(self):
        global job
        if not self.trusted():
            return self.respond({"error": "Not allowed"}, 403)
        url = urllib.parse.urlparse(self.path)
        if url.path == "/api/upload":
            return self.upload(url)
        try:
            form = self.read_json()
        except json.JSONDecodeError:
            return self.respond({"error": "Bad request"}, 400)
        if url.path == "/api/generate":
            if job_running():
                return self.respond({"error": "A generation is already running"}, 409)
            os.makedirs(os.path.dirname(JOB_LOG), exist_ok=True)
            log = open(JOB_LOG, "w", encoding="utf-8")
            job = subprocess.Popen(build_command(form), stdout=log, stderr=subprocess.STDOUT, cwd=HERE, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            log.close()
            return self.respond({"started": True})
        if url.path == "/api/stop":
            if job_running():
                job.terminate()
                urllib.request.urlopen(urllib.request.Request(g.SERVER + "/interrupt", data=b"{}", headers={"Content-Type": "application/json"}), timeout=5)
            return self.respond({"stopped": True})
        if url.path == "/api/rate":
            code, text = run_quick("--rate", str(form.get("file")), str(form.get("score")))
            return self.respond({"ok": code == 0, "message": text}, 200 if code == 0 else 400)
        if url.path == "/api/memory":
            arguments = ["--session", str(form.get("session")), "--memory"]
            if form.get("forget"):
                arguments += ["--forget", str(form["forget"])]
            if form.get("remember"):
                arguments += ["--remember", str(form["remember"])]
            code, text = run_quick(*arguments)
            return self.respond({"ok": code == 0, "message": text}, 200 if code == 0 else 400)
        return self.respond({"error": "Not found"}, 404)

    def upload(self, url):
        name = os.path.basename(urllib.parse.parse_qs(url.query).get("name", [""])[0])
        name = re.sub(r"[^\w.-]", "_", name)
        length = int(self.headers.get("Content-Length") or 0)
        if not name.lower().endswith((".mp4", ".mov", ".webm", ".mkv", ".png", ".jpg", ".jpeg", ".webp")):
            return self.respond({"error": "Use an MP4, MOV, WEBM, MKV, PNG, JPG or WEBP file"}, 400)
        if not 0 < length <= MAX_UPLOAD:
            return self.respond({"error": "File must be under 2 GB"}, 400)
        os.makedirs(UPLOADS, exist_ok=True)
        path = os.path.join(UPLOADS, name)
        with open(path, "wb") as f:
            remaining = length
            while remaining:
                chunk = self.rfile.read(min(remaining, 1024 * 1024))
                if not chunk:
                    break
                f.write(chunk)
                remaining -= len(chunk)
        return self.respond({"path": path})


if __name__ == "__main__":
    command = build_command({"prompt": " p ", "preset": "draft", "tiled": True, "lens": "", "remember": ["a=b"]})
    assert command[2:] == ["p", "--preset", "draft", "--tiled", "--remember", "a=b"]
    print(f"Open http://{HOST}:{PORT}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
