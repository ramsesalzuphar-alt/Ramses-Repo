"""Chat for the Studio page, using a small local model through Ollama.

The model only proposes settings. Everything it returns is checked against a fixed list here,
so a strange reply can't reach files, commands or options outside the form.
"""
import json
import os
import re
import urllib.request

import generate_video as g

OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL = "llama3.2:latest"
TEXT_LIMITS = {"prompt": 1000, "lighting": 200, "style": 200}
CHOICES = {"preset": ("draft", "balanced", "max"), "fps": (16, 24, 32, 48), "camera": tuple(g.CAMERA), "angle": tuple(g.ANGLE), "shot": tuple(g.SHOT)}
PRESET_LABELS = {"draft": "Draft: about 2 minutes, lowest quality", "balanced": "Balanced: about 6 to 16 minutes", "max": "Max: about an hour or more, sharpest"}
HISTORY_SENT = 12
SYSTEM = f"""You are the assistant inside a local AI video generator. Reply in plain, friendly, short sentences.
Turn what the user says into settings. Return JSON only, with these keys:
"reply": your short message to the user.
"settings": only what the user clearly asked for. Never guess. Possible keys:
  "prompt": a clear description of the video, without technical settings. When editing, only the change.
  "preset": draft, balanced or max.
  "fps": 16, 24, 32 or 48.
  "seconds": a number from 0.5 to 10.
  "camera": one of {", ".join(g.CAMERA)}.
  "angle": one of {", ".join(g.ANGLE)}.
  "shot": one of {", ".join(g.SHOT)}.
  "lens": millimetres from 10 to 300.
  "lighting" and "style": short text.
"edit": true only if the user wants to change their latest clip instead of making a new one.
"question": if something creative is unclear and you must not assume it, ask one short question. Otherwise an empty string.
"options": 2 to 4 short answers to your question, or an empty list.
Do not ask about quality or frame rate, as the page asks those. Do not claim a video has been made."""
SCHEMA = {"type": "object", "required": ["reply", "settings", "edit", "question", "options"], "properties": {
    "reply": {"type": "string"}, "settings": {"type": "object"}, "edit": {"type": "boolean"},
    "question": {"type": "string"}, "options": {"type": "array", "items": {"type": "string"}}}}


def clean_settings(settings):
    """Keep only values the form allows."""
    clean = {}
    if not isinstance(settings, dict):
        return clean
    for name, limit in TEXT_LIMITS.items():
        value = settings.get(name)
        if isinstance(value, str) and value.strip():
            # a leading dash would read as a command-line option
            clean[name] = value.strip().lstrip("- ")[:limit]
    for name, allowed in CHOICES.items():
        value = settings.get(name)
        if name == "fps" and isinstance(value, str) and value.isdigit():
            value = int(value)
        if value in allowed:
            clean[name] = value
    for name, low, high in (("seconds", 0.5, 10), ("lens", 10, 300)):
        value = settings.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and low <= value <= high:
            clean[name] = value
    return {name: value for name, value in clean.items() if value != ""}


def keep_mentioned(settings, message):
    """Drop any setting the user didn't actually say: small models invent values whatever the instructions say."""
    text = re.sub(r"[-_]", " ", message.lower())
    kept = {}
    for name, value in settings.items():
        if name == "prompt":
            kept[name] = value
        elif name in ("lighting", "style"):
            if any(word in text for word in re.findall(r"[a-z]{3,}", value.lower())):
                kept[name] = value
        elif name in ("seconds", "lens", "fps"):
            # whole numbers only, so a 5 can't match inside 50 or 15
            if re.search(rf"(?<![\d.]){value:g}(?![\d.])", text):
                kept[name] = value
        elif str(value).replace("-", " ") in text:
            kept[name] = value
    return kept


def required_questions(draft):
    """What the generator won't assume, in order. Fixed here so a weak model can't skip them."""
    if "prompt" not in draft:
        return []
    if "preset" not in draft:
        return [{"field": "preset", "question": "How long can this take?", "options": [{"label": PRESET_LABELS[p], "value": p} for p in CHOICES["preset"]]}]
    rates = (24, 48) if draft["preset"] == "draft" else (16, 32, 48)
    if draft.get("fps") not in rates:
        return [{"field": "fps", "question": "Which frame rate?", "options": [{"label": f"{r} fps" + (", smoothest" if r == 48 else ""), "value": r} for r in rates]}]
    return []


def path_for(session):
    return g.session_path(session)[:-5] + ".chatlog"


def load_chat(session):
    path = path_for(session)
    if not os.path.exists(path):
        return {"messages": [], "draft": {}, "edit": False, "question": None}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_chat(session, chat):
    os.makedirs(g.SESSIONS, exist_ok=True)
    path = path_for(session)
    # write then swap, so a crash can't lose the conversation
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(chat, f, indent=1)
    os.replace(path + ".tmp", path)


def ask_model(chat, has_clips):
    context = f"Current settings: {json.dumps(chat['draft'])}. This session {'has a latest clip to edit' if has_clips else 'has no clips yet, so nothing can be edited'}."
    messages = [{"role": "system", "content": SYSTEM + "\n" + context}]
    messages += [{"role": m["role"], "content": m["content"]} for m in chat["messages"][-HISTORY_SENT:]]
    # on the CPU, so chatting can't take GPU memory from a render
    body = {"model": MODEL, "messages": messages, "stream": False, "format": SCHEMA, "options": {"num_gpu": 0, "temperature": 0.2}}
    request = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=240) as response:
        return json.loads(json.load(response)["message"]["content"])


def view(chat):
    """What the page needs: messages, the next choices, and the settings once nothing is left to ask."""
    required = required_questions(chat["draft"])
    # the model's own question is optional, so a poor one can't block generating
    questions = required[:1] or ([chat["question"]] if chat.get("question") else [])
    return {"messages": chat["messages"], "draft": chat["draft"], "edit": chat["edit"], "questions": questions, "ready": "prompt" in chat["draft"] and not required}


def handle(session, form, has_clips):
    chat = load_chat(session)
    if form.get("reset"):
        # keep the user's quality choices for the next clip, but not the finished prompt
        chat["draft"] = {k: v for k, v in chat["draft"].items() if k in ("preset", "fps", "seconds")}
        chat["edit"] = False
        chat["question"] = None
    elif form.get("answer"):
        answer = form["answer"]
        accepted = clean_settings({answer.get("field"): answer.get("value")})
        if not accepted:
            raise ValueError("That option isn't available")
        chat["draft"].update(accepted)
        chat["messages"].append({"role": "user", "content": str(answer.get("label") or answer.get("value"))[:200]})
    elif str(form.get("message") or "").strip():
        message = str(form["message"]).strip()[:2000]
        chat["messages"].append({"role": "user", "content": message})
        chat["question"] = None
        result = ask_model(chat, has_clips)
        chat["draft"].update(keep_mentioned(clean_settings(result.get("settings")), message))
        chat["edit"] = bool(result.get("edit")) and has_clips
        if chat["edit"] and "prompt" not in chat["draft"]:
            chat["draft"]["prompt"] = message.lstrip("- ")[:TEXT_LIMITS["prompt"]]
        options = [str(o)[:100] for o in result.get("options") or [] if str(o).strip()][:4]
        if str(result.get("question") or "").strip() and len(options) >= 2:
            chat["question"] = {"field": None, "question": str(result["question"])[:300], "options": [{"label": o, "value": o} for o in options]}
        reply = str(result.get("reply") or "Okay.")[:2000]
        # only the Generate button makes video, whatever the model claims
        if re.search(r"\b(updated|generated|created|rendered|made)\b", reply.lower()):
            reply = "Got it. Nothing is made until you press Generate."
        chat["messages"].append({"role": "assistant", "content": reply})
    save_chat(session, chat)
    return view(chat)


if __name__ == "__main__":
    assert clean_settings({"preset": "max", "fps": "48", "camera": "rm -rf", "video": "C:/secret.mp4", "seconds": 99, "lens": 35, "prompt": "--video x run"}) == {"prompt": "video x run", "preset": "max", "fps": 48, "lens": 35}
    assert clean_settings("nope") == {}
    invented = {"prompt": "a dog", "preset": "draft", "fps": 24, "seconds": 5, "lens": 50, "camera": "tracking", "lighting": "urban", "shot": "wide"}
    assert keep_mentioned(invented, "a dog for 15 seconds") == {"prompt": "a dog"}
    assert keep_mentioned(invented, "a 5 second dog, wide shot, 24 fps, draft") == {"prompt": "a dog", "preset": "draft", "fps": 24, "seconds": 5, "shot": "wide"}
    assert keep_mentioned({"shot": "full-body", "camera": "dolly-in"}, "full body with a dolly in") == {"shot": "full-body", "camera": "dolly-in"}
    assert required_questions({}) == [] and required_questions({"prompt": "x"})[0]["field"] == "preset"
    assert required_questions({"prompt": "x", "preset": "draft", "fps": 16})[0]["field"] == "fps"
    assert required_questions({"prompt": "x", "preset": "balanced", "fps": 48}) == []
    print("ok")
