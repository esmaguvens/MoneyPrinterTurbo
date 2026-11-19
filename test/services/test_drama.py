import json
import os
from types import SimpleNamespace

import pytest

from app.services import drama, fal


SCREENPLAY = {
    "title": "The Last Train",
    "logline": "Two strangers share a secret on the final night train.",
    "aspect": "9:16",
    "visual_style": "Photorealistic film",
    "characters": [
        {"id": "c1", "name": "Mara", "appearance": "woman, 30, red coat", "voice": "low, calm"},
        {"id": "c2", "name": "Theo", "appearance": "man, 40, grey suit", "voice": "raspy"},
    ],
    "shots": [
        {
            "id": "s01",
            "setting": "empty night train",
            "characters": ["c1"],
            "action": "Mara looks up, startled.",
            "camera": "close-up",
            "dialogue": [{"character": "Theo", "line": "You dropped this."}],
            "sound": "rails clacking",
        },
        {"id": "s02", "setting": "station platform", "action": "The train leaves."},
    ],
}


def test_screenplay_normalizes_names_and_adds_speakers():
    play = drama.screenplay_from_dict(SCREENPLAY)
    first = play.shots[0]
    # Theo speaks in shot 1, so he must be on screen even if not listed.
    assert first.characters == ["c1", "c2"]
    assert first.dialogue[0].character == "c2"
    assert play.shots[1].characters == []


def test_screenplay_rejects_unknown_character_and_crowded_shots():
    bad = json.loads(json.dumps(SCREENPLAY))
    bad["shots"][0]["dialogue"][0]["character"] = "Nobody"
    with pytest.raises(drama.DramaError, match="unknown character"):
        drama.screenplay_from_dict(bad)

    crowded = json.loads(json.dumps(SCREENPLAY))
    crowded["characters"].append({"id": "c3", "name": "Ivy", "appearance": "girl, 10"})
    crowded["shots"][0]["characters"] = ["c1", "c2", "c3"]
    with pytest.raises(drama.DramaError, match="keep it to 2"):
        drama.screenplay_from_dict(crowded)


def test_extract_json_handles_thinking_and_fences():
    reply = "<think>plan...</think>\nHere you go:\n```json\n" + json.dumps(SCREENPLAY) + "\n```"
    assert drama.extract_json_object(reply)["title"] == "The Last Train"


def test_generate_screenplay_retries_bad_json():
    replies = iter(["not json at all", json.dumps(SCREENPLAY)])
    play = drama.generate_screenplay("a premise", shots=2, generate=lambda _p: next(replies))
    assert play.title == "The Last Train"


def test_shot_prompt_contains_appearance_dialogue_and_voice():
    play = drama.screenplay_from_dict(SCREENPLAY)
    prompt = drama.shot_prompt(play, play.shots[0])
    assert "red coat" in prompt and "grey suit" in prompt
    assert 'Theo says in a raspy voice: "You dropped this."' in prompt
    assert "no subtitles" in prompt.lower()
    assert "No one speaks." in drama.shot_prompt(play, play.shots[1])


def test_estimate_skips_existing_files(tmp_path):
    play = drama.screenplay_from_dict(SCREENPLAY)
    settings = dict(drama.DEFAULT_SETTINGS)
    full = drama.estimate_cost(str(tmp_path), play, settings)
    assert full["portraits_to_make"] == 2 and full["shots_to_make"] == 2
    assert full["total_usd"] == pytest.approx(2 * 0.15 + 2 * 8 * 0.15)

    (tmp_path / "shots").mkdir()
    (tmp_path / "shots" / "s01.mp4").write_bytes(b"x")
    assert drama.estimate_cost(str(tmp_path), play, settings)["shots_to_make"] == 1


class FakeClient:
    def __init__(self):
        self.calls = []

    def run(self, endpoint, payload, label=""):
        self.calls.append((endpoint, payload))
        if "nano-banana" in endpoint:
            return {"images": [{"url": "https://cdn.example/p.jpg"}]}
        return {"video": {"url": "https://cdn.example/v.mp4"}}

    def download(self, url, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(b"\xff\xd8fake" if path.endswith(".jpg") else b"video")
        return path


def test_render_uses_reference_images_and_resumes(tmp_path):
    drama.save_screenplay(str(tmp_path), drama.screenplay_from_dict(SCREENPLAY))
    client = FakeClient()
    drama.render(str(tmp_path), client=client, settings={}, assemble_film=False)

    endpoints = [endpoint for endpoint, _ in client.calls]
    assert endpoints == [
        "fal-ai/nano-banana-pro",
        "fal-ai/nano-banana-pro",
        "fal-ai/veo3.1/fast/reference-to-video",
        "fal-ai/veo3.1/fast",  # shot without characters -> text-to-video
    ]
    shot1 = client.calls[2][1]
    assert len(shot1["image_urls"]) == 2
    assert shot1["image_urls"][0].startswith("data:image/jpeg;base64,")
    assert shot1["generate_audio"] is True and shot1["aspect_ratio"] == "9:16"
    assert shot1["duration"] == "8s"

    # A second run must not pay again for anything already on disk.
    again = FakeClient()
    drama.render(str(tmp_path), client=again, settings={}, assemble_film=False)
    assert again.calls == []


def test_render_refuses_above_max_cost(tmp_path):
    drama.save_screenplay(str(tmp_path), drama.screenplay_from_dict(SCREENPLAY))
    with pytest.raises(drama.DramaError, match="max-cost"):
        drama.render(str(tmp_path), client=FakeClient(), settings={}, max_cost=1.0)


def test_cli_render_without_yes_spends_nothing(tmp_path, capsys):
    drama.save_screenplay(str(tmp_path), drama.screenplay_from_dict(SCREENPLAY))
    assert drama.main(["render", "--project", str(tmp_path)]) == 2
    assert "Re-run with --yes" in capsys.readouterr().out
    assert not (tmp_path / "characters").exists()


# --- fal client ---------------------------------------------------------
def _resp(payload, status=200):
    return SimpleNamespace(status_code=status, json=lambda: payload, text=json.dumps(payload))


class FakeSession:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.posts = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append((url, json, headers))
        return _resp({"request_id": "r1", "status_url": "https://q/s", "response_url": "https://q/r"})

    def get(self, url, headers=None, timeout=None, **_):
        if url.endswith("/s"):
            return _resp(self.statuses.pop(0))
        return _resp({"video": {"url": "https://cdn/v.mp4"}})


def test_fal_client_submits_polls_and_returns_result():
    session = FakeSession([{"status": "IN_QUEUE", "queue_position": 2}, {"status": "IN_PROGRESS"}, {"status": "COMPLETED"}])
    client = fal.FalClient("secret-key", session=session, sleep=lambda _s: None)
    result = client.run("fal-ai/veo3.1/fast", {"prompt": "hi"})
    assert result == {"video": {"url": "https://cdn/v.mp4"}}
    url, body, headers = session.posts[0]
    assert url == "https://queue.fal.run/fal-ai/veo3.1/fast"
    assert headers["Authorization"] == "Key secret-key"
    assert headers["X-Fal-No-Retry"] == "1"


def test_fal_client_reports_model_error_without_leaking_key():
    session = FakeSession([{"status": "COMPLETED", "error": "bad secret-key prompt", "error_type": "validation"}])
    client = fal.FalClient("secret-key", session=session, sleep=lambda _s: None)
    with pytest.raises(fal.FalError) as info:
        client.run("fal-ai/x", {})
    assert "secret-key" not in str(info.value) and "validation" in str(info.value)


def test_fal_client_requires_key(monkeypatch):
    monkeypatch.delenv("FAL_KEY", raising=False)
    with pytest.raises(fal.FalError, match="API key is missing"):
        fal.FalClient("")
