"""Short drama mode: premise -> screenplay -> character portraits -> shots -> film.

Workflow (see docs/drama.md):

1. ``plan``     The configured LLM writes ``screenplay.json`` (free with Ollama).
                Edit it by hand if you like; nothing has been paid for yet.
2. ``estimate`` Prints what rendering will cost on fal.ai.
3. ``render``   Generates one photoreal reference portrait per character, then
                one video shot per screenplay shot with lip-synced dialogue
                (Veo 3.1 reference-to-video by default) and assembles the film.
                Requires ``--yes``; finished files are reused, so an interrupted
                render resumes without paying again for completed shots.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from loguru import logger

SCREENPLAY_FILE = "screenplay.json"
FILM_FILE = "drama.mp4"
MAX_CHARACTERS = 4
MAX_CHARACTERS_PER_SHOT = 2
MAX_SHOTS = 24
MAX_DIALOGUE_WORDS_PER_SHOT = 22  # roughly what fits in one 8-second shot

DEFAULT_SETTINGS: dict[str, Any] = {
    "fal_drama_image_model": "fal-ai/nano-banana-pro",
    "fal_drama_reference_video_model": "fal-ai/veo3.1/fast/reference-to-video",
    "fal_drama_text_video_model": "fal-ai/veo3.1/fast",
    "fal_drama_resolution": "1080p",
    "fal_drama_shot_seconds": 8,
    # Published fal.ai list prices (Nov 2026); used only for estimates.
    "fal_drama_video_price_per_second": 0.15,
    "fal_drama_image_price": 0.15,
    "fal_drama_safety_tolerance": "4",
}

ASPECTS = {"9:16": (1080, 1920), "16:9": (1920, 1080)}

DEFAULT_STYLE = (
    "Photorealistic live-action film, shot on 35mm cinema camera, natural "
    "lighting, shallow depth of field, realistic skin texture, subtle film grain"
)


class DramaError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Screenplay model
# --------------------------------------------------------------------------
@dataclass
class Character:
    id: str
    name: str
    appearance: str
    voice: str = ""


@dataclass
class Line:
    character: str
    line: str


@dataclass
class Shot:
    id: str
    setting: str
    action: str
    characters: list[str] = field(default_factory=list)
    camera: str = ""
    dialogue: list[Line] = field(default_factory=list)
    sound: str = ""


@dataclass
class Screenplay:
    title: str
    logline: str
    aspect: str
    visual_style: str
    characters: list[Character]
    shots: list[Shot]

    def character(self, character_id: str) -> Character:
        for character in self.characters:
            if character.id == character_id:
                return character
        raise DramaError(f"unknown character id {character_id!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _text(value: Any, limit: int = 600) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _slug(value: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", str(value).lower()).strip("-")
    return slug[:40] or fallback


def screenplay_from_dict(data: Mapping[str, Any], aspect: str | None = None) -> Screenplay:
    """Validate and normalize a screenplay (LLM output or a hand-edited file)."""
    if not isinstance(data, Mapping):
        raise DramaError("screenplay must be a JSON object")
    aspect = aspect or str(data.get("aspect") or "9:16")
    if aspect not in ASPECTS:
        raise DramaError(f"aspect must be one of {', '.join(ASPECTS)}")

    characters: list[Character] = []
    seen: set[str] = set()
    for index, raw in enumerate(data.get("characters") or [], start=1):
        if not isinstance(raw, Mapping):
            continue
        character_id = _slug(raw.get("id") or raw.get("name") or "", f"c{index}")
        if character_id in seen:
            character_id = f"{character_id}-{index}"
        appearance = _text(raw.get("appearance"), 700)
        name = _text(raw.get("name"), 60) or character_id
        if not appearance:
            raise DramaError(f"character {name!r} needs an appearance description")
        seen.add(character_id)
        characters.append(
            Character(character_id, name, appearance, _text(raw.get("voice"), 200))
        )
    if len(characters) > MAX_CHARACTERS:
        raise DramaError(f"at most {MAX_CHARACTERS} characters are supported")

    # Let shots refer to characters by id or by name.
    lookup = {c.id: c.id for c in characters}
    lookup.update({c.name.lower(): c.id for c in characters})

    def resolve(ref: Any) -> str:
        key = str(ref or "").strip()
        resolved = lookup.get(key) or lookup.get(key.lower()) or lookup.get(_slug(key, ""))
        if not resolved:
            raise DramaError(f"shot refers to unknown character {key!r}")
        return resolved

    shots: list[Shot] = []
    for index, raw in enumerate(data.get("shots") or [], start=1):
        if not isinstance(raw, Mapping):
            continue
        shot_id = _slug(raw.get("id") or "", f"s{index:02d}")
        dialogue = []
        for entry in raw.get("dialogue") or []:
            if isinstance(entry, Mapping) and _text(entry.get("line")):
                dialogue.append(Line(resolve(entry.get("character")), _text(entry.get("line"), 300)))
        present = [resolve(ref) for ref in raw.get("characters") or []]
        for entry in dialogue:  # speakers must be on screen
            if entry.character not in present:
                present.append(entry.character)
        present = list(dict.fromkeys(present))
        if len(present) > MAX_CHARACTERS_PER_SHOT:
            raise DramaError(
                f"shot {shot_id} has {len(present)} characters; keep it to "
                f"{MAX_CHARACTERS_PER_SHOT} so faces stay consistent"
            )
        words = sum(len(entry.line.split()) for entry in dialogue)
        if words > MAX_DIALOGUE_WORDS_PER_SHOT:
            logger.warning(
                f"shot {shot_id} has {words} words of dialogue; more than "
                f"~{MAX_DIALOGUE_WORDS_PER_SHOT} may be cut off in an 8-second shot"
            )
        action = _text(raw.get("action"), 800)
        if not action:
            raise DramaError(f"shot {shot_id} needs an action description")
        shots.append(
            Shot(
                id=shot_id if shot_id not in {s.id for s in shots} else f"s{index:02d}",
                setting=_text(raw.get("setting"), 400),
                action=action,
                characters=present,
                camera=_text(raw.get("camera"), 200),
                dialogue=dialogue,
                sound=_text(raw.get("sound"), 200),
            )
        )
    if not shots:
        raise DramaError("screenplay has no shots")
    if len(shots) > MAX_SHOTS:
        raise DramaError(f"at most {MAX_SHOTS} shots are supported")

    return Screenplay(
        title=_text(data.get("title"), 120) or "Untitled drama",
        logline=_text(data.get("logline"), 400),
        aspect=aspect,
        visual_style=_text(data.get("visual_style"), 400) or DEFAULT_STYLE,
        characters=characters,
        shots=shots,
    )


def load_screenplay(project_dir: str) -> Screenplay:
    path = os.path.join(project_dir, SCREENPLAY_FILE)
    if not os.path.isfile(path):
        raise DramaError(f"{path} not found; run the plan step first")
    with open(path, encoding="utf-8") as handle:
        return screenplay_from_dict(json.load(handle))


def save_screenplay(project_dir: str, screenplay: Screenplay) -> str:
    os.makedirs(project_dir, exist_ok=True)
    path = os.path.join(project_dir, SCREENPLAY_FILE)
    temp = f"{path}.tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(screenplay.to_dict(), handle, ensure_ascii=False, indent=2)
    os.replace(temp, path)
    return path


# --------------------------------------------------------------------------
# Screenplay generation (LLM)
# --------------------------------------------------------------------------
def build_screenplay_prompt(premise: str, shots: int, aspect: str, style: str) -> str:
    return f"""You are a screenwriter for short vertical/horizontal social-media dramas.
Write a complete short drama in English based on this premise:

PREMISE: {premise}

Rules:
- Exactly {shots} shots. Each shot becomes ONE continuous 8-second live-action video clip.
- 2 to {MAX_CHARACTERS} characters total, at most {MAX_CHARACTERS_PER_SHOT} characters on screen per shot.
- Dialogue per shot: at most {MAX_DIALOGUE_WORDS_PER_SHOT} words in total, natural spoken English.
- Give the story a hook in shot 1, rising tension, a twist, and a satisfying final beat.
- "appearance" must be a detailed, FIXED visual description reused in every shot:
  age, ethnicity, face, hair, build, and one specific outfit. No celebrities or real people.
- "voice" describes how the character sounds (pitch, accent, tone).
- "action" describes what happens in the 8 seconds, including facial expressions.
- "camera" gives shot size and movement (e.g. "medium close-up, slow push-in").
- "sound" lists ambient sound effects only (no music).
- Keep content suitable for general audiences.

Return ONLY valid JSON, no commentary, in exactly this shape:
{{
  "title": "...",
  "logline": "one sentence",
  "aspect": "{aspect}",
  "visual_style": "{style}",
  "characters": [
    {{"id": "c1", "name": "...", "appearance": "...", "voice": "..."}}
  ],
  "shots": [
    {{"id": "s01", "setting": "...", "characters": ["c1"], "action": "...",
      "camera": "...", "dialogue": [{{"character": "c1", "line": "..."}}],
      "sound": "..."}}
  ]
}}"""


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of an LLM reply (handles code fences)."""
    cleaned = re.sub(r"<think\b[^>]*>.*?</think>", "", text or "", flags=re.S | re.I)
    cleaned = re.sub(r"```(?:json)?", "", cleaned)
    start = cleaned.find("{")
    if start < 0:
        raise DramaError("the model did not return JSON")
    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(cleaned[start:])
    except json.JSONDecodeError as exc:
        raise DramaError(f"the model returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise DramaError("the model's JSON is not an object")
    return value


def generate_screenplay(
    premise: str,
    *,
    shots: int = 6,
    aspect: str = "9:16",
    style: str = DEFAULT_STYLE,
    attempts: int = 3,
    generate: Callable[[str], str] | None = None,
) -> Screenplay:
    premise = _text(premise, 2000)
    if not premise:
        raise DramaError("premise must not be empty")
    if not 1 <= shots <= MAX_SHOTS:
        raise DramaError(f"shots must be between 1 and {MAX_SHOTS}")
    if generate is None:
        from app.services import llm

        generate = llm._generate_response  # uses the provider chosen in the WebUI
    prompt = build_screenplay_prompt(premise, shots, aspect, style)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        reply = generate(prompt)
        if isinstance(reply, str) and reply.startswith("Error: "):
            raise DramaError(f"LLM request failed: {reply}")
        try:
            screenplay = screenplay_from_dict(extract_json_object(reply), aspect)
        except DramaError as exc:
            last_error = exc
            logger.warning(f"screenplay attempt {attempt}/{attempts} unusable: {exc}")
            continue
        if len(screenplay.shots) != shots:
            logger.warning(f"asked for {shots} shots, model wrote {len(screenplay.shots)}")
        return screenplay
    raise DramaError(f"could not get a usable screenplay: {last_error}")


# --------------------------------------------------------------------------
# Prompts for fal.ai
# --------------------------------------------------------------------------
def portrait_prompt(screenplay: Screenplay, character: Character) -> str:
    return (
        f"Photorealistic character reference photograph for a live-action film. "
        f"{character.appearance}. Upper body, facing the camera, neutral relaxed "
        f"expression, soft even studio lighting, plain light-grey background, "
        f"sharp focus, 85mm lens, true-to-life skin texture. Single person only. "
        f"No text, no watermark."
    )


def shot_prompt(screenplay: Screenplay, shot: Shot) -> str:
    parts = [screenplay.visual_style.rstrip(".") + "."]
    if shot.setting:
        parts.append(f"Setting: {shot.setting.rstrip('.')}.")
    for character_id in shot.characters:
        character = screenplay.character(character_id)
        parts.append(f"{character.name}: {character.appearance.rstrip('.')}.")
    parts.append(shot.action.rstrip(".") + ".")
    if shot.camera:
        parts.append(f"Camera: {shot.camera.rstrip('.')}.")
    for entry in shot.dialogue:
        character = screenplay.character(entry.character)
        voice = f" in a {character.voice.rstrip('.')} voice" if character.voice else ""
        line = entry.line.replace('"', "'")
        parts.append(f'{character.name} says{voice}: "{line}"')
    if not shot.dialogue:
        parts.append("No one speaks.")
    if shot.sound:
        parts.append(f"Ambient sound: {shot.sound.rstrip('.')}.")
    parts.append("No background music, no subtitles, no on-screen text.")
    return " ".join(parts)


# --------------------------------------------------------------------------
# Cost estimate and rendering
# --------------------------------------------------------------------------
def drama_settings(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    settings = dict(DEFAULT_SETTINGS)
    try:
        from app.config import config

        for key in DEFAULT_SETTINGS:
            value = config.app.get(key)
            if value not in (None, ""):
                settings[key] = value
    except Exception as exc:  # config is optional for estimates/tests
        logger.debug(f"drama settings: using defaults ({exc})")
    if overrides:
        settings.update({k: v for k, v in overrides.items() if v not in (None, "")})
    return settings


def _portrait_path(project_dir: str, character: Character) -> str:
    return os.path.join(project_dir, "characters", f"{character.id}.jpg")


def _shot_path(project_dir: str, shot: Shot) -> str:
    return os.path.join(project_dir, "shots", f"{shot.id}.mp4")


def estimate_cost(
    project_dir: str, screenplay: Screenplay, settings: Mapping[str, Any]
) -> dict[str, Any]:
    seconds = float(settings["fal_drama_shot_seconds"])
    shot_price = seconds * float(settings["fal_drama_video_price_per_second"])
    image_price = float(settings["fal_drama_image_price"])
    portraits_todo = [
        c for c in screenplay.characters if not os.path.isfile(_portrait_path(project_dir, c))
    ]
    shots_todo = [s for s in screenplay.shots if not os.path.isfile(_shot_path(project_dir, s))]
    total = len(portraits_todo) * image_price + len(shots_todo) * shot_price
    return {
        "portraits_to_make": len(portraits_todo),
        "shots_to_make": len(shots_todo),
        "film_seconds": round(len(screenplay.shots) * seconds),
        "price_per_portrait": image_price,
        "price_per_shot": round(shot_price, 2),
        "total_usd": round(total, 2),
    }


def _veo_duration(seconds: Any) -> str:
    return f"{int(float(seconds))}s"


def render(
    project_dir: str,
    *,
    client=None,
    settings: Mapping[str, Any] | None = None,
    max_cost: float | None = None,
    assemble_film: bool = True,
    bgm: str | None = None,
    bgm_volume: float = 0.12,
) -> str | None:
    """Generate missing portraits and shots, then assemble the film."""
    from app.services.fal import FalClient, file_to_data_uri

    screenplay = load_screenplay(project_dir)
    settings = drama_settings(settings)
    estimate = estimate_cost(project_dir, screenplay, settings)
    if max_cost is not None and estimate["total_usd"] > max_cost:
        raise DramaError(
            f"estimated cost ${estimate['total_usd']:.2f} is above --max-cost ${max_cost:.2f}"
        )
    client = client or FalClient()
    resolution = str(settings["fal_drama_resolution"])
    safety = str(settings["fal_drama_safety_tolerance"])
    image_aspect = "3:4" if screenplay.aspect == "9:16" else "4:3"

    for character in screenplay.characters:
        path = _portrait_path(project_dir, character)
        if os.path.isfile(path):
            logger.info(f"portrait for {character.name}: reusing {path}")
            continue
        logger.info(f"portrait for {character.name}: generating")
        output = client.run(
            str(settings["fal_drama_image_model"]),
            {
                "prompt": portrait_prompt(screenplay, character),
                "aspect_ratio": image_aspect,
                "num_images": 1,
                "output_format": "jpeg",  # small enough to send back as a data URI
                "safety_tolerance": safety,
            },
            label=f"portrait {character.name}",
        )
        images = output.get("images") or []
        if not images or not images[0].get("url"):
            raise DramaError(f"no image returned for {character.name}: {output}")
        client.download(images[0]["url"], path)

    for index, shot in enumerate(screenplay.shots, start=1):
        path = _shot_path(project_dir, shot)
        if os.path.isfile(path):
            logger.info(f"shot {index}/{len(screenplay.shots)} ({shot.id}): reusing {path}")
            continue
        prompt = shot_prompt(screenplay, shot)
        payload: dict[str, Any] = {
            "prompt": prompt,
            "aspect_ratio": screenplay.aspect,
            "duration": _veo_duration(settings["fal_drama_shot_seconds"]),
            "resolution": resolution,
            "generate_audio": True,
            "safety_tolerance": safety,
        }
        if shot.characters:
            endpoint = str(settings["fal_drama_reference_video_model"])
            payload["image_urls"] = [
                file_to_data_uri(_portrait_path(project_dir, screenplay.character(cid)))
                for cid in shot.characters
            ]
        else:
            endpoint = str(settings["fal_drama_text_video_model"])
        logger.info(f"shot {index}/{len(screenplay.shots)} ({shot.id}): generating")
        output = client.run(endpoint, payload, label=f"shot {shot.id}")
        video = output.get("video") or {}
        if not video.get("url"):
            raise DramaError(f"no video returned for shot {shot.id}: {output}")
        client.download(video["url"], path)
        with open(os.path.join(project_dir, "shots", f"{shot.id}.prompt.txt"), "w", encoding="utf-8") as handle:
            handle.write(prompt)

    if not assemble_film:
        return None
    return assemble(project_dir, bgm=bgm, bgm_volume=bgm_volume)


# --------------------------------------------------------------------------
# Assembly (FFmpeg)
# --------------------------------------------------------------------------
def ffmpeg_exe() -> str:
    configured = os.environ.get("IMAGEIO_FFMPEG_EXE")
    if configured and os.path.isfile(configured):
        return configured
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        found = shutil.which("ffmpeg")
        if found:
            return found
    raise DramaError("ffmpeg not found; install it or set app.ffmpeg_path")


def _has_audio(ffmpeg: str, path: str) -> bool:
    probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", path], capture_output=True, text=True, errors="replace"
    )
    return "Audio:" in probe.stderr


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        raise DramaError(f"ffmpeg failed: {result.stderr[-1500:]}")


def assemble(project_dir: str, *, bgm: str | None = None, bgm_volume: float = 0.12) -> str:
    screenplay = load_screenplay(project_dir)
    width, height = ASPECTS[screenplay.aspect]
    clips = [_shot_path(project_dir, shot) for shot in screenplay.shots]
    missing = [clip for clip in clips if not os.path.isfile(clip)]
    if missing:
        raise DramaError(f"missing shots, run render first: {', '.join(missing)}")
    ffmpeg = ffmpeg_exe()
    video_filter = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps=30,format=yuv420p"
    )
    output = os.path.join(project_dir, FILM_FILE)
    with tempfile.TemporaryDirectory(dir=project_dir, prefix=".assemble-") as work:
        normalized = []
        for index, clip in enumerate(clips):
            target = os.path.join(work, f"{index:03d}.mp4")
            cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", clip]
            if _has_audio(ffmpeg, clip):
                cmd += ["-map", "0:v:0", "-map", "0:a:0"]
            else:
                cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
                        "-map", "0:v:0", "-map", "1:a:0", "-shortest"]
            cmd += ["-vf", video_filter, "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", target]
            _run(cmd)
            normalized.append(target)
        listing = os.path.join(work, "list.txt")
        with open(listing, "w", encoding="utf-8") as handle:
            for clip in normalized:
                handle.write(f"file '{clip.replace(os.sep, '/')}'\n")
        joined = os.path.join(work, "joined.mp4")
        _run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
              "-i", listing, "-c", "copy", "-movflags", "+faststart", joined])
        if bgm:
            if not os.path.isfile(bgm):
                raise DramaError(f"background music file not found: {bgm}")
            mixed = os.path.join(work, "mixed.mp4")
            _run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", joined,
                  "-stream_loop", "-1", "-i", bgm, "-filter_complex",
                  f"[1:a]volume={float(bgm_volume)}[m];[0:a][m]amix=inputs=2:duration=first:normalize=0[a]",
                  "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                  "-movflags", "+faststart", mixed])
            joined = mixed
        os.replace(joined, output)
    logger.success(f"drama assembled: {output}")
    return output


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _print_summary(screenplay: Screenplay) -> None:
    print(f"\n{screenplay.title}  ({screenplay.aspect}, {len(screenplay.shots)} shots)")
    if screenplay.logline:
        print(f"  {screenplay.logline}")
    print("\nCharacters:")
    for c in screenplay.characters:
        print(f"  [{c.id}] {c.name}: {c.appearance[:110]}")
    print("\nShots:")
    for s in screenplay.shots:
        who = ", ".join(screenplay.character(cid).name for cid in s.characters) or "no one"
        print(f"  {s.id} ({who}): {s.action[:100]}")
        for line in s.dialogue:
            print(f"      {screenplay.character(line.character).name}: \"{line.line}\"")


def _print_estimate(estimate: Mapping[str, Any]) -> None:
    print(
        f"\nfal.ai cost estimate: {estimate['portraits_to_make']} portrait(s) x "
        f"${estimate['price_per_portrait']:.2f} + {estimate['shots_to_make']} shot(s) x "
        f"${estimate['price_per_shot']:.2f} = ${estimate['total_usd']:.2f}  "
        f"(film length ~{estimate['film_seconds']}s; list prices, check fal.ai/pricing)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.services.drama",
        description="Make a photoreal short drama with lip-synced dialogue on fal.ai.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="write screenplay.json with the configured LLM (no fal.ai cost)")
    plan.add_argument("--project", required=True, help="project folder, e.g. storage/dramas/my-drama")
    plan.add_argument("--premise", required=True, help="story idea in one or two sentences")
    plan.add_argument("--shots", type=int, default=6, help="number of 8-second shots (default 6)")
    plan.add_argument("--aspect", choices=list(ASPECTS), default="9:16")
    plan.add_argument("--style", default=DEFAULT_STYLE, help="visual style applied to every shot")
    plan.add_argument("--force", action="store_true", help="overwrite an existing screenplay.json")

    est = sub.add_parser("estimate", help="show the fal.ai cost of rendering what is missing")
    est.add_argument("--project", required=True)

    ren = sub.add_parser("render", help="generate portraits and shots on fal.ai (costs money)")
    ren.add_argument("--project", required=True)
    ren.add_argument("--yes", action="store_true", help="confirm spending on fal.ai")
    ren.add_argument("--max-cost", type=float, default=None, help="refuse if the estimate is higher (USD)")
    ren.add_argument("--bgm", default=None, help="optional background music file")
    ren.add_argument("--bgm-volume", type=float, default=0.12)

    asm = sub.add_parser("assemble", help="re-join existing shots into drama.mp4 (free)")
    asm.add_argument("--project", required=True)
    asm.add_argument("--bgm", default=None)
    asm.add_argument("--bgm-volume", type=float, default=0.12)

    args = parser.parse_args(argv)
    project = os.path.abspath(args.project)
    try:
        if args.command == "plan":
            if os.path.isfile(os.path.join(project, SCREENPLAY_FILE)) and not args.force:
                raise DramaError(f"{project}/{SCREENPLAY_FILE} exists; pass --force to replace it")
            screenplay = generate_screenplay(
                args.premise, shots=args.shots, aspect=args.aspect, style=args.style
            )
            path = save_screenplay(project, screenplay)
            _print_summary(screenplay)
            _print_estimate(estimate_cost(project, screenplay, drama_settings()))
            print(f"\nSaved {path}. Edit it if you like, then run:\n"
                  f"  python -m app.services.drama render --project \"{args.project}\" --yes")
        elif args.command == "estimate":
            screenplay = load_screenplay(project)
            _print_summary(screenplay)
            _print_estimate(estimate_cost(project, screenplay, drama_settings()))
        elif args.command == "render":
            screenplay = load_screenplay(project)
            estimate = estimate_cost(project, screenplay, drama_settings())
            _print_estimate(estimate)
            if estimate["total_usd"] > 0 and not args.yes:
                print("\nNothing was generated. Re-run with --yes to spend this on fal.ai.")
                return 2
            film = render(project, max_cost=args.max_cost, bgm=args.bgm, bgm_volume=args.bgm_volume)
            print(f"\nDone: {film}")
        elif args.command == "assemble":
            print(f"Done: {assemble(project, bgm=args.bgm, bgm_volume=args.bgm_volume)}")
    except DramaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
