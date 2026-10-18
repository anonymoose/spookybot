#!/usr/bin/env python3
"""Audition Kokoro voices on mini's speech server before picking one for the bot.

Runs on your dev machine and plays each sample through its speakers. Nothing on
mini or the Pi changes; when you find one you like, set SB_OPENAI_VOICE (and
SB_TTS_SPEED) in .env and run `make rpi`.

    scripts/audition.py                          # the shortlist, at .env's speed
    scripts/audition.py --voice bm_george        # one voice, no loop
    scripts/audition.py bm_george bm_lewis       # just these voices
    scripts/audition.py --speed 0.85 --text "Come closer. I don't bite. Much."
    scripts/audition.py --list                   # every English voice on mini
    scripts/audition.py --all --save voices/     # write WAVs instead of playing
    scripts/audition.py --on-pi bm_george        # play on the bot's own speaker
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

MODEL = "speaches-ai/Kokoro-82M-v1.0-ONNX"
DEFAULT_URL = "http://mini:8000/v1"
SHORTLIST = ["am_onyx", "am_fenrir", "bm_george", "bm_lewis", "bm_fable", "bm_daniel"]
DEFAULT_TEXT = "Well, well. A vampire. Bold choice for someone so sun-kissed."
# Kokoro voice ids start with a language letter; a = American, b = British.
# The others (Spanish, French, Japanese, ...) mangle English.
ENGLISH = ("a", "b")
ENV_FILE = pathlib.Path(__file__).resolve().parent.parent / ".env"


def env_speed() -> float:
    """SB_TTS_SPEED from the project's .env, so samples match the bot."""
    try:
        for line in ENV_FILE.read_text().splitlines():
            if line.startswith("SB_TTS_SPEED="):
                return float(line.split("=", 1)[1].split("#")[0].strip())
    except (OSError, ValueError):
        pass
    return 1.0


def english_voices(url: str) -> list[str]:
    with urllib.request.urlopen(f"{url}/audio/voices", timeout=10) as resp:
        data = json.load(resp)
    voices = data.get("voices", data.get("data", data))
    ids = {v.get("voice_id") or v.get("id") or v.get("name") for v in voices
           if MODEL in (v.get("model_id") or v.get("model") or MODEL)}
    return sorted(v for v in ids if v and v[0] in ENGLISH)


def synthesize(url: str, voice: str, text: str, speed: float) -> bytes:
    body = json.dumps({"model": MODEL, "voice": voice, "input": text,
                       "speed": speed, "response_format": "wav"}).encode()
    req = urllib.request.Request(f"{url}/audio/speech", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def play_local(wav: bytes) -> None:
    for player in (["aplay", "-q"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-"]):
        if shutil.which(player[0]):
            subprocess.run(player, input=wav, check=False)
            return
    sys.exit("No audio player found (aplay or ffplay); use --save DIR instead.")


def play_on_pi(wav: bytes, host: str) -> None:
    # plughw converts rate and channels for the stereo-only USB adapter. The
    # bot holds the speaker while it runs, so this fails if the service is up.
    cmd = ("dev=$(aplay -l | sed -n 's/^card \\([0-9]*\\).*NS-MCX321K.*/\\1/p' | head -1); "
           "aplay -q -D plughw:${dev:-0},0")
    result = subprocess.run(["ssh", host, cmd], input=wav, check=False)
    if result.returncode != 0:
        print("   (playback on the Pi failed — is the spookybot service running? "
              "`make stop` first)")


def audition(voice: str, args, speed: float, out_dir: pathlib.Path | None) -> bool:
    """Synthesize one voice and play or save it. False if the server refused it."""
    print(f"== {voice}", flush=True)
    try:
        wav = synthesize(args.url, voice, args.text, speed)
    except urllib.error.HTTPError as exc:
        print(f"   failed: HTTP {exc.code} (unknown voice?)")
        return False
    except urllib.error.URLError as exc:
        sys.exit(f"Can't reach the speech server at {args.url}: {exc.reason}")
    if out_dir:
        (out_dir / f"{voice}.wav").write_bytes(wav)
    elif args.on_pi:
        play_on_pi(wav, args.on_pi)
    else:
        play_local(wav)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audition Kokoro voices on mini's speech server.")
    parser.add_argument("voices", nargs="*", help=f"voice ids (default: {' '.join(SHORTLIST)})")
    parser.add_argument("--voice", help="play just this one voice (skips the list/loop)")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="line to speak")
    parser.add_argument("--speed", type=float, default=None,
                        help="speaking speed (default: SB_TTS_SPEED from .env, else 1.0)")
    parser.add_argument("--all", action="store_true", help="every English voice on mini")
    parser.add_argument("--list", action="store_true", help="list English voices and exit")
    parser.add_argument("--save", metavar="DIR", help="write <voice>.wav files instead of playing")
    parser.add_argument("--on-pi", nargs="?", const="rpi", metavar="HOST",
                        help="play on the bot's speaker over ssh (default host: rpi)")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"speech server (default {DEFAULT_URL})")
    args = parser.parse_args()

    speed = args.speed if args.speed is not None else env_speed()
    out_dir = pathlib.Path(args.save) if args.save else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    if args.voice:
        print(f"Speed {speed:g}: {args.text!r}")
        ok = audition(args.voice, args, speed, out_dir)
        if ok and out_dir:
            print(f"Saved {out_dir / (args.voice + '.wav')}")
        return 0 if ok else 1

    try:
        if args.list:
            print(" ".join(english_voices(args.url)))
            return 0
        voices = english_voices(args.url) if args.all else (args.voices or SHORTLIST)
    except urllib.error.URLError as exc:
        sys.exit(f"Can't reach the speech server at {args.url}: {exc.reason}")

    print(f"Speed {speed:g}: {args.text!r}")
    saved = 0
    for voice in voices:
        if audition(voice, args, speed, out_dir) and out_dir:
            saved += 1

    if out_dir:
        print(f"Saved {saved} file(s) to {out_dir}/")
    print("Pick one: set SB_OPENAI_VOICE (and SB_TTS_SPEED) in .env, then `make rpi`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
