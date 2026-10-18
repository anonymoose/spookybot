# Spookybot — conversational Halloween talking head

A Raspberry Pi 5 + camera + Hailo AI HAT animatronic that sees guests walk up, greets them
by their costume, holds a short conversation, and moves its jaw in sync with
what it's saying.

```
IMX219 camera ──► person detection (YOLOv8 on Hailo-8L)
                     │
                     ├─► arrival/departure ──► state machine
                     ├─► pan/tilt tracking ──► servos
                     └─► costume still ──────► Claude (vision)

ReSpeaker mic ──► VAD + noise gate ──► Whisper ──► Claude ──► TTS
                                                              │
                                    speaker ◄── PCM stream ───┤
                                    jaw servo ◄── RMS envelope┘
```

## The one thing to read before buying anything

**The Pi 5 has no 3.5mm audio jack.** It was removed after the Pi 4. Your audio
output options are HDMI, USB, or an I2S HAT. Everything in the hardware list
below follows from that.

## Install

```bash
sudo apt update
sudo apt install -y python3-picamera2 hailo-all \
                    python3-venv libportaudio2 i2c-tools
sudo raspi-config nointeractive do_i2c 0     # enable I2C for the servo driver
sudo reboot

git clone <your repo> ~/spookybot && cd ~/spookybot
python3 -m venv .venv --system-site-packages   # picamera2 is an apt package
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env && nano .env              # add API keys
```

`--system-site-packages` is required: `picamera2` installs via apt against the
system libcamera, and a sealed venv can't see it.

## Bring-up order

Do these in order. Each one isolates a different failure.

```bash
python -m spookybot.tools.list_devices          # 1. find your audio devices
python -m spookybot.tools.test_pipeline --speaker
python -m spookybot.tools.calibrate_noise --speak-test   # 2. set the noise gate
python -m spookybot.tools.test_pipeline --mic
python -m spookybot.tools.test_pipeline --brain
python -m spookybot.tools.test_pipeline --vision
python -m spookybot.tools.servo_jog --servo jaw # 3. BEFORE bolting into skull
python -m spookybot.tools.test_pipeline --jaw
python -m spookybot.main                        # 4. the whole thing
```

Step 2 is the one that decides whether this works at an actual party. Run
`calibrate_noise` **during party conditions** — music on, people talking. A
threshold tuned in a quiet garage makes the head respond to the entire room.

## Running it for real

Deployment is driven from a dev machine with the `Makefile` (needs SSH keys to
both hosts; override with `make RPI=host MINI=host ...`):

```bash
make mini      # speech server: Whisper STT + Kokoro TTS (speaches, Docker) on mini
make rpi       # sync code + .env to the Pi, install deps, (re)start the service
make           # both
make status    # service + container state
make logs      # follow the bot's journal   (make logs-mini for the speech server)
make logs tag=VOICE   # one stream: VOICE (speech in/out), VISION, CHAT (Claude, tokens, cost), MOTION
make logs tag="MOTION JAW"   # one servo: MOTION JAW, MOTION PAN, MOTION TILT
```

`deploy/mini/setup.sh` and `deploy/rpi/setup.sh` hold every install step and
are safe to re-run. The service runs as `kbedwell` from `~/spookybot`.

If mini is unreachable the bot falls back to faster-whisper and Piper on the
Pi itself: slower, but it keeps talking.

## Tuning

| Symptom | Setting |
|---|---|
| Responds to the whole room | Raise `SB_RMS_FLOOR` (run `calibrate_noise`) |
| Doesn't hear quiet guests | Lower `SB_RMS_FLOOR`, move mic closer |
| Cuts people off mid-sentence | Raise `SB_VAD_END_FRAMES` (35 ≈ 700ms) |
| Long pause before replying | Lower `SB_MAX_TOKENS`; check `--brain` timing |
| Jaw barely moves | Widen `ServoConfig.jaw` range in `config.py` |
| Jaw hangs open / chatters | Lower `SB_JAW_ALPHA` toward 0.4 |
| Greets people across the yard | Raise `SB_MIN_BOX_HEIGHT` toward 0.45 |
| Hears itself, talks in a loop | Keep `SB_HALF_DUPLEX=true` |
| Never greets first; waits to be spoken to | Hailo not detected, so it's in voice-activated mode. Check `lspci \| grep -i hailo` and the PCIe ribbon |

### Model choice

Measured from the Pi with `persona.md`, 18 calls each (median time to the
first full sentence, which is when the head starts talking):

| `SB_MODEL` | First sentence | Verdict |
|---|---|---|
| `claude-haiku-4-5` | 0.8 s | Fastest, but flatter jokes and still slips in spoken `*stage directions*` |
| **`claude-sonnet-5-5`** | **1.0 s** | **Default.** Funniest of the fast options, clean spoken output |
| `claude-sonnet-5` | 1.2 s | Good, but replies come as one long sentence, so nothing streams early |
| `claude-opus-5-5` | 1.9 s | Funny, too slow for a party |

Each model needs its own fastest settings: see `SB_THINKING` / `SB_EFFORT`
in `.env.example`.

## Barge-in (letting guests interrupt)

Off by default (`SB_HALF_DUPLEX=true` hard-mutes the mic while speaking). It's
the safe choice: without it, an open mic next to a loud speaker will hear the
head's own voice, transcribe it, and answer itself in a loop.

If you have a ReSpeaker with hardware AEC and want interruption, set
`SB_HALF_DUPLEX=false` and test at full party volume before committing. The
echo canceller has to handle your actual speaker at your actual level; it will
look fine at conversation volume and fail at party volume.

## Local fallback

Party wifi fails. Set these up even if you plan to run hosted:

```bash
pip install faster-whisper          # then: SB_STT=faster-whisper
# Piper for local TTS:
mkdir -p /opt/piper && cd /opt/piper
wget https://github.com/rhasspy/piper/releases/latest/download/piper_linux_aarch64.tar.gz
tar xzf piper_linux_aarch64.tar.gz
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
# then: SB_TTS=piper
```

TTS already falls back to Piper automatically when the hosted call fails, and
STT falls back to local Whisper — but only if they're installed. Test the
fallback path once with wifi off.

## Safety notes

- **Never power servos from the Pi's 5V rail.** A stalled MG996R pulls over 2A
  and will brown out the Pi mid-sentence. Separate 6V supply into the PCA9685
  V+ terminal, grounds tied together.
- **Jog servo limits before bolting into the skull.** A servo driven past a
  mechanical stop strips its gears in seconds.
- `persona.md` holds the PG-13 rules and the don't-break-character rules. Edit
  the voice freely; keep those constraints — kids will talk to this thing.

## Verification

`python3 verify.py` runs the full logic suite with hardware and APIs stubbed.
No Pi, camera, or API keys needed. Run it after any change to the state machine.
