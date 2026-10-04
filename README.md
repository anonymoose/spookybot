# Spookybot — conversational Halloween talking head

A Raspberry Pi 5 + AI Camera animatronic that sees guests walk up, greets them
by their costume, holds a short conversation, and moves its jaw in sync with
what it's saying.

```
IMX500 AI Camera ──► person detection (on-sensor)
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
sudo apt install -y python3-picamera2 imx500-all imx500-tools \
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
