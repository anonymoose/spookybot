#!/usr/bin/env bash
# Runs ON the Pi from the synced project dir (via `make rpi`). Idempotent:
# system packages, I2C, venv, local fallbacks, then (re)starts the service.
set -euo pipefail
cd "$(dirname "$0")/../.."
APP_DIR=$(pwd)

PIPER_VERSION=2023.11.14-2
PIPER_VOICE=en_US-lessac-medium
PIPER_DIR=/opt/piper

echo "== apt packages"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  python3-picamera2 hailo-all python3-venv libportaudio2 i2c-tools

echo "== I2C (servo driver)"
if [ "$(sudo raspi-config nonint get_i2c)" != 0 ]; then
  sudo raspi-config nonint do_i2c 0
fi
[ -e /dev/i2c-1 ] || echo "!! /dev/i2c-1 missing: reboot the Pi to finish enabling I2C (make rpi-reboot)"

echo "== python venv"
# --system-site-packages: picamera2 comes from apt, built against system libcamera.
[ -d .venv ] || python3 -m venv .venv --system-site-packages
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt

echo "== piper (local TTS fallback)"
if [ ! -x "$PIPER_DIR/piper" ]; then
  tmp=$(mktemp -d)
  curl -fsSL -o "$tmp/piper.tgz" \
    "https://github.com/rhasspy/piper/releases/download/$PIPER_VERSION/piper_linux_aarch64.tar.gz"
  sudo mkdir -p "$PIPER_DIR"
  sudo tar -xzf "$tmp/piper.tgz" -C "$PIPER_DIR" --strip-components=1
  rm -rf "$tmp"
fi
sudo ln -sf "$PIPER_DIR/piper" /usr/local/bin/piper
for ext in onnx onnx.json; do
  f="$PIPER_DIR/$PIPER_VOICE.$ext"
  [ -s "$f" ] || sudo curl -fsSL -o "$f" \
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/$PIPER_VOICE.$ext"
done

echo "== faster-whisper (local STT fallback)"
# Fetch the model now; party wifi is exactly when the fallback is needed.
.venv/bin/python -c "
import os
from faster_whisper import WhisperModel
WhisperModel(os.environ.get('SB_STT_LOCAL_MODEL', 'small.en'), device='cpu', compute_type='int8')
print('ok')"

echo "== systemd service"
[ -f .env ] || echo "!! $APP_DIR/.env missing: service will fail until it exists"
chmod 600 .env 2>/dev/null || true
sudo install -m 644 spookybot.service /etc/systemd/system/spookybot.service
sudo systemctl daemon-reload
sudo systemctl enable spookybot
sudo systemctl restart spookybot
sleep 5
systemctl --no-pager status spookybot | head -n 5 || true
