#!/usr/bin/env bash
# Runs ON mini (via `make mini`). Idempotent: brings the speaches container up
# to date and makes sure every model spookybot asks for is downloaded.
set -euo pipefail
cd "$(dirname "$0")"

URL=http://localhost:8000
# Keep in sync with SB_STT_MODEL / SB_OPENAI_TTS_MODEL in .env.
MODELS=(
  Systran/faster-distil-whisper-small.en
  speaches-ai/Kokoro-82M-v1.0-ONNX
)

command -v docker >/dev/null || { echo "docker is not installed on $(hostname)" >&2; exit 1; }

docker compose pull
docker compose up -d --remove-orphans

echo "Waiting for speaches..."
for _ in $(seq 120); do
  curl -fs "$URL/health" >/dev/null && break
  sleep 1
done
curl -fs "$URL/health" >/dev/null || { docker compose logs --tail 50; exit 1; }

for m in "${MODELS[@]}"; do
  echo "Model: $m"
  curl -fsS -X POST "$URL/v1/models/$m"; echo
  # Load into memory now so the first guest doesn't pay the load time.
  curl -s -o /dev/null -X POST "$URL/api/ps/$m" || true
done

# Smoke test: synthesize a line, then transcribe it back.
tmp=$(mktemp --suffix=.wav)
trap 'rm -f "$tmp"' EXIT
curl -fsS "$URL/v1/audio/speech" -H 'Content-Type: application/json' -o "$tmp" \
  -d '{"model":"speaches-ai/Kokoro-82M-v1.0-ONNX","voice":"am_onyx","input":"Happy Halloween.","response_format":"wav","sample_rate":16000}'
echo -n "Round trip: "
curl -fsS "$URL/v1/audio/transcriptions" -F "file=@$tmp" \
  -F model=Systran/faster-distil-whisper-small.en -F language=en
echo
