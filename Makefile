# Deploy spookybot.  `make` / `make all` updates both machines.
#   make mini   speech server (Whisper + Kokoro in Docker) on mini
#   make rpi    the bot itself, as a systemd service on the Pi
MINI ?= mini
RPI  ?= rpi
RPI_DIR  ?= spookybot
MINI_DIR ?= spookybot-speech

.PHONY: all mini rpi verify status logs logs-mini restart stop rpi-reboot

all: mini rpi

mini:
	rsync -a --delete deploy/mini/ $(MINI):$(MINI_DIR)/
	ssh $(MINI) 'bash $(MINI_DIR)/setup.sh'

# .env is synced on purpose (it holds the API key); .venv lives only on the Pi.
rpi: verify
	rsync -a --delete --exclude .git --exclude .venv --exclude __pycache__ \
		./ $(RPI):$(RPI_DIR)/
	ssh $(RPI) 'bash $(RPI_DIR)/deploy/rpi/setup.sh'

verify:
	python3 verify.py

status:
	ssh $(RPI) 'systemctl --no-pager status spookybot | head -n 12'
	ssh $(MINI) 'docker ps --filter name=spookybot-speaches --format "{{.Names}}: {{.Status}}"'

# make logs            follow everything
# make logs tag=VOICE  follow one stream: VOICE, VISION, CHAT or MOTION
# make logs tag="MOTION JAW"   one servo: MOTION JAW, MOTION PAN, MOTION TILT
logs:
ifdef tag
	ssh -t $(RPI) "journalctl -u spookybot -f -o cat --grep ': $(shell echo '$(tag)' | tr a-z A-Z) '"
else
	ssh -t $(RPI) 'journalctl -u spookybot -f'
endif

logs-mini:
	ssh -t $(MINI) 'cd $(MINI_DIR) && docker compose logs -f'

restart:
	ssh $(RPI) 'sudo systemctl restart spookybot'

stop:
	ssh $(RPI) 'sudo systemctl stop spookybot'

rpi-reboot:
	ssh $(RPI) 'sudo reboot' || true
