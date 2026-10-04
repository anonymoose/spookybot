# Halloween Talking Head — Hardware Design & Shopping List

## The constraint that drives everything

**The Raspberry Pi 5 has no 3.5mm audio jack.** It was dropped after the Pi 4.
Output is HDMI, USB, or an I2S HAT only. Most Pi audio tutorials predate this and
will send you down a dead end.

So the audio chain has to be: **USB mic → Pi → USB or I2S out → amplified speaker.**

---

## Architecture

```
                    ┌─────────────────────────────┐
                    │      Raspberry Pi 5         │
                    │                             │
  AI Camera ────────┤ CSI   person detect (sensor)│
  (IMX500)          │       ├─ arrival/departure  │
                    │       ├─ pan/tilt target    │
                    │       └─ costume still      │
                    │                             │
  ReSpeaker ────────┤ USB   VAD + noise gate      │
  mic array         │       └─► Whisper ─► Claude │
                    │                       │     │
  USB DAC ──────────┤ USB   ◄── TTS PCM ────┘     │
    │               │           │                 │
    │               │           └─► RMS envelope  │
    │               │                     │       │
  PCA9685 ──────────┤ I2C   ◄─────────────┘       │
    │               └─────────────────────────────┘
    │                         │
  jaw/pan/tilt servos    6V 5A supply (separate!)
    │
  Powered speaker ◄── 3.5mm from USB DAC
```

### Why the jaw sync works

The audio callback that hands each block of samples to the speaker computes
that block's RMS and pushes it to the jaw in the same call. The jaw is driven
by the exact buffer the DAC is about to play, so there is no clock to drift and
no sync to tune. This is the single most important design decision in the
build — timer-based approaches drift within a sentence and look wrong.

### Why latency stays low

Claude's response is streamed and cut at sentence boundaries. The first
sentence goes to TTS while Claude is still writing the second. First audio
lands at first-sentence latency (~800ms) instead of full-response latency
(~3s). At a party, that gap is the difference between conversation and someone
walking away.

---

## Microphone

### Buy: ReSpeaker USB Mic Array v2.0 — ~$80

This is the right answer for a party and it isn't close. The XMOS XVF-3000 chip
does beamforming, noise suppression, and acoustic echo cancellation **in
hardware**, before audio reaches the Pi. At a party you are fighting music,
crowd noise, and the prop's own speaker three feet away. A $10 USB mic hands
the Pi a wall of noise and no amount of software fixes it.

Specifically it gives you:
- **Beamforming** — steers a pickup lobe toward whoever is talking and
  attenuates the rest of the room. This is the feature you're actually paying for.
- **AEC** — lets the mic stay open while the speaker plays, which is what makes
  barge-in possible at all.
- **USB UAC 1.0** — appears as a standard audio device, no driver hell.
- **Direction of arrival** — bonus: you can turn the head toward a voice even
  when the camera hasn't found them.

Get the **USB Mic Array** (enclosed version) rather than the bare
**Mic Array v2.0** board if you can — same PCB, injection-molded case, and it's
going inside a prop that people will touch.

### Budget alternative: ReSpeaker 2-Mic HAT — ~$25

Half the pickup quality and no hardware AEC, which means `SB_HALF_DUPLEX=true`
is mandatory (no interruption). Workable if the head is on a quiet porch rather
than in the middle of the party. It's a HAT, so it occupies the GPIO header —
you'd stack the PCA9685 above it.

### Don't buy

- Bare electret capsules or USB "lavalier" mics — no array, no AEC, and at 4ft
  in a loud room you'll get nothing usable.
- USB conference speakerphones (Jabra etc.) — the AEC is tuned for a quiet
  office and they aggressively duck the far end, which fights your VAD.

### Placement

Mouth-height, front of the prop, **physically decoupled from the speaker.** Put
foam or rubber between the mic and any panel the speaker is bolted to.
Structure-borne vibration defeats acoustic echo cancellation entirely — the AEC
models an acoustic path, not a mechanical one.

---

## Speaker

You asked for "pretty powerful to sound clear at a party." Two things matter
and only one of them is watts: **intelligibility at distance**, which is mostly
midrange clarity and horn/driver directivity, and **headroom**, so speech
doesn't compress into mush when the room gets loud.

### Recommended: powered PA speaker — ~$100–150

**Rockville RPG-Series 8" or 10" powered PA speaker**, or equivalent from
Pyle / Mackie Thump. Look for:
- 100W+ RMS (ignore "peak power" marketing numbers entirely)
- 8" or 10" woofer with a horn tweeter
- 3.5mm or 1/4" line input
- Built-in amp (no separate amplifier to power and hide)

A PA speaker is designed for exactly this problem — speech that has to stay
intelligible across a crowded space. A bookshelf or Bluetooth speaker is voiced
for music and turns to mud at volume, which is the opposite of what you need.

The horn tweeter matters more than the wattage: it throws midrange forward
instead of spraying it, so guests standing in front hear words while the
neighbors hear less.

### Compact alternative: Dayton Audio sub-compact powered speaker — ~$60

If the prop has to be self-contained and small. Noticeably less headroom;
fine for a porch, marginal in a full party.

### Loud outdoor option: JBL PartyBox / IRS-style battery PA — ~$200+

Overkill for intelligibility but gives you battery power and no extension cord,
which is genuinely useful for a yard setup.

### Don't buy

- Small USB-powered speakers (Adafruit mini, etc.) — fine for a bench test,
  inaudible at a party.
- Bluetooth-only speakers — added latency destroys jaw sync, and the pairing
  will drop at the worst moment. **Use a wired connection.**

---

## Getting audio out of the Pi 5

### Option A (recommended): USB DAC — ~$10

**Sabrent USB External Stereo Sound Adapter (AU-MMSA)** or any generic
"USB Audio Device" dongle. Plug into a USB-A port, 3.5mm cable to the speaker's
line input. Shows up as an ALSA device immediately, no configuration.

Set `SB_OUTPUT_DEVICE` to a substring of its name (run
`python -m spookybot.tools.list_devices` to see it).

### Option B: use the ReSpeaker's own output

The ReSpeaker Mic Array v2.0 has a 3.5mm output jack and a WM8960 codec.
One fewer dongle, and it keeps playback and capture on the same clock domain,
which helps the AEC.

The catch: **output is locked to 16kHz** on the standard firmware. Speech is
perfectly intelligible at 16k, but it sounds noticeably telephone-ish. For a
spooky voice coming out of a severed head, that's arguably a feature. If you
go this route, set `SB_PLAYBACK_RATE=16000`.

### Option C: I2S HAT — ~$25

**Adafruit I2S 3W Amplifier (MAX98357A)** or a HiFiBerry DAC. Best audio
quality, but it consumes GPIO pins you may want for servos and adds a
device-tree overlay to configure. Only worth it if you're driving a raw driver
rather than a powered speaker.

---

## Servos and motion

| Part | Price | Notes |
|---|---|---|
| PCA9685 16-channel PWM driver | ~$6 | **Required.** See below. |
| MG996R metal-gear servo ×1 (jaw) | ~$8 | Metal gears; the jaw takes abuse |
| MG90S or SG90 ×2 (pan, tilt) | ~$4 ea | Lighter duty |
| 6V 5A DC supply | ~$12 | **Separate from the Pi's supply** |

**Why a PCA9685 instead of Pi GPIO PWM:** the Pi 5's software PWM jitters under
load, and a jittering servo *buzzes audibly*. Two feet from a microphone, that
buzz lands in every transcription. The PCA9685 generates PWM in hardware and
costs six dollars.

**Power warning:** do not feed servos from the Pi's 5V rail. A stalled MG996R
pulls over 2A and will brown out the Pi mid-sentence. Separate 6V supply into
PCA9685 V+, grounds tied together.

Wiring:
```
Pi GPIO2 (SDA)  ──► PCA9685 SDA
Pi GPIO3 (SCL)  ──► PCA9685 SCL
Pi 3.3V         ──► PCA9685 VCC   (logic only)
Pi GND          ──► PCA9685 GND ──┬── 6V supply GND   (common ground!)
6V 5A supply +  ──► PCA9685 V+  ──┘

PCA9685 ch0 ──► jaw servo
        ch1 ──► pan servo
        ch2 ──► tilt servo
```

**Calibrate servo limits before bolting anything into a skull.** Run
`python -m spookybot.tools.servo_jog --servo jaw`. A servo driven past a
mechanical stop strips its gears in seconds, at full torque.

---

## Complete bill of materials

| Item | Price |
|---|---|
| ReSpeaker USB Mic Array v2.0 | $80 |
| Powered PA speaker, 8–10" | $120 |
| USB audio adapter (Sabrent) | $10 |
| PCA9685 PWM driver | $6 |
| MG996R servo (jaw) | $8 |
| MG90S servos ×2 (pan/tilt) | $8 |
| 6V 5A power supply | $12 |
| 3.5mm cable, jumpers, foam | $15 |
| **Total (excluding Pi + camera you own)** | **~$259** |

Budget version — ReSpeaker 2-Mic HAT + compact powered speaker: **~$135**,
with no barge-in and less reach.

---

## Running costs

Roughly **$0.02–0.04 per conversation** (Claude + Whisper + ElevenLabs).
A busy party of 150 conversations lands around $4. ElevenLabs is the largest
share; switching to Piper for local TTS makes it nearly free at some quality
cost.

---

## Build order

1. **Pi + camera first.** Get `test_pipeline --vision` detecting a person
   before anything else is attached.
2. **Audio out.** USB DAC → speaker, `test_pipeline --speaker`.
3. **Audio in.** ReSpeaker, then `calibrate_noise` and `test_pipeline --mic`.
4. **Brain.** `test_pipeline --brain`, confirm first-sentence latency under ~1s.
5. **Servos on the bench**, unattached. `servo_jog` each one, record limits into
   `config.py`.
6. **Jaw sync.** `test_pipeline --jaw`, tune range and `SB_JAW_ALPHA` by eye.
7. **Assemble into the head.** Decouple the mic from the speaker mechanically.
8. **Re-calibrate the noise floor in the assembled prop** — the enclosure
   changes the acoustics, sometimes a lot.
9. **Dress rehearsal at party volume**, with music playing. This is where you
   find out whether barge-in works and whether the threshold holds.

---

## Known trade-offs

- **Half-duplex by default.** Guests can't interrupt. The alternative risks the
  head hearing itself and looping. Enable barge-in only after testing at
  volume.
- **One conversation at a time.** With a crowd, the head talks to whoever is
  nearest and loudest. Handling a group is a much harder problem.
- **Costume vision costs a round trip.** ~300–500ms added to the greeting.
  Worth it — a specific costume callback is what makes people call their
  friends over. Disable with `SB_COSTUME_VISION=false` if latency bothers you.
- **Cloud dependency.** Local fallbacks exist for both STT and TTS but are
  slower and lower quality. Install and test them before the night.
