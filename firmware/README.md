# Firmware

Two sketches for two classic ESP32 dev boards (ESP32-WROOM-32), Arduino-ESP32 core 3.x.
Both compile with `arduino-cli compile --fqbn esp32:esp32:esp32 <sketch>` (the CLI bundled
with Arduino IDE 2 was used: receiver 931,139 B flash, mic capture 926,695 B). Neither has
been run on hardware in this project; only compilation and the software stand-ins in
`tools/` were tested.

| sketch | role |
|---|---|
| `bb8_receiver/` | drive board: UDP wheel commands -> TB6612FNG, clamp, slew limit, 300 ms watchdog, latched e-stop, optional encoder PI, telemetry |
| `bb8_mics/` | capture board: 4 x INMP441 on two I2S buses, raw 48 kHz 4-channel UDP stream |

## What runs where

| stage | where | why |
|---|---|---|
| I2S capture, 24->16 bit, packetising | mic ESP32 | real-time, fixed cost |
| STFT, deep localizer (2.07 M params), tracker, controller | companion app (laptop) | about 10 ms per window on the RTX 4060 laptop GPU used here (see `results/localization.json` for measured CPU and GPU latency); far beyond the ESP32 (no FPU-backed tensor path, 520 KB SRAM) |
| duty clamp, slew, watchdog, e-stop, optional wheel PI | drive ESP32 | must keep working when Wi-Fi drops |

A reduced on-device option was considered and not implemented: quadrature demodulation of
the 20-22.5 kHz band to complex baseband at 6 kHz would cut the stream to about 0.77 Mbit/s
and the app could re-modulate it before the STFT, since the network only uses the 16.9-22.9 kHz
bins. Raw streaming was kept because it is simpler to verify.

## 4 x INMP441 wiring (capture board)

A classic ESP32 has two I2S peripherals. Each INMP441 outputs on one slot of a stereo I2S
frame, chosen by its L/R pin, so two mics share one data line and four mics need both
peripherals. **Constraint:** the two buses must share one bit clock, or the channels drift
apart and the inter-mic delays (all under 0.3 ms, i.e. under 14 samples, for a 5 cm array)
become meaningless. I2S0 is therefore the clock master and I2S1 runs as a slave on I2S0's
clocks through two jumper wires.

| signal | ESP32 pin | goes to |
|---|---|---|
| BCLK (I2S0 out, 3.072 MHz) | GPIO 26 | SCK of all 4 mics, and jumper to GPIO 14 |
| WS (I2S0 out, 48 kHz) | GPIO 25 | WS of all 4 mics, and jumper to GPIO 27 |
| BCLK in (I2S1 slave) | GPIO 14 | jumper from GPIO 26 |
| WS in (I2S1 slave) | GPIO 27 | jumper from GPIO 25 |
| SD0 (I2S0 data in) | GPIO 33 | SD of mic 0 and mic 1, 100 kOhm to GND |
| SD1 (I2S1 data in) | GPIO 32 | SD of mic 2 and mic 3, 100 kOhm to GND |
| 3V3, GND | | VDD/GND of all mics, 100 nF at each mic |

L/R pin: mic 0 and mic 2 to GND (left slot), mic 1 and mic 3 to 3V3 (right slot). Mic index
order must match `config.MIC_ANGLES` (45, 135, 225, 315 degrees, counter-clockwise from the
robot's forward axis, 5 cm radius). Keep the SCK/WS fan-out short (under about 15 cm); at
3 MHz long ribbon runs ring. The INMP441 tri-states SD outside its slot, which is why the
pull-down is needed.

**Bus offset.** Even with a shared clock the two DMA engines can start one or more frames
apart. The sketch enables the slave before the master to make that unlikely, but it is not
guaranteed. Measure it once after power-up: play the chirp from straight ahead (0 degrees)
at 1 m and check that the mic-1 vs mic-2 GCC-PHAT lag is near zero (they are symmetric about
the forward axis); a lag of a whole number of samples, constant across chirps, is a bus
offset and must be removed in the app before the STFT. This calibration was not carried out
in this project (no hardware).

**Front-end limit.** The INMP441's decimation filter passes up to about 0.42 fs (about
20 kHz at 48 kHz) and stops by 0.5 fs, so the 20-22.5 kHz chirp is attenuated by the
microphone itself. The simulator's "realistic" channel models this (with the exact roll-off
marked "to be confirmed" against the datasheet), and it is the reason the 17-19.5 kHz
fallback chirp exists.

## Drive board pins (TB6612FNG)

| function | GPIO |
|---|---|
| left AIN1 / AIN2 / PWMA | 25 / 33 / 32 |
| right BIN1 / BIN2 / PWMB | 27 / 14 / 12 |
| STBY | 13 |
| encoders L A/B, R A/B (optional) | 34 / 35 / 36 / 26 |
| battery divider 1:3 (ADC1) | 39 |

GPIO 12 is a strapping pin (flash voltage); the TB6612 input is high-impedance so it does
not pull it at boot, but do not add a pull-up there.

## Network and safety

* Default: the drive board is a WPA2 access point `BB8-DL`; the laptop and mic board join it
  (mic board static IP 192.168.4.3). The mic stream is 3.07 Mbit/s of payload and the AP
  must relay it, so a phone hotspot or travel router with both boards as stations is the
  more robust layout (`WIFI_AP_MODE 0` on the drive board).
* **Change the placeholder passphrase** (`change-me-123`) before use. Anyone on the network
  can send drive commands: the packets carry no authentication. For anything beyond a
  supervised lab demo, add an HMAC over the packet with a shared key.
* Safety limits on the drive board: duty clamped to +/-0.8, slew 4 duty/s, motors stop if
  no valid command for 300 ms, e-stop latched until an explicit zero command, stale or
  out-of-order sequence numbers dropped (a new app session is accepted after the watchdog
  has expired).
