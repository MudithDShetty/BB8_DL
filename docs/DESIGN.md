# Design notes

These notes fix the interfaces before any training. All constants live in `bb8dl/config.py`.

## Problem

A BB-8 style sphere (internal two-wheel cart, hamster-ball drive) follows a person who carries
only a phone. The phone plays a short linear chirp every 50 ms. A 4-mic array in the robot's
head records each chirp; the robot estimates bearing and range to the phone and drives toward it,
stopping at a 0.8 m standoff.

## Acoustic channels

| channel | chirp | front-end |
|---|---|---|
| `A_us` | 20.0-22.5 kHz | ideal transducer + flat mic |
| `B_us` | 20.0-22.5 kHz | phone speaker roll-off + INMP441-like mic + I2S decimation filter |
| `B_fb` | 17.0-19.5 kHz | same realistic front-end (hardware fallback chirp) |

INMP441 is specified to about 15 kHz and its decimation filter passband ends near 0.423 fs
(20.3 kHz at 48 kHz), so `B_us` loses most of its upper band. That is the point of modelling it:
the ultrasonic beacon is a sim-only best case and `B_fb` is what real hardware would likely use.

Range is observable only because we assume the phone emits on a schedule the robot knows
(one-time clock sync). The residual sync error is modelled (100 us jitter per chirp,
up to 200 us per-episode bias). Without sync only bearing and a weak level cue remain.

## Signal path (shared by dataset, closed loop and app)

`acoustics.Scene` (geometry + nuisance) -> `acoustics.build_paths` (direct, ground image,
facade image, obstacle scatter, shoebox images up to order 2, diffuse tail) ->
`acoustics.Renderer.render` (exact fractional delays in the frequency domain, front-end,
white noise) -> audio `(n, 4, 2048)` float32 at 48 kHz.

Consumers:

* `features.STFTFeatures` -> network input `(n, 18, 32, 65)` (256-point STFT, hop 32, bins
  16.9-22.9 kHz): 4 log-power maps; real and imaginary parts of the cross-spectrum
  X_i X_j* for all 6 pairs, normalised per frequency by its sum over frames and zeroed outside
  the known chirp band; time ramp; frequency ramp.
* `model.Localizer` (2.07 M params): a steering layer initialised to SRP-PHAT steering vectors
  for 72 azimuths x 7 elevations turns the 12 phase channels into beam-time maps; a
  circular-padded CNN processes them; a spectral CNN processes log-power; a 2-layer BiGRU
  over the 65 frames feeds two attention poolings (azimuth from the pooled beam maps, range
  from the GRU states).
* `features.Classical` -> SRP-PHAT azimuth, pairwise GCC-PHAT + least-squares azimuth,
  matched-filter time-of-flight range, and a 17-d feature vector for RL.

## Robot

`robot.step` integrates a Lagrangian sphere + pendulum-cart model (pitch), a passive lateral
pendulum (roll, gives downhill drift), and first-order yaw from differential wheel torque.
Inputs are wheel duty cycles `u = (u_L, u_R)` in [-1, 1], exactly what the ESP32 firmware
receives. Physics at 200 Hz, control at 20 Hz (one chirp per decision).

## Terrains

`flat`, `slope` (5-10 deg plane), `rough` (random height field + bumps), `grass` (high rolling
resistance and drag, absorptive ground, 0.5-1.5 dB/m excess attenuation), `indoor` (6-10 m room,
walls block motion, image-source reverberation), `obstacles` (8 posts that block motion,
scatter sound and attenuate the direct path when they occlude it).

`world.make_scenarios(terrain, n, seed)` returns the full episode description (including the
person's trajectory), so every controller sees identical episodes for a given seed.

## Controllers (all output `u` in [-1, 1]^2)

| name | localization | control |
|---|---|---|
| `oracle` | ground truth | pursuit law (upper bound) |
| `gcc_track` | SRP-PHAT + matched filter + Kalman tracker | pursuit law |
| `dl_track` | deep net + Kalman tracker (net's variance sets R) | pursuit law |
| `dl_pid` | deep net + tracker | cascaded PID (outer bearing/range, inner speed/yaw-rate PI) |
| `dl_pid_res` | deep net + tracker | as above + learned terrain residual |
| `rl` | classical acoustic features + proprioception (frame-stacked) | PPO policy |

The residual is a supervised inverse-dynamics model: given the last 0.5 s of proprioception and
commands plus the PID's desired (v, omega), it predicts the wheel duty that achieves them;
the residual is that prediction minus the nominal flat-ground mixer.

## Localizer design history

Two designs that fed per-bin phase (unit phasors) to a shared-weight 2-D CNN barely learned
bearing (validation MAE plateaued near 75 degrees, median near 68, against 90 for chance;
stopped after 7 and 11 epochs; histories in `results/aborted_localizer_v*_history.json`). With a 5 cm array at 17-23 kHz the inter-mic
spacing exceeds half a wavelength, so phase is spatially aliased and the phase-to-direction
map differs per frequency bin; a convolution shared across frequency cannot represent it.
`scripts/debug_steer.py` then showed that a fixed steered response over the STFT reaches the
classical baseline's accuracy only if frequencies are equally weighted *and* out-of-band bins
are masked, which motivated the current features and the physics-initialised steering layer.
A first run of the current design without augmentation overfit (train 7.4 deg vs validation
16.9 deg MAE after epoch 9 of 30, stopped; `results/stopped_localizer_v3_noaug_history.json`);
training now applies the 8 exact symmetries of the square array (`bb8dl/augment.py`).

## Repo layout

```
bb8dl/        library (config, acoustics, features, augment, robot, world, model, tracker,
              controllers, residual, rl_env, evaluate, protocol)
scripts/      entry points (gen_dataset, sanity_checks, train_localizer, eval_localization,
              train_residual, train_rl, eval_closed_loop, make_figures, export_sample, debug_*)
browser/      Flask data + results browser
app/          companion app (loads the localizer checkpoint, sim replay or live UDP audio,
              sends UDP wheel commands)
firmware/     ESP32 sketches: drive receiver, 4 x INMP441 capture
tools/        chirp WAV generator, mock drive ESP32, fake mic-ESP32 stream
assets/       chirp WAVs
results/      metric JSON + figures (committed)
paper/        LaTeX paper
```
