# BB8_DL: following a phone by ear with a BB-8 style sphere

A spherical robot with an internal two-wheel cart follows a person whose phone emits short
chirps near or above 20 kHz, heard by a 4 x INMP441 microphone array. This repo contains the
simulator, a deep multichannel localizer, a classical GCC-PHAT / SRP-PHAT baseline, three
controller families (DL + tracker, DL + PID with a learned terrain residual, PPO), a data and
results browser, a companion app that drives an ESP32 over UDP, ESP32 firmware, and a paper.

**Everything evaluated here is simulation.** The firmware compiles and the app was tested
against software stand-ins for the ESP32 and the microphones. No hardware experiment was run.

## Machine used for every reported number

| item | value |
|---|---|
| OS | Windows 11 (10.0.22631) |
| CPU / RAM | AMD Ryzen 9 8945H, 15.3 GB |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB, driver 617.14 |
| Python / PyTorch | 3.13.5 / 2.11.0+cu128 (CUDA 12.8) |
| other packages | pinned in `requirements.txt` |

The device is written into every training and evaluation JSON (`"device"` field).

## Headline results (measured, from `results/*.json`)

Localization, same 8,000 test recordings for every method (`results/localization.json`):

| method | bearing MAE | bearing median | range MAE |
|---|---|---|---|
| SRP-PHAT (+ matched-filter ToF) | 23.27 deg | 5.12 deg | 0.692 m |
| GCC-PHAT LS-TDOA | 25.23 deg | 6.11 deg | 0.692 m |
| DL, magnitude + phase (2,073,603 params) | 13.25 deg | 1.27 deg | 0.117 m |
| DL, magnitude only (ablation) | 89.28 deg | 89.03 deg | 0.136 m |
| DL, trained without multipath (ablation) | 15.78 deg | 1.96 deg | 0.145 m |

Closed-loop follow success, 30 seeded 60 s episodes per terrain, the same scenarios for every
controller (success = within 1.5 m for at least 80% of the time after a 10 s warm-up, and no
tumble). Pooled over six terrains, from `results/closed_loop_{A_us,B_fb}.json`:

| controller | A_us (ideal) | B_fb (realistic, 17-19.5 kHz) |
|---|---|---|
| Oracle bearing/range + pursuit | 80% | 80% |
| GCC-PHAT + Kalman tracker + pursuit | 70% | 65% |
| DL + tracker + pursuit | 74% | 76% |
| DL + cascaded PID | 56% (tumbles) | 52% |
| DL + PID + learned terrain residual | 76% | 77% |
| PPO, 4,005,888 steps, classical features, trained on A_us | 91% | 81% |

On slope and grass even the oracle succeeds in only 50% and 30% of episodes, so control
rather than localization is the bottleneck there; PPO handles slopes (93% on A_us). The
residual learns the duty the flat-ground mixer is missing to hold the current motion on the
current terrain, inferred from 0.5 s of duty and proprioception history. It removes the PID's
tumbles (2.51 to 0.03 tumble steps per episode on A_us). Per-terrain tables are in the paper
and in the JSON.

All figures are in `results/figures/`; the paper is `paper/main.pdf`.

## Layout

```
bb8dl/        simulator and models: acoustics, robot, world, features, model, tracker,
              controllers, residual, rl_env, evaluate, protocol, augment, config
scripts/      dataset generation, training, evaluation, figures, paper tables, debug tools
app/          companion app (checkpoint -> L/R wheel commands over UDP) + status page
browser/      Flask data and results browser
firmware/     bb8_receiver (drive ESP32) and bb8_mics (4 x INMP441 streamer), README
tools/        mock ESP32, fake microphone stream, chirp WAV generator, log comparison
assets/       chirp WAVs (generated)
data/sample/  144 stratified test examples for the browser
checkpoints/  localizer_{full,mag,anechoic}.pt (fp16), residual.pt, rl_ppo.zip
results/      metric JSON, figures, trajectories
paper/        LaTeX source, generated tables and macros
docs/         DESIGN.md: assumptions, sphere dynamics, design history
```

## Setup

```powershell
python -m venv $HOME\.venvs\bb8dl
& $HOME\.venvs\bb8dl\Scripts\Activate.ps1
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Generated data goes to `$env:BB8DL_DATA` (default `~/bb8dl_data`, about 2.3 GB), outside the
repo. The venv lives outside the OneDrive folder on purpose.

## Commands that produced the reported results, in order

```powershell
python scripts/sanity_checks.py                                  # results/sanity.json
python scripts/gen_dataset.py --split train                      # 64,000 examples
python scripts/gen_dataset.py --split val                        # 8,000
python scripts/gen_dataset.py --split test                       # 8,000
python scripts/gen_dataset.py --split train_anechoic --anechoic  # 64,000, ablation
python scripts/train_localizer.py --tag full --epochs 30
python scripts/train_localizer.py --tag mag --mode mag --epochs 30
python scripts/train_localizer.py --tag anechoic --train-split train_anechoic --epochs 30
python scripts/train_residual.py
python scripts/train_rl.py --steps 4000000                       # finished at 4,005,888 steps
python scripts/eval_localization.py
python scripts/eval_closed_loop.py --n 30 --channel A_us         # run in parts, see below
python scripts/eval_closed_loop.py --n 30 --channel B_fb
python scripts/make_figures.py
python scripts/paper_tables.py
python scripts/export_sample.py --n 144
cd paper; pdflatex main; bibtex main; pdflatex main; pdflatex main
```

The A_us closed-loop evaluation was run in four parts because the first process hung after
another GPU process exited (logs in `logs/eval_closed_loop_A_us_part*.log`, gitignored). Its
DL + PID + residual rows were discarded and rerun with the corrected residual. Each run writes
into the same JSON, keyed by method and terrain, and every method sees the same seeded
scenarios.

Interrupted runs, kept for honesty: the localizer without D4 augmentation (stopped after 9
epochs, `results/stopped_localizer_v3_noaug_history.json`), two earlier localizer designs
(`results/aborted_localizer_v{1,2}_history.json`), the second PPO run (crashed at 1,515,520
steps, `results/crashed_rl_run2_curve.json`) and the first residual network, which was
invalid because of a label leak (`results/superseded_residual_v1.json`).

## Browser, companion app, firmware

```powershell
python browser/app.py                       # http://127.0.0.1:5000: waveforms, 4-ch spectrograms, labels, predictions, figures
python tools/make_chirp_wav.py              # 10 s chirp WAVs for the phone
python tools/mock_esp32.py                  # stand-in for the drive ESP32
python app/companion.py --source sim --terrain grass --controller dl_pid_res --esp32 127.0.0.1
python tools/fake_mic_stream.py --terrain flat --seconds 30     # stand-in for the mic ESP32
python app/companion.py --source udp --mics 127.0.0.1 --esp32 127.0.0.1 --gain-db <printed> --calib-range <printed>
```

The companion app serves a status page with an E-STOP on `http://127.0.0.1:8090`, sends
duties clamped to +-0.8, and sends a stop three times on exit. The firmware is described in
`firmware/README.md` (wiring, the I2S constraint, what runs where). Both sketches compile for
`esp32:esp32:esp32` with arduino-cli.

**Phones may not emit above 20 kHz.** Many phone speakers and DACs roll off before 20 kHz. The
realistic channel models this and shows the 20-22.5 kHz chirp losing most of its energy; the
17-19.5 kHz fallback chirp is audible to some people, especially children.

## Limitations

Simulation only. Range requires the phone and robot to share a chirp schedule; we assume a
sync with 100 us jitter and up to 200 us bias and did not build it. The sphere model is planar
Lagrangian pitch dynamics plus simplified roll and yaw, not a full 3-D contact model. The
simulator drives duties up to 1.0 with no slew limit, while the firmware clamps to 0.8 and
slews at 4 per second. The PPO policy sees classical acoustic features, not the deep localizer,
and was trained only on channel A_us. Nothing here is comparable to Daredevil or MilliSonic,
which were evaluated on real hardware. See the paper's limitations section for the full list.

## Security

The UDP control protocol has no authentication. Anyone on the robot's Wi-Fi can drive it. The
firmware ships with a placeholder Wi-Fi password, which must be changed. `firmware/README.md`
suggests an HMAC on each packet.
