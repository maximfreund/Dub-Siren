#!/usr/bin/env python3
"""
Retuned preset simulations — compare *_retune.wav against originals.

Changes made:
  - classicDubSirenLow/High : much higher delay feedback + darker filter
  - airRaidSiren             : LFO 0.027 Hz -> 0.12 Hz (was 37 s/cycle, now ~8 s)
  - policeSiren              : LFO 0.17 Hz -> 0.80 Hz (realistic siren pace)
  - fogHorn                  : feedback 0.28 -> 0.42, filter slightly darker
  - bombExploding            : feedback 0.28 -> 0.55, filter much darker (rumble tail)
  - bombFalling              : feedback 0.28 -> 0.40
  - levelUp / marioCoin      : feedback and filter tuned to match their character

Run:    python simulate_retune.py
Output: simulation_output/*_retune.wav
"""

import importlib.util, os, copy
import numpy as np
from scipy.io import wavfile

_spec = importlib.util.spec_from_file_location(
    "simulate", os.path.join(os.path.dirname(__file__), "simulate.py"))
_sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sim)
render      = _sim.render
SAMPLE_RATE = _sim.SAMPLE_RATE
ORIG        = _sim.PRESETS          # originals for reference

def retune(name, **changes):
    d = copy.copy(ORIG[name])
    d.update(changes)
    return name, d

RETUNED = dict([

    # ── fogHorn ───────────────────────────────────────────────────────────────
    # Harbor ambience: a bit of echo, slightly darker filter, not heavy dub
    retune('fogHorn',
        delayFeedback=0.42,
        delayFilterFrequency=4800.0,
    ),

    # ── levelUp ───────────────────────────────────────────────────────────────
    # Retro game: brighter filter, slightly more feedback for a zingy repeat
    retune('levelUp',
        delayFeedback=0.34,
        delayFilterFrequency=8500.0,
    ),

    # ── marioCoin ─────────────────────────────────────────────────────────────
    # Coin bling: very fast decay (low feedback), very bright
    retune('marioCoin',
        delayFeedback=0.18,
        delayFilterFrequency=11000.0,
    ),

    # ── bombFalling ───────────────────────────────────────────────────────────
    # More presence on the descending tail
    retune('bombFalling',
        delayFeedback=0.40,
        delayFilterFrequency=5000.0,
    ),

    # ── policeSiren ───────────────────────────────────────────────────────────
    # LFO speed: 0.17 Hz (6 s/cycle) -> 0.80 Hz (1.25 s/cycle — realistic)
    # Delay kept dry/bright — police sirens don't have dub echo
    retune('policeSiren',
        lfoFrequency=0.80,
        delayFeedback=0.20,
        delayFilterFrequency=7500.0,
    ),

    # ── classicDubSirenLow ────────────────────────────────────────────────────
    # THE dub siren fix: heavy echo, dark filter — this is what "dub" means
    retune('classicDubSirenLow',
        delayFeedback=0.62,
        delayFilterFrequency=3000.0,
    ),

    # ── classicDubSirenHigh ───────────────────────────────────────────────────
    # Same treatment, slightly more feedback to match its higher, wilder range
    retune('classicDubSirenHigh',
        delayFeedback=0.68,
        delayFilterFrequency=3500.0,
    ),

    # ── bombExploding ─────────────────────────────────────────────────────────
    # Big boom deserves a long dark rumbling tail
    retune('bombExploding',
        delayFeedback=0.55,
        delayFilterFrequency=1800.0,
    ),

    # ── airRaidSiren ──────────────────────────────────────────────────────────
    # LFO: 0.027 Hz (37 s/cycle) -> 0.12 Hz (~8 s/cycle — real air raid pace)
    retune('airRaidSiren',
        lfoFrequency=0.12,
        delayFeedback=0.32,
        delayFilterFrequency=5500.0,
    ),

    # dubGlideBass is already well-tuned — no retune needed
])

if __name__ == '__main__':
    out_dir = os.path.join(os.path.dirname(__file__), 'simulation_output')
    os.makedirs(out_dir, exist_ok=True)

    print("Rendering retuned presets...\n")
    for name, preset in RETUNED.items():
        out_name = f'{name}_retune'
        print(f'  {out_name} ...', end=' ', flush=True)
        audio = render(preset)
        peak  = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.9
        wavfile.write(
            os.path.join(out_dir, f'{out_name}.wav'),
            SAMPLE_RATE,
            (audio * 32767).astype(np.int16),
        )
        print('OK')

    print(f'\nDone -- compare simulation_output/<name>.wav vs <name>_retune.wav')
    print()
    print('Key changes to check:')
    print('  classicDubSirenLow_retune  -- heavier echo, darker (vs thin original)')
    print('  classicDubSirenHigh_retune -- heavier echo, darker (vs thin original)')
    print('  airRaidSiren_retune        -- sweeps every ~8 s (vs 37 s original)')
    print('  policeSiren_retune         -- sweeps every 1.25 s (vs 6 s original)')
    print('  bombExploding_retune       -- dark rumble tail (vs dry original)')
