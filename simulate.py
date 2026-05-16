#!/usr/bin/env python3
"""
Dub Siren audio simulator.
Renders every preset to a WAV file so you can hear them without hardware.
Matches Synthesiser.c / Waveforms.c logic exactly.

Install deps:  pip install numpy scipy
Run:           python simulate.py
Output:        simulation_output/*.wav
"""

import math, os, numpy as np
from scipy.io import wavfile

SAMPLE_RATE          = 96000
PREEMPTIVE_GATE      = 0.01   # seconds  (PREEMPTIVE_GATE_PERIOD in C)
DELAY_BUFFER_SIZE    = 128000
RENDER_DURATION      = 4.5    # seconds per preset

# ─────────────────────────────────────────────────────────────────────────────
# Waveform generators  (mirror of Waveforms.c)
# ─────────────────────────────────────────────────────────────────────────────

def _lim(p):
    return p % 1.0

def wf_sine(p):
    return math.sin(2.0 * math.pi * p)

def wf_asymmetric_sine(p, shape):
    shape = max(1e-9, min(shape, 1.0 - 1e-9))
    skewed = (p / shape * 0.5) if p < shape else (0.5 + (p - shape) / (1.0 - shape) * 0.5)
    return wf_sine(_lim(skewed - 0.25))

def wf_triangle(p, shape):
    shape = max(1e-9, min(shape, 1.0 - 1e-9))
    return (-1.0 + 2.0 * p / shape) if p < shape else (1.0 - 2.0 * (p - shape) / (1.0 - shape))

def wf_sawtooth(p, shape):
    nw = min(max((1.0 + shape * shape * 10.0) * p, 0.0), 1.0)
    return 2.0 * (nw - 0.5)

def wf_square(p, shape):
    return -1.0 if p < shape else 1.0

def wf_stepped_triangle(p, shape):
    n = 3 + round(shape * 29.0)
    m = float(n - 1)
    nw = (math.floor((2.0 * p - 1.0) * m) / m) if p > 0.5 else (math.ceil((-2.0 * p + 1.0) * m) / m)
    return -2.0 * (nw - 0.5)

def wf_stepped_sawtooth(p, shape):
    n = 3 + round(shape * 29.0)
    nw = math.floor(p * n) / float(n - 1)
    return 2.0 * (nw - 0.5)

# VCO waveforms (simplified / non-bandwidth-limited — fine for auditioning)
def vco_triangle(p):
    return (4.0 * p - 1.0) if p < 0.5 else (3.0 - 4.0 * p)

def vco_sawtooth(p):
    return 1.0 - 2.0 * p

def vco_square(p):
    return 1.0 if p < 0.5 else -1.0

HALF_PULSE = 0.125
def vco_pulse(p):
    return 1.0 if (p < HALF_PULSE or p > 1.0 - HALF_PULSE) else -1.0

class _LFSR:
    def __init__(self):
        self.reg, self.val, self.ctr = 0xACE1, 1.0, 0
    def get(self, freq):
        spu = max(1, int(SAMPLE_RATE / max(abs(freq), 0.5)))
        self.ctr += 1
        if self.ctr >= spu:
            self.ctr = 0
            lsb = (self.reg & 1) != 0
            self.reg = ((self.reg >> 1) ^ (0xB400 if lsb else 0)) & 0xFFFF
            self.val = 1.0 if lsb else -1.0
        return self.val

LFO_FN = {
    'sine':             wf_asymmetric_sine,
    'triangle':         wf_triangle,
    'sawtooth':         wf_sawtooth,
    'square':           wf_square,
    'stepped_triangle': wf_stepped_triangle,
    'stepped_sawtooth': wf_stepped_sawtooth,
}

# ─────────────────────────────────────────────────────────────────────────────
# Simple IIR low-pass filter  (FirstOrderFilter.c)
# ─────────────────────────────────────────────────────────────────────────────

class LP:
    def __init__(self, fc):
        self.a = math.exp(-2.0 * math.pi * fc / SAMPLE_RATE)
        self.y = 0.0
    def __call__(self, x):
        self.y = (1.0 - self.a) * x + self.a * self.y
        return self.y

# ─────────────────────────────────────────────────────────────────────────────
# Renderer  (AudioUpdate loop from Synthesiser.c)
# ─────────────────────────────────────────────────────────────────────────────

def render(p, duration=RENDER_DURATION):
    n        = int(SAMPLE_RATE * duration)
    out      = np.zeros(n, dtype=np.float32)
    gate_lp  = LP(100.0)
    dt_lp    = LP(1.0)
    df_lp    = LP(p.get('delayFilterFrequency', 20000.0)) if p.get('delayFilterType') == 'lp' else None
    delay    = np.zeros(DELAY_BUFFER_SIZE, dtype=np.float32)
    d_idx    = 0
    lfsr     = _LFSR()
    lfo_clk  = 0.0
    vco_clk  = 0.0
    gate     = True
    trigger  = True

    lfo_fn = LFO_FN[p['lfoWaveform']]

    for i in range(n):
        if trigger:
            trigger = False; lfo_clk = 0.0; gate = True

        # LFO
        lfo_val  = lfo_fn(lfo_clk, p['lfoShape'])
        lfo_clk += p['lfoFrequency'] / SAMPLE_RATE
        if p['lfoGateControl'] and lfo_clk >= (1.0 - PREEMPTIVE_GATE * p['lfoFrequency']):
            gate = False
        lfo_clk  = _lim(lfo_clk)
        vco_freq = p['vcoFrequency'] + p['lfoAmplitude'] * lfo_val

        # VCO
        vcw = p['vcoWaveform']
        if   vcw == 'sine':     sample = wf_sine(vco_clk)
        elif vcw == 'triangle': sample = vco_triangle(vco_clk)
        elif vcw == 'sawtooth': sample = vco_sawtooth(vco_clk)
        elif vcw == 'square':   sample = vco_square(vco_clk)
        elif vcw == 'pulse':    sample = vco_pulse(vco_clk)
        else:                   sample = lfsr.get(vco_freq)           # noise
        vco_clk = _lim(vco_clk + vco_freq / SAMPLE_RATE)

        # Gate envelope
        sample *= gate_lp(1.0 if gate else 0.0)
        sample *= 0.25  # attenuate (matches C code)

        # Delay
        delay[d_idx] = sample
        dt         = dt_lp(p['delayTime'])
        r_off      = min(int(dt * SAMPLE_RATE), DELAY_BUFFER_SIZE - 1)
        r_idx      = (d_idx - r_off) % DELAY_BUFFER_SIZE
        d_smp      = p['delayFeedback'] * delay[r_idx]
        if df_lp:
            d_smp  = df_lp(d_smp)
        delay[d_idx] = min(max(delay[d_idx] + d_smp, -1.0), 1.0)
        d_idx      = (d_idx + 1) % DELAY_BUFFER_SIZE
        sample    += d_smp

        out[i] = sample

    return out

# ─────────────────────────────────────────────────────────────────────────────
# Preset definitions  (DefaultPresets.c — exact values)
# ─────────────────────────────────────────────────────────────────────────────

PRESETS = {
    'fogHorn': dict(
        lfoWaveform='sawtooth',    lfoShape=0.870488, lfoFrequency=0.918656,
        lfoAmplitude=-105.533592,  lfoGateControl=True,
        vcoWaveform='square',      vcoFrequency=182.045883,
        delayTime=0.380044,        delayFeedback=0.284430,
        delayFilterType='lp',      delayFilterFrequency=6879.085938,
    ),
    'levelUp': dict(
        lfoWaveform='stepped_sawtooth', lfoShape=0.748286, lfoFrequency=1.156882,
        lfoAmplitude=466.714294,        lfoGateControl=True,
        vcoWaveform='pulse',            vcoFrequency=471.714294,
        delayTime=0.380060,             delayFeedback=0.284444,
        delayFilterType='lp',           delayFilterFrequency=6879.655273,
    ),
    'marioCoin': dict(
        lfoWaveform='square',      lfoShape=0.347211, lfoFrequency=5.576817,
        lfoAmplitude=374.693115,   lfoGateControl=True,
        vcoWaveform='square',      vcoFrequency=1380.171265,
        delayTime=0.380063,        delayFeedback=0.284503,
        delayFilterType='lp',      delayFilterFrequency=6878.091309,
    ),
    'bombFalling': dict(
        lfoWaveform='stepped_sawtooth', lfoShape=1.0,   lfoFrequency=0.418580,
        lfoAmplitude=-716.046814,       lfoGateControl=True,
        vcoWaveform='pulse',            vcoFrequency=1123.452881,
        delayTime=0.380076,             delayFeedback=0.284415,
        delayFilterType='lp',           delayFilterFrequency=6877.664551,
    ),
    'policeSiren': dict(
        lfoWaveform='triangle',    lfoShape=0.258210, lfoFrequency=0.170868,
        lfoAmplitude=688.041199,   lfoGateControl=False,
        vcoWaveform='triangle',    vcoFrequency=1013.351929,
        delayTime=0.380083,        delayFeedback=0.284545,
        delayFilterType='lp',      delayFilterFrequency=6879.085938,
    ),
    'classicDubSirenLow': dict(
        lfoWaveform='triangle',    lfoShape=0.497863, lfoFrequency=1.808816,
        lfoAmplitude=148.919556,   lfoGateControl=False,
        vcoWaveform='square',      vcoFrequency=205.318161,
        delayTime=0.380047,        delayFeedback=0.284479,
        delayFilterType='lp',      delayFilterFrequency=6876.527832,
    ),
    'dubGlideBass': dict(
        lfoWaveform='sine',        lfoShape=0.5,      lfoFrequency=1.8,
        lfoAmplitude=18.0,         lfoGateControl=False,
        vcoWaveform='triangle',    vcoFrequency=50.0,
        delayTime=0.38,            delayFeedback=0.50,
        delayFilterType='lp',      delayFilterFrequency=600.0,
    ),
    'classicDubSirenHigh': dict(
        lfoWaveform='triangle',    lfoShape=0.497863, lfoFrequency=2.738417,
        lfoAmplitude=650.351685,   lfoGateControl=False,
        vcoWaveform='pulse',       vcoFrequency=655.351685,
        delayTime=0.379998,        delayFeedback=0.284547,
        delayFilterType='lp',      delayFilterFrequency=6876.810547,
    ),
    'bombExploding': dict(
        lfoWaveform='stepped_sawtooth', lfoShape=1.0,   lfoFrequency=0.597933,
        lfoAmplitude=-813.184753,       lfoGateControl=True,
        vcoWaveform='noise',            vcoFrequency=1325.321045,
        delayTime=0.380083,             delayFeedback=0.284484,
        delayFilterType='lp',           delayFilterFrequency=6876.667969,
    ),
    'airRaidSiren': dict(
        lfoWaveform='sine',        lfoShape=0.672904, lfoFrequency=0.027004,
        lfoAmplitude=420.044983,   lfoGateControl=False,
        vcoWaveform='sawtooth',    vcoFrequency=468.001923,
        delayTime=0.380086,        delayFeedback=0.284405,
        delayFilterType='lp',      delayFilterFrequency=6873.684570,
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    out_dir = os.path.join(os.path.dirname(__file__), 'simulation_output')
    os.makedirs(out_dir, exist_ok=True)

    for name, preset in PRESETS.items():
        print(f'Rendering {name} ...', end=' ', flush=True)
        audio = render(preset)
        peak  = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.9          # normalize so every file is audible
        wavfile.write(os.path.join(out_dir, f'{name}.wav'), SAMPLE_RATE,
                      (audio * 32767).astype(np.int16))
        print('OK')

    print(f'\nDone — WAV files saved to: {out_dir}')
