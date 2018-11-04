import time
from pathlib import Path

import pyworld as pw
import librosa
from scipy.io.wavfile import read

from .WORLD import TestWORLD
from .STFT_iSTFT import testSTFTiSTFT

def contest(process, waveform, sr, *args):
    """
    Measure performance
    """
    startTime = time.time()
    generated_waveform = process(waveform, sr, *args)
    elapsed_time = time.time() - startTime
    return generated_waveform, elapsed_time

def compare():
    filepath = Path("./data/vcc2016/f1/uemura_normal_050.wav")

    # In order to avoid resampling effect
    sr_acquired = read(filepath)[0]
    waveform, sr_librosa = librosa.load(filepath, sr=sr_acquired, mono = True, dtype="float64")
    assert sr_acquired == sr_librosa

    # comparison
    WORLD_wave, WORLD_time = contest(TestWORLD, waveform, sr_librosa)
    pureSTFT_wave, pureSTFT_time = contest(testSTFTiSTFT, waveform, sr_librosa)

    # save origin and reconstructed
    librosa.output.write_wav("./results/origin.wav", waveform.astype("float32"), sr_librosa)
    librosa.output.write_wav("./results/world.wav", WORLD_wave.astype("float32"), sr_librosa)
    librosa.output.write_wav("./results/pureSTFT.wav", pureSTFT_wave.astype("float32"), sr_librosa)
    print(f"time WORLD: {WORLD_time}")
    print(f"time pureSTFT: {pureSTFT_time}")
    return None
