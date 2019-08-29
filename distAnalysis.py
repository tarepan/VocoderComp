from pathlib import Path
import time

import numpy as np
import pyworld as pw
import librosa
from scipy.io.wavfile import read
from matplotlib import pyplot

data_root = Path("./original_data")
dataset = [
    {
        "path": data_root/"hiroshiba_normal_049.wav",
        "name": "hiho"
    },{
        "path": data_root/"tsuchiya_normal_049.wav",
        "name": "tsuchiya"
    }
]
for data in dataset:
    sr_acquired = read(data["path"])[0]
    waveform, sr_librosa = librosa.load(data["path"], sr=sr_acquired, mono = True, dtype="float64")
    assert sr_acquired == sr_librosa

    C = librosa.core.stft(waveform)
    mag, arg = np.abs(C), np.angle(C, deg=False)

    mag_maximum = mag.max()
    mag_copy = mag.copy()
    mag_copy = mag_copy.reshape(-1)

    threthold = 0.01
    small = list(filter(lambda x: x < mag_maximum*threthold, mag_copy))
    print(f"lowers {threthold} ({mag_maximum*threthold}): {len(small)/len(mag_copy)*100}%, n={len(mag_copy) - len(small)}")
    pyplot.hist(mag_copy, bins=100, log=True)
    pyplot.show()
