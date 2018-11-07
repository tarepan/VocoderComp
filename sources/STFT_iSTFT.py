import librosa
import numpy as np


def testSTFT_iSTFT(waveform, sr):
     C = librosa.core.stft(waveform)
     reconstructed = librosa.core.istft(C)
     return reconstructed


def testSTFTspectrogram_iSTFT(waveform, sr, mode="Noise"):
    """
    test 'STFT encoding into spectrogram (discard phase info) + iSTFT decoding)'
    phase information is completely discarded after stft
    phase is random
    """
    C = librosa.core.stft(waveform)
    magnitude, _ = librosa.magphase(C)
    phaseCuriousC = None
    if mode=="Noise":
        # random noise phase
        random_angles = np.exp(2j * np.pi * np.random.rand(*magnitude.shape))
        phaseCuriousC = magnitude.astype(np.complex64) * random_angles
    elif mode=="Zero":
    # random noise phase
        phaseCuriousC = magnitude.astype(np.complex64)

    reconstructed = librosa.core.istft(phaseCuriousC)
    return reconstructed
