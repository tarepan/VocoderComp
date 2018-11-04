import librosa

def testSTFTiSTFT(waveform, sr):
     C = librosa.core.stft(waveform)
     reconstructed = librosa.core.istft(C)
     return reconstructed
