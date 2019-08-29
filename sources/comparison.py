import time
from pathlib import Path

import pyworld as pw
import librosa
from scipy.io.wavfile import read

from .WORLD import TestWORLD
from .STFT_iSTFT import testSTFT_iSTFT, testSTFTspectrogram_iSTFT

def contest(process, waveform, sr, *args):
    """
    Measure performance
    """
    startTime = time.time()
    generated_waveform = process(waveform, sr, *args)
    elapsed_time = time.time() - startTime
    return generated_waveform, elapsed_time

def compare(data):
    # In order to avoid resampling effect
    sr_acquired = read(data["path"])[0]
    waveform, sr_librosa = librosa.load(data["path"], sr=sr_acquired, mono = True, dtype="float64")
    assert sr_acquired == sr_librosa

    # comparison
    WORLD_wave, WORLD_time = contest(TestWORLD, waveform, sr_librosa)
    pureSTFT_wave, pureSTFT_time = contest(testSTFT_iSTFT, waveform, sr_librosa)
    spectrogramNoise_wave, spectrogramNoise_time = contest(testSTFTspectrogram_iSTFT, waveform, sr_librosa, "randomphase")
    spectrogramZero_wave, spectrogramZero_time = contest(testSTFTspectrogram_iSTFT, waveform, sr_librosa, "zerophase")
    phase_random_bias_stft, _ = contest(testSTFTspectrogram_iSTFT, waveform, sr_librosa, "bias_random")
    phase_bias_channelwise_stft, _ = contest(testSTFTspectrogram_iSTFT, waveform, sr_librosa, "bias_channelwise")
    phase_bias_uniform, _ = contest(testSTFTspectrogram_iSTFT, waveform, sr_librosa, "bias_uniform")
    randomphase_at_strongmag, _ = contest(testSTFTspectrogram_iSTFT, waveform, sr_librosa, "randomphase_at_strongmag")
    randomphase_at_weekmag, _ = contest(testSTFTspectrogram_iSTFT, waveform, sr_librosa, "randomphase_at_weekmag")
    mag_rand_at_weak, _ = contest(testSTFTspectrogram_iSTFT, waveform, sr_librosa, "mag_rand_at_weak")
    mag_zero_at_weak, _ = contest(testSTFTspectrogram_iSTFT, waveform, sr_librosa, "mag_zero_at_weak")
    bias_timewise, _ = contest(testSTFTspectrogram_iSTFT, waveform, sr_librosa, "bias_timewise")
    magflip_at_strongmag, _ = contest(testSTFTspectrogram_iSTFT, waveform, sr_librosa, "magflip_at_strongmag")
    freq_shift, _ = contest(testSTFTspectrogram_iSTFT, waveform, sr_librosa, "freq_shift")
    cutout, _ = contest(testSTFTspectrogram_iSTFT, waveform, sr_librosa, "cutout")
    freq_shift_grade, _ = contest(testSTFTspectrogram_iSTFT, waveform, sr_librosa, "freq_shift_grade")
    freq_shift_up, _ = contest(testSTFTspectrogram_iSTFT, waveform, sr_librosa, "freq_shift_up")
    cutout_low, _ = contest(testSTFTspectrogram_iSTFT, waveform, sr_librosa, "cutout_low")
    # save origin and reconstructed
    librosa.output.write_wav(Path("./results")/data["name"]/"origin.wav", waveform.astype("float32"), sr_librosa)

    librosa.output.write_wav(Path("./results")/data["name"]/"world.wav", WORLD_wave.astype("float32"), sr_librosa)
    librosa.output.write_wav(Path("./results")/data["name"]/"pureSTFT.wav", pureSTFT_wave.astype("float32"), sr_librosa)
    librosa.output.write_wav(Path("./results")/data["name"]/"spectrogram_noise.wav", spectrogramNoise_wave.astype("float32"), sr_librosa)
    librosa.output.write_wav(Path("./results")/data["name"]/"spectrogram_zero.wav", spectrogramZero_wave.astype("float32"), sr_librosa)
    librosa.output.write_wav(Path("./results")/data["name"]/"phase_random_bias_stft.wav", phase_random_bias_stft.astype("float32"), sr_librosa)
    librosa.output.write_wav(Path("./results")/data["name"]/"phase_bias_channelwise_stft.wav", phase_bias_channelwise_stft.astype("float32"), sr_librosa)
    librosa.output.write_wav(Path("./results")/data["name"]/"phase_bias_uniform.wav", phase_bias_uniform.astype("float32"), sr_librosa)
    librosa.output.write_wav(Path("./results")/data["name"]/"randomphase_at_strongmag.wav", randomphase_at_strongmag.astype("float32"), sr_librosa)
    librosa.output.write_wav(Path("./results")/data["name"]/"randomphase_at_weekmag.wav", randomphase_at_weekmag.astype("float32"), sr_librosa)
    librosa.output.write_wav(Path("./results")/data["name"]/"mag_rand_at_weak.wav", mag_rand_at_weak.astype("float32"), sr_librosa)
    librosa.output.write_wav(Path("./results")/data["name"]/"mag_zero_at_weak.wav", mag_zero_at_weak.astype("float32"), sr_librosa)
    librosa.output.write_wav(Path("./results")/data["name"]/"bias_timewise.wav", bias_timewise.astype("float32"), sr_librosa)
    librosa.output.write_wav(Path("./results")/data["name"]/"magflip_at_strongmag.wav", magflip_at_strongmag.astype("float32"), sr_librosa)
    librosa.output.write_wav(Path("./results")/data["name"]/"freq_shift.wav", freq_shift.astype("float32"), sr_librosa)
    librosa.output.write_wav(Path("./results")/data["name"]/"cutout.wav", cutout.astype("float32"), sr_librosa)
    librosa.output.write_wav(Path("./results")/data["name"]/"freq_shift_grade.wav", freq_shift_grade.astype("float32"), sr_librosa)
    librosa.output.write_wav(Path("./results")/data["name"]/"freq_shift_up.wav", freq_shift_up.astype("float32"), sr_librosa)
    librosa.output.write_wav(Path("./results")/data["name"]/"cutout_low.wav", cutout_low.astype("float32"), sr_librosa)

    print(f"time WORLD: {WORLD_time}")
    print(f"time pureSTFT: {pureSTFT_time}")
    return None
