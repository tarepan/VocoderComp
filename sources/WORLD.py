import pyworld as pw

def TestWORLD(waveform, sr, *args):
    """
    Test performance of WORLD (pyworld)
    """
    # extract features
    _f0, t = pw.dio(waveform, sr)    # raw pitch extractor
    f0 = pw.stonemask(waveform, _f0, t, sr)  # pitch refinement
    sp = pw.cheaptrick(waveform, f0, t, sr)  # extract smoothed spectrogram
    ap = pw.d4c(waveform, f0, t, sr)         # extract aperiodicity
    # synthesize waveform
    generated_waveform = pw.synthesize(f0, sp, ap, sr)
    return generated_waveform
