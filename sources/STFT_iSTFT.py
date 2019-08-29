import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def testSTFT_iSTFT(waveform, sr):
     C = librosa.core.stft(waveform)
     reconstructed = librosa.core.istft(C)
     return reconstructed


def testSTFTspectrogram_iSTFT(waveform, sr, mode=None):
    """
    test 'STFT encoding into spectrogram (discard phase info) + iSTFT decoding)'
    phase information is completely discarded after stft
    phase is random
    """
    # * phase
    #   + zero phase: "zerophase"
    #   + random phase "randomphase"
    #   + biased"bias_"
    #     - random noize "bias_random"
    #     - channel_wise noize "bias_channelwise"
    #     - uniform bias "bias_uniform"
    # * mag/arg relation
    #   + random phase @ strong magnitude "randomphase_at_strongmag"
    #   + random phase @ week magnitude "randomphase_at_weekmag"
    # * magnitude
    #   + randomized @ strong magnitude "mag_rand_at_strong"
    #   + randomized @ week magnitude "mag_rand_at_weak"
    C = librosa.core.stft(waveform)
    mag, arg = np.abs(C), np.angle(C, deg=False)
    print(C.shape)
    bias = np.zeros(C.shape)
    # phase modification
    if mode == "zerophase":
        print("zerophase")
        arg = np.zeros(C.shape)
    elif mode == "randomphase":
        print("randomphase")
        arg = np.random.rand(C.shape[0], C.shape[1]) * np.pi
    elif mode == "bias_random":
        print("bias_random")
        bias = np.random.rand(C.shape[0], C.shape[1]) * np.pi
    elif mode == "bias_channelwise":
        print("bias_channelwise")
        for i in range(C.shape[0]):
            single_f_seq = bias[i:i+1, :]
            single_f_seq.fill(np.random.rand() * np.pi)
            bias[i:i+1, :] = single_f_seq
        print(bias)
    elif mode == "bias_timewise":
        print("bias_timewise")
        for i in range(C.shape[1]):
            single_t = bias[:, i:i+1]
            single_t.fill(np.random.rand() * np.pi)
            bias[:, i:i+1] = single_t
        print(bias)
    elif mode == "bias_uniform":
        print("bias_uniform")
        shift = np.random.rand() * np.pi
        bias.fill(shift)
    elif mode == "randomphase_at_strongmag":
        cnt = 0
        maximum = mag.max()
        for feat in range(0, C.shape[0]):
            for time in range(0, C.shape[1]):
                # selection with threthold
                if mag[feat, time] > maximum * 0.01:
                    arg[feat, time] = np.random.rand() * np.pi
                    cnt+=1
        print(f"randomphase_at_strongmag. {cnt}/{C.shape[0]*C.shape[1]} == {cnt/(C.shape[0]*C.shape[1])*100}%")
    elif mode == "randomphase_at_weekmag":
        cnt = 0
        maximum = mag.max()
        for feat in range(0, C.shape[0]):
            for time in range(0, C.shape[1]):
                # selection with threthold
                if mag[feat, time] < maximum * 0.01:
                    arg[feat, time] = np.random.rand() * np.pi
                    cnt+=1
        print(f"randomphase_at_weekmag. {cnt}/{C.shape[0]*C.shape[1]} == {cnt/(C.shape[0]*C.shape[1])*100}%")
    elif mode == "mag_rand_at_strong":
        pass
    elif mode == "mag_rand_at_weak":
        cnt = 0
        maximum = mag.max()
        for feat in range(0, C.shape[0]):
            for time in range(0, C.shape[1]):
                # selection with threthold
                if mag[feat, time] < maximum * 0.01:
                    mag[feat, time] = np.random.rand()
                    cnt+=1
        print(f"mag_rand_at_weak. {cnt}/{C.shape[0]*C.shape[1]} == {cnt/(C.shape[0]*C.shape[1])*100}%")
    elif mode == "mag_zero_at_weak":
        cnt = 0
        maximum = mag.max()
        for feat in range(0, C.shape[0]):
            for time in range(0, C.shape[1]):
                # selection with threthold
                if mag[feat, time] < maximum * 0.01:
                    mag[feat, time] = 0
                    cnt+=1
        print(f"mag_zero_at_weak. {cnt}/{C.shape[0]*C.shape[1]} == {cnt/(C.shape[0]*C.shape[1])*100}%")
    elif mode == "magflip_at_strongmag":
        cnt = 0
        maximum = mag.max()
        for feat in range(1, C.shape[0]):
            for time in range(0, C.shape[1]):
                # selection with threthold
                if mag[feat, time] > maximum * 0.01:
                    # print(f"\nbefore: {mag[feat-20, time]}, {mag[feat, time]}")
                    low = mag[feat-2, time]
                    this_point = mag[feat, time]
                    mag[feat, time] = low
                    mag[feat-2, time] = this_point
                    cnt+=1
                    # print(f"after: {mag[feat-20, time]}, {mag[feat, time]}")
        print(f"magflip_at_strongmag. {cnt}/{C.shape[0]*C.shape[1]} == {cnt/(C.shape[0]*C.shape[1])*100}%")
    elif mode == "freq_shift_up":
        print("\nbefore transform:")
        print(mag)
        shift = 2
        for feat in range(C.shape[0]-1, -1+shift, -1):
            mag[feat, :] = mag[feat-shift, :]
            arg[feat, :] = arg[feat-shift, :]
        for feat in range(0, shift):
            print(f"maximum of zero-converted region#{feat}: {mag[feat,:].max()}")
            mag[feat, :] = np.zeros((1, C.shape[1]))
            arg[feat, :] = np.zeros((1, C.shape[1]))
        print("\nfreq_shift (up):")
        print(mag[0:shift+2, :])
        print("......")
        print(mag[C.shape[0]-2-shift:C.shape[0], :])
        print("\n")
    elif mode == "freq_shift":
        print("\nbefore transform:")
        print(mag)
        shift = 1
        for feat in range(0, C.shape[0]-shift):
            mag[feat, :] = mag[feat+shift, :]
            arg[feat, :] = arg[feat+shift, :]
        for feat in range(C.shape[0]-shift, C.shape[0]):
            print(f"maximum of zero-converted region#{feat}: {mag[feat,:].max()}")
            mag[feat, :] = np.zeros((1, C.shape[1]))
            arg[feat, :] = np.zeros((1, C.shape[1]))
        print("\nfreq_shift (down):")
        print(mag[0:shift+2, :])
        print("......")
        print(mag[C.shape[0]-2-shift:C.shape[0], :])
        print("\n")
    elif mode == "freq_shift_grade":
        print("\nbefore grade transform:")
        print(mag)
        shift = 1
        for feat in range(0, C.shape[0]-shift):
            mag[feat, :] = mag[feat+shift, :]
            arg[feat, :] = arg[feat+shift, :]
        for feat in range(C.shape[0]-shift, C.shape[0]):
            print(f"maximum of zero-converted region#{feat}: {mag[feat,:].max()}")
            mag[feat, :] = np.zeros((1, C.shape[1]))
            arg[feat, :] = np.zeros((1, C.shape[1]))
        # additional shift at high frequency
        shift = 3 - 1
        border_line = 100
        print(f"border missing max: {mag[border_line:border_line+shift, :].max()}")
        for feat in range(border_line, C.shape[0]-shift):
            mag[feat, :] = mag[feat+shift, :]
            arg[feat, :] = arg[feat+shift, :]
        # for feat in range(C.shape[0]-shift, C.shape[0]):
        #     print(f"maximum of zero-converted region#{feat}: {mag[feat,:].max()}")
        #     mag[feat, :] = np.zeros((1, C.shape[1]))
        #     arg[feat, :] = np.zeros((1, C.shape[1]))
    elif mode == "cutout":
        border_line = 400
        cut_line = border_line
        print(f"\ncutout max: {mag[cut_line:-1, :].max()}")
        print(mag)
        for feat in range(cut_line, C.shape[0]):
            mag[feat, :] = np.zeros((1, C.shape[1]))
        print("after cutout:")
        print(mag)
    elif mode == "cutout_low":
        border_line = 15
        cut_line = border_line
        print(f"\nlow cutout max: {mag[0:cut_line+1, :].max()}")
        print(mag)
        for feat in range(0, cut_line + 1):
            mag[feat, :] = np.zeros((1, C.shape[1]))
        print("after cutout:")
        print(mag)


    # print(f"arg: {arg}")
    # print(f"bias: {bias}")
    print(bias)
    mod_C = (mag + 0j) * np.exp(1j*arg) * np.exp(1j*bias)
    librosa.display.specshow(np.abs(mod_C[0:200, :]))
    plt.show()

    reconstructed = librosa.core.istft(mod_C)
    return reconstructed
