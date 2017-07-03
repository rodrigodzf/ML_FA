from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import write

def generate_signal(foldername):
    Fs = 16000
    for freq in range(40, 5000, 500):
        samples = 4096
        t = np.linspace(0, 1 / Fs * samples, samples)  # generate samples
        sig = np.sin(2 * np.pi * freq * t)
        name = foldername + "p" + str(0) + "_" + str(freq) + ".wav"
        write(name, 16000, sig)

# "Filter Design"
# filtSignal = list()
# for wn in range(20, 12000, 10):
#     wn = float(wn) / (Fs / 2)
#     b, a = signal.butter(2, wn, 'lowpass')
#     filtSignal.append(signal.lfilter(b, a, drySignal))

generate_signal('corpus/')