from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import write

"Dry Signal generator"
Fs = 44100
drySignal = list()
for freq in range(40, 22000, 5):
    t = np.linspace(0, 0.075, Fs*0.075, endpoint=False)
    drySignal.append(signal.square(2 * np.pi * freq * t))
    # plt.plot(t, signal.square(2 * np.pi * freq * t))
    # plt.ylim(-2, 2)
    # plt.show()
# print drySignal

"Filter Design"
filtSignal = list()
for wn in range(20, 12000, 10):
    wn = float(wn) / (Fs/2)
    b, a = signal.butter(2, wn, 'lowpass')
    filtSignal.append(signal.lfilter(b, a, drySignal))


