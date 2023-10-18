import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

file = Path(__file__).parent / "samples" / "sample.csv"
X = pd.read_csv(file)
print(X)


def average_filter(values, n=3):
    """
    Calculate the sliding window average for the give time series.
    Mathematically, res[i] = sum_{j=i-t+1}^{i} values[j] / t, where t = min(n, i+1)
    :param values: list.
        a list of float numbers
    :param n: int, default 3.
        window size.
    :return res: list.
        a list of value after the average_filter process.
    """

    if n >= len(values):
        n = len(values)

    res = np.cumsum(values, dtype=float)
    res[n:] = res[n:] - res[:-n]
    res[n:] = res[n:] / n

    for i in range(1, n):
        res[i] /= (i + 1)

    return res

# Values
X = X['value'].values

# FFT
fft = np.fft.fft(X)
mag = np.sqrt(fft.real ** 2 + fft.imag ** 2).ravel()
fft_log = np.log(mag)

# Log Filter
# printing
window = 100
log_filter = average_filter(fft_log, n=window)
plt.title("Shape: %s" % (fft_log.shape,))
plt.plot(fft_log, color="red", label="fft")
plt.plot(log_filter, color="black", label=f"average n={window}")
plt.legend()
plt.show()

# Spectral
spectral = np.exp(fft_log - log_filter)
plt.title("Spectral")
plt.plot(spectral)
plt.show()

# signal
fft.real = fft.real * spectral / mag
fft.imag = fft.imag * spectral / mag

wave_r = np.fft.ifft(fft)
mag = np.sqrt(wave_r.real ** 2 + wave_r.imag ** 2)

plt.title("Shape: %s" % (fft_log.shape,))
#plt.plot(mag, color="red", label="reconstructed")
plt.plot(X, color="black", label=f"original")
# ALARMS
alarms = np.where(mag > 0.1)[0]
for alarm in alarms:
    plt.axvline(alarm, c='red', ls='--', lw=0.2)
plt.legend()
plt.show()