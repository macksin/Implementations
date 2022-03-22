import numpy as np
import matplotlib.pyplot as plt
from mcdiarmid import MDDM

# make dataset
N = 100
x = np.linspace(0, N, N*100)
y = np.sin(x)

# Targets
yt = y > 0
# Abrupt Change
ichange = int(len(yt)/2)
yt[ichange:] = ~yt[ichange:]

mddm = MDDM(windowSize=5, delta=0.1)

# Detect
true_labels = y > 0
detector = []
i = 0
for predicted, true in zip(true_labels, yt):
    correct = true == predicted
    detected = mddm.detect(correct)
    detector.append(detected)
    if detected:
        print("Position: %d" % i)
        alarm = i
    i += 1

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
ax[0].plot(x, y, label='real_signal')
ax[0].plot(x[ichange:], yt[ichange:], label='abrupt change')
ax[0].plot(x[:ichange], yt[:ichange], label='real')
ax[0].legend()
ax[1].plot(x, detector, label='detector', color='red', lw=2.0)
ax[1].legend()

plt.show()
