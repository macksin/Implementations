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

plt.plot(x, y)
plt.scatter(x[ichange:], yt[ichange:], label='real')
plt.scatter(x[:ichange], yt[:ichange], label='abrupt change')
plt.legend()
plt.show()


mddm = MDDM(windowSize=10, delta=0.01)

# Detect