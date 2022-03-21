import numpy as np
import logging


logger = logging.getLogger(__name__)


class MDDM:

    def __init__(self, windowSize: int, delta: float):
        self.n = windowSize
        self.dw = delta
        self.__weights()
        self.epslon = self.__calculate_epslon()
        logger.info(f"Epslon: {self.epslon}")
        self.win = []
        self.mean_t = 0
        self.mean_m = 0

    def __weights(self, method='geometric'):
        r = 1.3
        if method == 'geometric':
            self.weights = np.array(list(range(self.n)))
            self.weights = r ** (self.weights - 1)

        logger.info(f'weights: {str(self.weights)}')

    def __calculate_epslon(self):
        vi = self.weights/self.weights.sum()
        s = sum(vi ** 2)
        return np.sqrt(s/2 * np.log(1/self.dw))

    def __weighted_mean(self):
        return np.average(self.win, weights=self.weights)

    def __reset(self):
        self.win = []
        self.mean_m = 0

    def detect(self, pr):
        if len(self.win) == self.n:
            self.win = self.win[1:]
        self.win.append(pr)
        if len(self.win) < self.n:
            return False
        else:
            self.mean_t = self.__weighted_mean()
            if self.mean_m < self.mean_t:
                self.mean_m = self.mean_t
            self.diff = self.mean_m - self.mean_t
            if self.diff >= self.epslon:
                self.__reset()
                return True
            else:
                return False
