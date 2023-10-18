import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

file = Path(__file__).parent / "samples" / "sample.csv"
X = pd.read_csv(file)

points = X['value'].iloc[210:220].values

last_number = points[-1]
n = len(points)
slopes = [(last_number - y)/(n - 1 - i) for i, y in enumerate(points[:-1])]
print("slopes: %s" % slopes)

predicted_value = points[1] + np.sum(slopes)
print("Values = %s" % str(points))
print("Predicted Value = %2.4f" % predicted_value)

# function
def estimate_point(values):
    if len(values) < 2:
        raise ValueError("Must be at least dim(values) = 2")
    last_number = values[-1]
    n = len(points)
    slopes = [(last_number - y)/(n - 1 - i) for i, y in enumerate(points[:-1])]
    predicted_value = points[1] + np.sum(slopes)
    return predicted_value

# Test
print("Let's estimate 3 points:")
print("Values = %s" % str(points))
predicted_value = [estimate_point(points)] * 3
print(np.concatenate((points, predicted_value)))