from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
import onnxruntime

NUM_POINTS = 600
GRID = np.linspace(0, 90, NUM_POINTS)


def standardize_input(x, y):
    # ext1 means we fill with zero for the extrapolated parts
    interpolation = InterpolatedUnivariateSpline(x, y, ext=1)
    interpolated = interpolation(GRID)
    return interpolated / interpolated.max()



def reshape_input(vector, cnn=True):
    if cnn: 
        return vector.reshape(-1, 1, NUM_POINTS).astype(np.float32)
    return vector.reshape(-1, NUM_POINTS).astype(np.float32)