from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
import onnxruntime
import os
import joblib

THIS_DIR = os.path.dirname(os.path.realpath(__file__))


ORT_SESSION = onnxruntime.InferenceSession(os.path.join(THIS_DIR, "ecoder_test.onnx"))

KD_TREE = joblib.load(os.path.join(THIS_DIR, "KDtree.joblib"))

NUM_POINTS = 600
GRID = np.linspace(0, 90, NUM_POINTS)
MODELTYPE = 'CNN'

def standardize_input(x, y):
    # ext1 means we fill with zero for the extrapolated parts
    interpolation = InterpolatedUnivariateSpline(x, y, ext=1)
    interpolated = interpolation(GRID)
    return interpolated / interpolated.max()


def reshape_input(vector, cnn=True):
    if cnn:
        return vector.reshape(-1, 1, NUM_POINTS).astype(np.float32)
    return vector.reshape(-1, NUM_POINTS).astype(np.float32)


def run_encoder(vector): 
    ort_inputs = {ORT_SESSION.get_inputs()[0].name: vector}
    return ORT_SESSION.run(None, ort_inputs)


def query_kd_tree(prediction, k=10): 
    return KD_TREE.query(prediction.flatten(), k=k)


def run_matching(x, y, k=10):
    cnn = True if MODELTYPE == "CNN" else False
    ready_for_encoder = reshape_input(standardize_input(x,y), cnn) 

    encoded_vector = run_encoder(ready_for_encoder)
    kd_tree_response = query_kd_tree(encoded_vector)