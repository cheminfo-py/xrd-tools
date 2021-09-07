from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
import onnxruntime
import os
import joblib
from .utils import read_pickle

THIS_DIR = os.path.dirname(os.path.realpath(__file__))


ENCODER_SESSION = onnxruntime.InferenceSession(os.path.join(THIS_DIR, "20210907_encoder.onnx"))
#DECODER_SESSION = onnxruntime.InferenceSession(os.path.join(THIS_DIR, "20210907_decoder.onnx"))

KD_TREE = joblib.load(os.path.join(THIS_DIR, "kdtree.joblib"))
LABEL_TO_NAME = read_pickle(os.path.join(THIS_DIR, 'label_dict.pkl'))

NUM_POINTS = 400
GRID = np.linspace(0, 70, NUM_POINTS)
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
    ort_inputs = {ENCODER_SESSION.get_inputs()[0].name: vector}
    return ENCODER_SESSION.run(None, ort_inputs)

def run_decoder(vector): 
    ort_inputs = {DECODER_SESSION.get_inputs()[0].name: vector}
    return DECODER_SESSION.run(None, ort_inputs)

def query_kd_tree(prediction, k=10): 
    return KD_TREE.query(prediction.reshape(1,-1), k=k)


def prepare_encoding(x,y):
    x = np.array(x)
    y = np.array(y)
    cnn = True if MODELTYPE == "CNN" else False
    ready_for_encoder = reshape_input(standardize_input(x,y), cnn) 
    return ready_for_encoder

def encode(x):
    encoded_vector = run_encoder(x)
    encoded_vector = np.array(encoded_vector)
    return encoded_vector

def decode(x):
    decoded_vector = run_decoder(x)
    decoded_vector = np.array(decoded_vector)
    return decoded_vector

def run_denoising(x,y) -> np.ndarray:
    encoder_input = prepare_encoding(x,y)
    encoded_vector = encode(encoder_input)
    decoded = decode(encoded_vector)
    return decoded

def run_matching(x, y, k: int=10):
    encoder_input = prepare_encoding(x,y)
    encoded_vector = encode(encoder_input)
    kd_tree_response = query_kd_tree(encoded_vector, k)

    names = [LABEL_TO_NAME[number] for number in kd_tree_response[1][0]]
    return names

def reshape_input(vector, cnn=True):
    if cnn: 
        return vector.reshape(-1, 1, NUM_POINTS).astype(np.float32)
    return vector.reshape(-1, NUM_POINTS).astype(np.float32)
