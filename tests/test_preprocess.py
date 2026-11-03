import numpy as np

from cognition_pre.config import PreprocessConfig
from cognition_pre.preprocess import Preprocessor, center_square_crop


def test_center_square_crop():
    frame = np.zeros((360, 480, 3), np.uint8)
    crop, (x0, y0) = center_square_crop(frame)
    assert crop.shape == (360, 360, 3)
    assert (x0, y0) == (60, 0)


def test_to_model_rgb():
    pre = Preprocessor(PreprocessConfig(model_input_size=224))
    frame = np.zeros((360, 480, 3), np.uint8)
    crop, _ = pre.process(frame)
    model_in = pre.to_model_rgb(crop)
    assert model_in.shape == (224, 224, 3)
    assert model_in.dtype == np.uint8
