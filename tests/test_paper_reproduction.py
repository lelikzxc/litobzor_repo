"""Regression checks for native wafer die encodings used by retained baselines."""
import numpy as np
import pytest
from papers.wafer_encoding import decode_die_map


@pytest.mark.parametrize("encoded", [[0, 1, 2], [0, 127, 254], [0, 128, 255]])
def test_die_encoding(encoded):
    assert decode_die_map(np.array([encoded], dtype=np.uint8)).tolist() == [[0, 1, 2]]
    # Missing 'normal' pixels must not turn defective die into normal die.
    assert decode_die_map(np.array([[encoded[0], encoded[2]]], dtype=np.uint8)).tolist() == [[0, 2]]


def test_invalid_encoding_rejected():
    with pytest.raises(ValueError, match="encoding"):
        decode_die_map(np.array([[0, 73, 255]], dtype=np.uint8))
