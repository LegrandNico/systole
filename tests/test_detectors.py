# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from systole import import_dataset1, import_ppg
from systole.detectors import (
    christov,
    engelse_zeelenberg,
    hamilton,
    moving_average,
    msptd,
    pan_tompkins,
)

signal_df = import_dataset1(modalities=["ECG"])[: 20 * 2000]


def test_msptd():
    """Test msptd function."""
    ppg = import_ppg().ppg.to_numpy()
    peaks = msptd(signal=ppg, sfreq=75, kind="peaks")
    onsets = msptd(signal=ppg, sfreq=75, kind="onsets")
    peaks_onsets = msptd(signal=ppg, sfreq=75, kind="peaks-onsets")
    assert (peaks_onsets[0] == peaks).all()
    assert (peaks_onsets[1] == onsets).all()


def test_moving_average():
    """Test moving average function."""
    peaks = moving_average(signal=signal_df.ecg.to_numpy(), sfreq=1000)
    assert peaks.sum() == 1037313


def test_pan_tompkins():
    """Test moving average function."""
    peaks = pan_tompkins(signal=signal_df.ecg.to_numpy(), sfreq=1000)
    assert peaks.sum() == 1038115


def test_hamilton():
    """Test moving average function."""
    peaks = hamilton(signal=signal_df.ecg.to_numpy(), sfreq=1000)
    assert peaks.sum() == 1066453


def test_christov():
    """Test moving average function."""
    peaks = christov(signal=signal_df.ecg.to_numpy(), sfreq=1000)
    assert peaks.sum() == 1037238


def test_engelse_zeelenberg():
    """Test moving average function."""
    peaks = engelse_zeelenberg(signal=signal_df.ecg.to_numpy(), sfreq=1000)
    assert peaks.sum() == 1036188
