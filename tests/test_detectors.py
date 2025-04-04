# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
from unittest import TestCase

from systole import import_ppg, import_dataset1
from systole.detectors import msptd, moving_average, pan_tompkins, hamilton, christov, engelse_zeelenberg

signal_df = import_dataset1(modalities=["ECG"])[: 20 * 2000]

class TestDetectors(TestCase):
    def test_msptd(self):
        """Test msptd function"""
        ppg = import_ppg().ppg.to_numpy()
        peaks = msptd(signal=ppg, sfreq=75, kind="peaks")
        onsets = msptd(signal=ppg, sfreq=75, kind="onsets")
        peaks_onsets = msptd(signal=ppg, sfreq=75, kind="peaks-onsets")
        assert (peaks_onsets[0] == peaks).all()
        assert (peaks_onsets[1] == onsets).all()

    def test_moving_average(self):
        """Test moving average function"""
        peaks = moving_average(signal=signal_df.ecg.to_numpy(), sfreq=1000)
        assert peaks.sum() == 1037313

    def test_pan_tompkins(self):
        """Test moving average function"""
        peaks = pan_tompkins(signal=signal_df.ecg.to_numpy(), sfreq=1000)
        assert peaks.sum() == 1038115

    def test_hamilton(self):
        """Test moving average function"""
        peaks = hamilton(signal=signal_df.ecg.to_numpy(), sfreq=1000)
        assert peaks.sum() == 1066453

    def test_christov(self):
        """Test moving average function"""
        peaks = christov(signal=signal_df.ecg.to_numpy(), sfreq=1000)
        assert peaks.sum() == 1037238

    def test_engelse_zeelenberg(self):
        """Test moving average function"""
        peaks = engelse_zeelenberg(signal=signal_df.ecg.to_numpy(), sfreq=1000)
        assert peaks.sum() == 1036188


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
