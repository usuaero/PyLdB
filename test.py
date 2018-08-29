#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pyldb unit testing
"""

import pytest
import pyldb
import numpy as np
import os.path


def test_PLDB():
    test_sig_fname = os.path.join("misc", "panair_r1.sig")
    time, pressure = np.genfromtxt(test_sig_fname, skip_header=3).T

    PLdB = pyldb.perceivedloudness(time, pressure, pad_front=6, pad_rear=6,
                                   len_window=800)
    PLdB_test = 77.67985293502309

    assert np.allclose(PLdB, PLdB_test, rtol=0.0, atol=10e-12)
