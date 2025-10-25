#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pyldb unit testing
"""

import pytest
from pyldb.core import PyLdB
import numpy as np
import os.path

def test_table_import():
# Get the absolute path of the directory containing this script
    test_sig_fname = os.path.join("misc", "panair_r1.sig")
    pldb_instance = PyLdB()
    pldb_instance._import_sig(test_sig_fname, header_lines=3)
    sones_table_indep = pldb_instance.sones_table_indep
    sum_factor_table = pldb_instance.sum_factor_table
    sones_table_indep_test = np.array([0.181, 0.196, 0.212, 0.23, 0.248])
    sum_factor_table_test = np.array([0.1, 0.122, 0.14, 0.158, 0.174])

    assert np.allclose(sones_table_indep[:5], sones_table_indep_test, rtol=0.0, atol=10e-8)
    assert np.allclose(sum_factor_table[:5], sum_factor_table_test, rtol=0.0, atol=10e-8)

def test_import_sig():
    test_sig_fname = os.path.join("misc", "panair_r1.sig")
    pldb_instance = PyLdB()
    pldb_instance._import_sig(test_sig_fname, header_lines=3)

    time_test = np.array([0.0, 1.2986455383213524328312e-02,
                          2.5972910766412837801909e-02, 3.8959366149626362130221e-02,
                          5.1945821532825675603817e-02])
    pressure_test = np.array([0.0, -3.0142774286245854907653e-11,
                              -6.0274806344079990772505e-11, -9.0404394856597904208941e-11,
                              -1.2053215227760139782411e-10])

    assert np.allclose(pldb_instance.sig_time_ms[:5], time_test, rtol=0.0, atol=10e-12)
    assert np.allclose(pldb_instance.sig_pressure_psf[:5], pressure_test, rtol=0.0, atol=10e-12)

def test_PLDB():
    test_sig_fname = os.path.join("misc", "panair_r1.sig")
    time, pressure = np.genfromtxt(test_sig_fname, skip_header=3).T

    PLdB = pyldb.perceivedloudness(time, pressure, pad_front=6, pad_rear=6,
                                   len_window=800)
    PLdB_test = 77.67985293502309

    assert np.allclose(PLdB, PLdB_test, rtol=0.0, atol=10e-12)
