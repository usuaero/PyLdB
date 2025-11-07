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

def test_padding_time_shift():
    # Initialize PyLdB instance and set up test data
    instance = PyLdB()
    instance.sig_pressure_psf = np.array([1, 2, 3, 4, 5])
    instance.sig_time_ms = np.array([0, 1, 2, 3, 4])
    instance.dt_ms = 1.0

    # Apply padding
    n_pad_points = 2
    instance.padding(n_pad_points)

    # Expected results
    expected_pressure = np.array([0, 0, 1, 2, 3, 4, 5, 0, 0])
    expected_time = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

    # Assertions
    np.testing.assert_array_equal(instance.sig_pressure_psf, expected_pressure)
    np.testing.assert_array_equal(instance.sig_time_ms, expected_time)

def test_padding_data_points():
    # Create an instance of PyLdB
    instance = PyLdB()
    
    # Set up test data
    instance.sig_pressure_psf = np.array([1, 2, 3, 4, 5], dtype=float)
    instance.sig_time_ms = np.array([0, 1, 2, 3, 4], dtype=float)
    instance.dt_ms = 1.0  # Time step in milliseconds
    n_pad_points = 2  # Number of padding points
    
    # Calculate the expected number of data points
    expected_data_points = len(instance.sig_pressure_psf) + 2 * n_pad_points
    
    # Apply the padding function
    instance.padding(n_pad_points)
    
    # Assert that the number of data points matches the expected value
    assert instance.n_data_points == expected_data_points, (
        f"Expected {expected_data_points} data points, but got {instance.n_data_points}"
    )

def test_window_symmetry():
    # Create an instance of PyLdB
    instance = PyLdB()
    
    # Set up test data
    instance.sig_pressure_psf = np.ones(10)  # Example signal
    len_window = 3  # Length of the window
    
    # Apply the window function
    instance.window(len_window)
    
    # Extract the modified sections
    front_windowed = instance.sig_pressure_psf[:len_window]
    rear_windowed = instance.sig_pressure_psf[-len_window:]
    
    # Assert symmetry: the front windowed section should equal the reverse of the rear windowed section
    np.testing.assert_array_almost_equal(front_windowed, rear_windowed[::-1])

def test_window_boundaries():
    # Create an instance of PyLdB
    instance = PyLdB()
    
    # Set up test data
    instance.sig_pressure_psf = np.ones(10)  # Example signal
    len_window = 3  # Length of the window
    
    # Apply the window function
    instance.window(len_window)
    
    # Assert that the first and last elements of the signal are 0
    assert instance.sig_pressure_psf[0] == 0, "First element is not zero"
    assert instance.sig_pressure_psf[-1] == 0, "Last element is not zero"

def test_hanning_window_analytical():
    # Create an instance of PyLdB
    instance = PyLdB()
    
    # Set up a simple test signal
    instance.sig_pressure_psf = np.ones(10)  # Constant signal
    len_window = 5  # Length of the window
    
    # Apply the window function
    instance.window(len_window)
    
    # Compute the expected output analytically
    hanning_window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(len_window * 2) / (len_window * 2 - 1)))
    expected_output = np.ones(10)
    expected_output[:len_window] *= hanning_window[:len_window]
    expected_output[-len_window:] *= hanning_window[len_window:]
    
    # Assert that the actual output matches the expected output
    np.testing.assert_array_almost_equal(instance.sig_pressure_psf, expected_output)

def test_window_center_unchanged():
    # Create an instance of PyLdB
    instance = PyLdB()
    
    # Set up test data
    instance.sig_pressure_psf = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    len_window = 3  # Length of the window
    
    # Copy the original center values for comparison
    original_center = instance.sig_pressure_psf[len_window:-len_window]
    
    # Apply the window function
    instance.window(len_window)
    
    # Extract the center values after applying the window
    modified_center = instance.sig_pressure_psf[len_window:-len_window]
    
    # Assert that the center values remain unchanged
    np.testing.assert_array_equal(original_center, modified_center)
