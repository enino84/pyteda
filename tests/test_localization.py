# -*- coding: utf-8 -*-
"""Tests for pyteda.models._localization."""

import numpy as np
import pytest

from pyteda.models._localization import (
    resolve_radius,
    pairwise_radius,
    radius_at,
)


class TestResolveRadius:
    def test_scalar_int(self):
        r = resolve_radius(2, n_state=10)
        assert r.shape == (10,)
        assert np.all(r == 2.0)

    def test_scalar_float(self):
        r = resolve_radius(2.5, n_state=10)
        assert r.shape == (10,)
        assert np.all(r == 2.5)

    def test_array(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = resolve_radius(arr, n_state=5)
        assert np.array_equal(r, arr)

    def test_dict_with_var_blocks(self):
        var_blocks = {"q": slice(0, 5), "psi": slice(5, 10)}
        r = resolve_radius({"q": 2, "psi": 4}, n_state=10, var_blocks=var_blocks)
        assert np.all(r[:5] == 2.0)
        assert np.all(r[5:] == 4.0)

    def test_dict_without_var_blocks_raises(self):
        with pytest.raises(ValueError, match="var_blocks"):
            resolve_radius({"q": 2}, n_state=10, var_blocks=None)

    def test_dict_unknown_block_raises(self):
        var_blocks = {"q": slice(0, 5), "psi": slice(5, 10)}
        with pytest.raises(ValueError, match="Unknown variable block"):
            resolve_radius({"foo": 2}, n_state=10, var_blocks=var_blocks)

    def test_dict_missing_block_raises(self):
        var_blocks = {"q": slice(0, 5), "psi": slice(5, 10)}
        with pytest.raises(ValueError, match="missing block"):
            resolve_radius({"q": 2}, n_state=10, var_blocks=var_blocks)

    def test_array_wrong_size_raises(self):
        with pytest.raises(ValueError, match="shape"):
            resolve_radius(np.array([1.0, 2.0]), n_state=10)

    def test_negative_scalar_raises(self):
        with pytest.raises(ValueError, match="positive"):
            resolve_radius(-1.0, n_state=10)

    def test_zero_array_raises(self):
        with pytest.raises(ValueError, match="positive"):
            resolve_radius(np.zeros(10), n_state=10)

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported"):
            resolve_radius("invalid", n_state=10)

    def test_three_forms_equivalent(self):
        """int=2, dict={'x':2}, array=ones*2 must produce identical output."""
        var_blocks = {"x": slice(0, 10)}
        r1 = resolve_radius(2, n_state=10)
        r2 = resolve_radius({"x": 2}, n_state=10, var_blocks=var_blocks)
        r3 = resolve_radius(np.full(10, 2.0), n_state=10)
        assert np.array_equal(r1, r2)
        assert np.array_equal(r1, r3)


class TestPairwiseRadius:
    def test_uniform_mean_equals_value(self):
        r_arr = np.full(5, 3.0)
        R = pairwise_radius(r_arr, combine="mean")
        assert R.shape == (5, 5)
        assert np.allclose(R, 3.0)

    def test_uniform_min_equals_value(self):
        r_arr = np.full(5, 3.0)
        R = pairwise_radius(r_arr, combine="min")
        assert np.allclose(R, 3.0)

    def test_mean_default_is_mean(self):
        r_arr = np.array([2.0, 4.0])
        R = pairwise_radius(r_arr)
        assert R[0, 0] == 2.0
        assert R[1, 1] == 4.0
        assert R[0, 1] == 3.0  # (2+4)/2
        assert R[1, 0] == 3.0

    def test_min_picks_smaller(self):
        r_arr = np.array([2.0, 4.0])
        R = pairwise_radius(r_arr, combine="min")
        assert R[0, 1] == 2.0
        assert R[1, 0] == 2.0

    def test_symmetric(self):
        r_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        for combine in ("mean", "min"):
            R = pairwise_radius(r_arr, combine=combine)
            assert np.allclose(R, R.T)

    def test_invalid_combine_raises(self):
        r_arr = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="combine"):
            pairwise_radius(r_arr, combine="median")


class TestRadiusAt:
    def test_returns_float_at_index(self):
        r_arr = np.array([1.0, 2.5, 3.0])
        assert radius_at(r_arr, 0) == 1.0
        assert radius_at(r_arr, 1) == 2.5
        assert radius_at(r_arr, 2) == 3.0
