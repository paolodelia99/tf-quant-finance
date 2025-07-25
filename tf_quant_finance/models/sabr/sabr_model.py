# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implements the SABR model defined by the following equations.

```
  dF_t = v_t * F_t ^ beta * dW_{F,t}
  dv_t = volvol * v_t * dW_{v,t}
  dW_{F,t} * dW_{v,t} = rho * dt
```

`F_t` is forward. `v_t` is volatility. `beta` is the CEV parameter.
`volvol` is volatility of volatility. `W_{F,t}` and `W_{v,t}` are two
correlated Wiener processes with instantaneous correlation `rho`.

This model supports time dependent parameters. In other words, `beta`,
`volvol`, and `rho` can be functions of time. When all of these parameters are
constants, `beta` != 1, and `enable_unbiased_sampling` is True, the almost
exact sampling procedure described in Ref [1] will be used. Otherwise it will
default to Euler sampling.

### References:
  [1]: Chen B, Oosterlee CW, Van Der Weide H. Efficient unbiased simulation
  scheme for the SABR stochastic volatility model. 2011
  Link: http://ta.twi.tudelft.nl/mf/users/oosterle/oosterlee/SABRMC.pdf
  [2]: Andersen, Leif B.G., Efficient Simulation of the Heston Stochastic
  Volatility Model (January 23, 2007). Available at SSRN:
  https://ssrn.com/abstract=946405 or http://dx.doi.org/10.2139/ssrn.946405
"""

from typing import Optional

import tensorflow as tf

from tf_quant_finance import types
from tf_quant_finance import utils as tff_utils
from tf_quant_finance.math import random_ops as random
from tf_quant_finance.math.root_search.newton import root_finder
from tf_quant_finance.models import generic_ito_process
from tf_quant_finance.models import utils

__all__ = ["SabrModel"]


class SabrModel(generic_ito_process.GenericItoProcess):
    """Implements the SABR model defined by the following equations.

    ```
      dF_t = v_t * F_t ^ beta * dW_{F,t}
      dv_t = volvol * v_t * dW_{v,t}
      dW_{F,t} * dW_{v,t} = rho * dt
    ```

    `F_t` is the forward. `v_t` is volatility. `beta` is the CEV parameter.
    `volvol` is volatility of volatility. `W_{F,t}` and `W_{v,t}` are two
    correlated Wiener processes with instantaneous correlation `rho`.

    This model supports time dependent parameters. In other words, `beta`,
    `volvol`, and `rho` can be functions of time. When all of these parameters are
    constants, `beta` != 1, and `enable_unbiased_sampling` is True, the almost
    exact sampling procedure described in Ref [1] will be used. Otherwise it will
    default to Euler sampling.

    ### Example. Simple example with time independent params.

    ```none
      process = sabr.SabrModel(beta=0.5, volvol=1, rho=0.5, dtype=tf.float32)
      paths = process.sample_paths(
          initial_forward=100,
          initial_volatility=0.05,
          times=[1,2,5],
          time_step=0.01,
          num_samples=10000)
    ```

    ### Example 2. Callable volvol and correlation coefficient.

    ```none
      def volvol_fn(t):
          return tf.where(t < 2, 0.5, 1.0)
      def rho_fn(t):
          return tf.where(t < 1, 0., 0.5)

      process = sabr.SabrModel(
          beta=0.5, volvol=volvol_fn, rho=rho_fn, dtype=tf.float32)

      # Any time where parameters vary drastically (e.g. t=2 above) should be
      # added to `times` to reduce numerical error.
      paths = process.sample_paths(
          initial_forward=100,
          initial_volatility=0.05,
          times=[1,2,5],
          time_step=0.01,
          num_samples=10000)
    ```

    ### References:
    [1]: Chen B, Oosterlee CW, Van Der Weide H. Efficient unbiased simulation
    scheme for the SABR stochastic volatility model. 2011
    Link: http://ta.twi.tudelft.nl/mf/users/oosterle/oosterlee/SABRMC.pdf
    [2]: Andersen, Leif B.G., Efficient Simulation of the Heston Stochastic
    Volatility Model (January 23, 2007). Available at SSRN:
    https://ssrn.com/abstract=946405 or http://dx.doi.org/10.2139/ssrn.946405
    """

    def __init__(
        self,
        beta: types.RealTensor,
        volvol: types.RealTensor,
        rho: types.RealTensor = 0,
        shift: types.RealTensor = 0,
        enable_unbiased_sampling: bool = False,
        psi_threshold: types.RealTensor = 2,
        ncx2_cdf_truncation: int = 10,
        dtype: Optional[tf.DType] = None,
        name: Optional[str] = None,
    ):
        """Initializes the SABR Model.

        Args:
          beta: CEV parameter of SABR model. The type and the shape must be
            one of the following: (a) A scalar real tensor in [0, 1]. When beta = 1,
              the algorithm falls back to the Euler sampling scheme. (b) A python
              callable accepting one real `Tensor` argument time t. It should return
              a scalar real value or `Tensor`.
          volvol: Volatility of volatility. Either a scalar non-negative real
            tensor, or a python callable accepting same parameters as beta callable.
          rho: The correlation coefficient between the two correlated Wiener
            processes for the forward and the volatility. Either a scalar real
            tensor in (-1, 1), or a python callable accepting same parameters as
            beta callable. Default value: 0.
          shift: Tensor holding a non-negative scalar, specifying the shift
            parameter. In the shifted model, the process modeling the forward is
            modified as: dF = sigma * (F + shift) ^ beta * dW.  With this
              modification, negative forward rates are valid as long as F > -shift.
          enable_unbiased_sampling: bool. If True, use the sampling procedure
            described in ref [1]. Default value: False
          psi_threshold: The threshold of applicability of Andersen L. (2008)
            Quadratic Exponential (QE) algorithm. See ref [1] page 13 and ref [2]. A
            scalar float value in [1, 2]. Default value: 2.
          ncx2_cdf_truncation: A positive integer. When computing the CDF of a
            noncentral X2 distribution, it needs to calculate the sum of an
            expression from 0 to infinity. In practice, it needs to be truncated to
            compute an approximate value. This argument is the index of the last
            term that will be included in the sum. Default value: 10.
          dtype: The float type to use. Default value: `tf.float32`
          name: str. The name to give to the ops created by this class.
            Default value: None which maps to the default name `sabr_model`.
        ### References:
        [1]: Chen B, Oosterlee CW, Van Der Weide H. Efficient unbiased simulation
          scheme for the SABR stochastic volatility model. 2011
        Link: http://ta.twi.tudelft.nl/mf/users/oosterle/oosterlee/SABRMC.pdf
        [2]: Andersen, Leif B.G., Efficient Simulation of the Heston Stochastic
        Volatility Model (January 23, 2007). Available at SSRN:
        https://ssrn.com/abstract=946405 or http://dx.doi.org/10.2139/ssrn.946405
        """
        self._dtype = dtype or tf.float32
        self._name = name or "sabr_model"
        self._enable_unbiased_sampling = enable_unbiased_sampling

        self._beta = (
            beta
            if _is_callable(beta)
            else tf.convert_to_tensor(beta, dtype=self._dtype, name="beta")
        )
        self._volvol = (
            volvol
            if _is_callable(volvol)
            else tf.convert_to_tensor(volvol, dtype=self._dtype, name="volvol")
        )
        self._rho = (
            rho
            if _is_callable(rho)
            else tf.convert_to_tensor(rho, dtype=self._dtype, name="rho")
        )
        self._psi_threshold = tf.convert_to_tensor(
            psi_threshold, dtype=self._dtype, name="psi_threshold"
        )
        self._ncx2_cdf_truncation = ncx2_cdf_truncation
        self._shift = tf.convert_to_tensor(shift, dtype=self._dtype, name="shift")

        def drift_fn(_, x):
            return tf.zeros_like(x)

        def _vol_fn(t, x):
            """The volatility function for SABR model."""
            f = x[..., 0]
            v = x[..., 1]

            beta = (
                tf.convert_to_tensor(self._beta(t), dtype=self._dtype, name="beta_fn")
                if _is_callable(self._beta)
                else self._beta
            )
            volvol = (
                tf.convert_to_tensor(
                    self._volvol(t), dtype=self._dtype, name="volvol_fn"
                )
                if _is_callable(self._volvol)
                else self._volvol
            )
            rho = (
                tf.convert_to_tensor(self._rho(t), dtype=self._dtype, name="rho_fn")
                if _is_callable(self._rho)
                else self._rho
            )

            fb = f**beta
            m11 = v * fb * tf.math.sqrt(1 - tf.square(rho))
            m12 = v * fb * rho
            m21 = tf.zeros_like(m11)
            m22 = volvol * v
            mc1 = tf.concat([tf.expand_dims(m11, -1), tf.expand_dims(m21, -1)], -1)
            mc2 = tf.concat([tf.expand_dims(m12, -1), tf.expand_dims(m22, -1)], -1)

            # Set up absorbing boundary.
            should_be_zero = tf.expand_dims(
                tf.expand_dims((beta != 0) & (f <= 0.0), -1), -1
            )
            vol_matrix = tf.concat(
                [tf.expand_dims(mc1, -1), tf.expand_dims(mc2, -1)], -1
            )
            return tf.where(should_be_zero, tf.zeros_like(vol_matrix), vol_matrix)

        # Validate args
        control_dependencies = []
        if not _is_callable(self._beta):
            control_dependencies.append(
                tf.compat.v1.debugging.assert_greater_equal(
                    self._beta,
                    tf.constant(0, dtype=self._dtype),
                    message="Beta must be greater or equal to zero",
                )
            )
            control_dependencies.append(
                tf.compat.v1.debugging.assert_less_equal(
                    self._beta,
                    tf.constant(1, dtype=self._dtype),
                    message="Beta must be less than or equal to 1",
                )
            )
        if not _is_callable(self._volvol):
            control_dependencies.append(
                tf.compat.v1.debugging.assert_greater_equal(
                    self._volvol,
                    tf.constant(0, dtype=self._dtype),
                    message=(
                        "Volatility of volatility must be greater or equal to" + " zero"
                    ),
                )
            )
        if not _is_callable(self._rho):
            control_dependencies.append(
                tf.compat.v1.debugging.assert_greater(
                    self._rho,
                    tf.constant(-1, dtype=self._dtype),
                    message="Correlation coefficient `rho` must be in (-1, 1)",
                )
            )
            control_dependencies.append(
                tf.compat.v1.debugging.assert_less(
                    self._rho,
                    tf.constant(1, dtype=self._dtype),
                    message="Correlation coefficient `rho` must be in (-1, 1)",
                )
            )
        control_dependencies.append(
            tf.compat.v1.debugging.assert_greater_equal(
                self._psi_threshold,
                tf.constant(1, dtype=self._dtype),
                message="Psi threshold must be in [1, 2]",
            )
        )
        control_dependencies.append(
            tf.compat.v1.debugging.assert_less_equal(
                self._psi_threshold,
                tf.constant(2, dtype=self._dtype),
                message="Psi threshold must be in [1, 2]",
            )
        )
        self.control_dependencies = control_dependencies

        super(SabrModel, self).__init__(2, drift_fn, _vol_fn, self._dtype, self._name)

    def sample_paths(
        self,
        initial_forward: types.RealTensor,
        initial_volatility: types.RealTensor,
        times: types.RealTensor,
        time_step: types.RealTensor,
        num_samples: types.RealTensor = 1,
        random_type: Optional[random.RandomType] = None,
        seed: Optional[types.IntTensor] = None,
        skip: types.IntTensor = 0,
        validate_args: bool = False,
        precompute_normal_draws: bool = True,
        name: Optional[str] = None,
    ):
        """Returns a sample of paths from the process.

        Generates samples of paths from the process at the specified time points.

        Currently only supports absorbing boundary conditions.

        Args:
          initial_forward: Initial value of the forward. A scalar real tensor.
          initial_volatility: Initial value of the volatilities. A scalar real
            tensor.
          times: The times at which the path points are to be evaluated. Rank 1
            `Tensor` of positive real values. This `Tensor` should be sorted in
            ascending order.
          time_step: Positive Python float or a scalar `Tensor `to denote time
            discretization parameter.
          num_samples: Positive scalar `int`. The number of paths to draw.
          random_type: Enum value of `RandomType`. The type of (quasi)-random number
            generator to use to generate the paths.
            Default value: None which maps to the standard pseudo-random numbers.
          seed: Seed for the random number generator. The seed is
            only relevant if `random_type` is one of
            `[STATELESS, PSEUDO, HALTON_RANDOMIZED, PSEUDO_ANTITHETIC,
              STATELESS_ANTITHETIC]`. For `PSEUDO`, `PSEUDO_ANTITHETIC` and
            `HALTON_RANDOMIZED` the seed should be an Python integer. For
            `STATELESS` and  `STATELESS_ANTITHETIC` must be supplied as an integer
            `Tensor` of shape `[2]`.
            Default value: `None` which means no seed is set.
          skip: `int32` 0-d `Tensor`. The number of initial points of the Sobol or
            Halton sequence to skip. Used only when `random_type` is 'SOBOL',
            'HALTON', or 'HALTON_RANDOMIZED', otherwise ignored.
            Default value: `0`.
          validate_args: Python `bool`. When `True`, input `Tensor's` are checked
            for validity despite possibly degrading runtime performance. The checks
            verify that `times` is strictly increasing. When `False` invalid inputs
            may silently render incorrect outputs. Default value: False.
          precompute_normal_draws: Python bool. Indicates whether the noise
            increments are precomputed upfront (see `models.euler_sampling.sample`).
            For `HALTON` and `SOBOL` random types the increments are always
            precomputed. While the resulting graph consumes more memory, the
            performance gains might be significant. Default value: `True`.
          name: str. The name to give this op. If not supplied, default name of
            `sample_paths` is used.

        Returns:
          A `Tensor`s of shape [num_samples, k, 2] where `k` is the size of the
          `times`.  The first values in `Tensor` are the simulated forward `F(t)`,
          whereas the second values in `Tensor` are the simulated volatility
          trajectories `V(t)`.
        """
        name = name or self._name + "_sample_path"
        with tf.name_scope(name):
            initial_forward = tf.convert_to_tensor(
                initial_forward, self._dtype, name="initial_forward"
            )
            initial_volatility = tf.convert_to_tensor(
                initial_volatility, self._dtype, name="initial_volatility"
            )
            times = tf.convert_to_tensor(times, self._dtype, name="times")
            time_step = tf.convert_to_tensor(time_step, self._dtype, name="time_step")

            if validate_args:
                self.control_dependencies.append(
                    tf.compat.v1.debugging.Assert(
                        tf.compat.v1.debugging.is_strictly_increasing(times), [times]
                    )
                )

            with tf.compat.v1.control_dependencies(self.control_dependencies):
                initial_forward += self._shift
                if self._enable_unbiased_sampling and not (
                    _is_callable(self._beta)
                    or _is_callable(self._volvol)
                    or _is_callable(self._rho)
                    or self._beta == 1
                ):
                    paths = self._sabr_sample_paths(
                        initial_forward,
                        initial_volatility,
                        times,
                        time_step,
                        num_samples,
                        random_type,
                        seed,
                        precompute_normal_draws,
                        skip=skip,
                    )
                else:
                    paths = super(SabrModel, self).sample_paths(
                        times,
                        num_samples,
                        [initial_forward, initial_volatility],
                        random_type,
                        seed,
                        skip=skip,
                        time_step=time_step,
                        precompute_normal_draws=precompute_normal_draws,
                    )

                forwards = tf.expand_dims(paths[:, :, 0] - self._shift, axis=-1)
                volatilities = tf.expand_dims(paths[:, :, 1], axis=-1)

                return tf.concat([forwards, volatilities], axis=-1)

    def _sabr_sample_paths(
        self,
        initial_forward,
        initial_volatility,
        times,
        time_step,
        num_samples,
        random_type,
        seed,
        precompute_normal_draws,
        skip,
    ):
        """Returns a sample of paths from the process."""
        num_requested_times = tff_utils.get_shape(times)[0]
        # Prepare results format
        forward = tf.zeros(shape=(num_samples,), dtype=self._dtype) + initial_forward
        vol = tf.zeros(shape=(num_samples,), dtype=self._dtype) + initial_volatility
        if isinstance(num_requested_times, int) and num_requested_times == 1:
            record_samples = False
            forward_paths = forward
            vol_paths = vol
        else:
            # If more than one sample has to be recorded, create a TensorArray
            record_samples = True
            element_shape = forward.shape
            forward_paths = tf.TensorArray(
                dtype=times.dtype,
                size=num_requested_times,
                element_shape=element_shape,
                clear_after_read=False,
            )
            vol_paths = tf.TensorArray(
                dtype=times.dtype,
                size=num_requested_times,
                element_shape=element_shape,
                clear_after_read=False,
            )

        # Define sampling while_loop body function
        def cond_fn(index, *args):
            return index < tf.size(times)

        # In order to use low-discrepancy random_type we need to generate the
        # sequence of independent random normals upfront. We also precompute random
        # numbers for stateless random type in order to ensure independent samples
        # for multiple function calls with different seeds.
        if precompute_normal_draws or random_type in (
            random.RandomType.SOBOL,
            random.RandomType.HALTON,
            random.RandomType.HALTON_RANDOMIZED,
            random.RandomType.STATELESS,
            random.RandomType.STATELESS_ANTITHETIC,
        ):
            num_time_steps = (
                tf.cast(
                    tf.math.ceil(tf.math.divide(times[-1], time_step)), dtype=tf.int32
                )
                + times.shape[0]
            )
            # We need a [3] + initial_forward.shape tensor of random draws.
            # This will be accessed by normal_draws_index.
            num_normal_draws = 3 * tf.size(initial_forward)
            normal_draws = utils.generate_mc_normal_draws(
                num_normal_draws=num_normal_draws,
                num_time_steps=num_time_steps,
                num_sample_paths=num_samples,
                random_type=random_type,
                seed=seed,
                skip=skip,
                dtype=self._dtype,
            )
        else:
            normal_draws = None

        def body_fn(
            index,
            current_time,
            forward,
            vol,
            forward_paths,
            vol_paths,
            normal_draws_index,
        ):
            """Simulate Sabr process to the next time point."""
            forward, vol, normal_draws_index = self._propagate_to_time(
                forward,
                vol,
                current_time,
                times[index],
                time_step,
                random_type,
                seed,
                normal_draws,
                normal_draws_index,
                num_time_steps,
            )
            # Always update paths in outer loop.
            if record_samples:
                # Update volatility paths
                vol_paths = vol_paths.write(index, vol)
                # Update forward paths
                forward_paths = forward_paths.write(index, forward)
            else:
                vol_paths = vol
                forward_paths = forward
            return (
                index + 1,
                times[index],
                forward,
                vol,
                forward_paths,
                vol_paths,
                normal_draws_index,
            )

        start_time = tf.constant(0, dtype=self._dtype)
        # Sample paths
        _, _, _, _, forward_paths, vol_paths, _ = tf.while_loop(
            cond_fn,
            body_fn,
            (0, start_time, forward, vol, forward_paths, vol_paths, 0),
            maximum_iterations=tf.size(times),
        )
        if not record_samples:
            # shape [num_samples, 1]
            vol_paths = tf.expand_dims(vol_paths, axis=-1)
            forward_paths = tf.expand_dims(forward_paths, axis=-1)
            # shape [num_samples, 1, 1]
            return tf.stack([forward_paths, vol_paths], -1)
        # Shape [num_time_points] + [num_samples]
        vol_paths = vol_paths.stack()
        forward_paths = forward_paths.stack()
        # transpose to shape [num_samples, num_time_points]
        vol_paths = tf.transpose(vol_paths)
        forward_paths = tf.transpose(forward_paths)
        # Shape [num_samples, num_time_points, 2]
        return tf.stack([forward_paths, vol_paths], -1)

    def _propagate_to_time(
        self,
        start_forward,
        start_vol,
        start_time,
        end_time,
        time_step,
        random_type,
        seed,
        normal_draws,
        normal_draws_index,
        num_time_steps,
    ):
        def cond_fn(t, *args):
            return t < end_time

        def body_fn(current_time, forward, vol, normal_draws_index):
            """Simulate Sabr process for one time step."""
            if normal_draws is not None:
                random_numbers = normal_draws[normal_draws_index]
                random_numbers = tf.reshape(random_numbers, [3] + forward.shape)
            else:
                random_numbers = random.mv_normal_sample(
                    [3] + forward.shape,
                    mean=tf.constant([0.0], dtype=self._dtype),
                    random_type=random_type,
                    seed=seed,
                )
                random_numbers = tf.squeeze(random_numbers, -1)
            dwv = random_numbers[0]
            uniforms = 0.5 * (1 + tf.math.erf(random_numbers[1]))
            z = random_numbers[2]

            time_to_end = end_time - current_time
            dt = tf.where(time_to_end <= time_step, time_to_end, time_step)

            next_vol = self._sample_next_volatilities(vol, dt, dwv)
            iv = self._sample_integrated_variance(vol, next_vol, dt)
            next_forward = self._sample_forwards(
                forward, vol, next_vol, iv, uniforms, z
            )

            return current_time + dt, next_forward, next_vol, normal_draws_index + 1

        _, next_forward, next_vol, normal_draws_index = tf.while_loop(
            cond_fn,
            body_fn,
            (start_time, start_forward, start_vol, normal_draws_index),
            maximum_iterations=num_time_steps,
        )
        return next_forward, next_vol, normal_draws_index

    def _sample_next_volatilities(self, vol, dt, dwv):
        return vol * tf.exp(
            self._volvol * dwv * tf.sqrt(dt) - self._volvol**2 * dt * 0.5
        )

    def _sample_integrated_variance(self, vol, next_vol, dt):
        return (1.0 - self._rho**2) * dt * (vol**2 + next_vol**2) / 2.0

    def _sample_forwards(self, forward, vol, next_vol, iv, uniforms, z):
        # See the "Direct inversion scheme" in section 3.4 in Reference [1].
        a = (
            1.0
            / iv
            * (
                forward ** (1.0 - self._beta) / (1.0 - self._beta)
                + self._rho / self._volvol * (next_vol - vol)
            )
            ** 2
        )
        b = 2.0 - (1.0 - 2 * self._beta - self._rho**2 * (1.0 - self._beta)) / (
            (1.0 - self._beta) * (1 - self._rho**2)
        )
        # Broadcast a to same shape as forward.
        b += tf.zeros_like(forward)

        absorption_prob = 1.0 - tf.math.igamma(b / 2.0, a / 2.0)
        should_be_zero = (self._beta != 0) & (
            (forward <= 0.0) | (absorption_prob > uniforms)
        )

        # See equation 3.9 in reference [1].
        k = 2.0 - b
        s2 = 2 * (k + 2 * a)
        m = k + a
        psi = s2 / m**2
        e2 = (2.0 / psi) - 1.0 + tf.sqrt(2.0 / psi) * tf.sqrt((2.0 / psi) - 1.0)
        e = tf.sqrt(e2)
        d = m / (1.0 + e2)
        next_forward_cond_1 = ((1.0 - self._beta) ** 2 * iv * d * (e + z) ** 2) ** (
            0.5 / (1.0 - self._beta)
        )

        c_star = self._root_chi2(a, b, uniforms)
        next_forward_cond_2 = (c_star * (1 - self._beta) ** 2 * iv) ** (
            1 / (2 - 2 * self._beta)
        )

        next_forward = tf.compat.v2.where(
            (m >= 0) & (psi <= self._psi_threshold),
            next_forward_cond_1,
            next_forward_cond_2,
        )

        return tf.compat.v2.where(should_be_zero, tf.zeros_like(forward), next_forward)

    def _root_chi2(self, a, b, uniforms):
        c_init = a

        def equation(c_star):
            p, dpc = ncx2cdf_and_gradient(a, b, c_star, self._ncx2_cdf_truncation)
            return 1 - p - uniforms, -dpc

        result, _, _ = root_finder(equation, c_init)
        return tf.math.maximum(result, 0)


def ncx2cdf_and_gradient(x, k, l, truncation=10):
    """Returns the CDF of noncentral X2 distribution and its gradient over l.

    Args:
      x: Values of the random variable following a noncentral X2 distribution. A
        real `Tensor`.
      k: Degrees of freedom. A positive real `Tensor` of same shape as `x`.
      l: Non-centrality parameter. A positive real `Tensor` of same shape as `x`.
      truncation: A positive integer. When computing the CDF of a noncentral X2
        distribution, it needs to calculate the sum of an expression from 0 to
        infinity. In practice, it needs to be truncated to compute an approximate
        value. This argument is the index of the last
          term that will be included in the sum. Default value: 10.

    Returns:
      A tuple of two `Tensor`s. The first `Tensor` is the CDF. The second
      `Tensor` is the gradient of the CDF over l. Both of the `Tensors` are of
      same shape as `x`.
    """
    g = 0.0
    dg = 0.0
    factorial = 1.0
    for j in range(truncation + 1):
        factorial *= j if j > 0 else 1
        h = (1 - tf.math.igammac((k + 2 * j) / 2.0, x / 2.0)) / factorial
        g += h * (l * 0.5) ** j
        dg += h * 0.5 * j * (l * 0.5) ** (j - 1)
    f = tf.math.exp(-0.5 * l)
    df = -0.5 * f
    p = f * g
    dp = df * g + f * dg
    return p, dp


def _is_callable(var_or_fn):
    """Returns whether an object is callable or not."""
    # Python 2.7 as well as Python 3.x with x > 2 support 'callable'.
    # In between, callable was removed hence we need to do a more expansive check
    if hasattr(var_or_fn, "__call__"):
        return True
    try:
        return callable(var_or_fn)
    except NameError:
        return False
