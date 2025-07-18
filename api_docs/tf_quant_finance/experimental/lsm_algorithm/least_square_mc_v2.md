<!--
This file is generated by a tool. Do not edit directly.
For open-source contributions the docs will be updated automatically.
-->

*Last updated: 2023-03-16.*

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.experimental.lsm_algorithm.least_square_mc_v2" />
<meta itemprop="path" content="Stable" />
</div>

# tf_quant_finance.experimental.lsm_algorithm.least_square_mc_v2

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/paolodelia99/tf-quant-finance/blob/main/tf_quant_finance/experimental/lsm_algorithm/lsm_v2.py">View source</a>



Values Amercian style options using the LSM algorithm.

```python
tf_quant_finance.experimental.lsm_algorithm.least_square_mc_v2(
    sample_paths, exercise_times, payoff_fn, basis_fn, discount_factors=None,
    num_calibration_samples=None, dtype=None, name=None
)
```



<!-- Placeholder for "Used in" -->

The Least-Squares Monte-Carlo (LSM) algorithm is a Monte-Carlo approach to
valuation of American style options. Using the sample paths of underlying
assets, and a user supplied payoff function it attempts to find the optimal
exercise point along each sample path. With optimal exercise points known,
the option is valued as the average payoff assuming optimal exercise
discounted to present value.

#### Example. American put option price through Monte Carlo
```python
# Let the underlying model be a Black-Scholes process
# dS_t / S_t = rate dt + sigma**2 dW_t, S_0 = 1.0
# with `rate = 0.1`, and volatility `sigma = 1.0`.
# Define drift and volatility functions for log(S_t)
rate = 0.1
def drift_fn(_, x):
  return rate - tf.ones_like(x) / 2.
def vol_fn(_, x):
  return tf.expand_dims(tf.ones_like(x), axis=-1)
# Use Euler scheme to propagate 100000 paths for 1 year into the future
times = np.linspace(0., 1, num=50)
num_samples = 100000
log_paths = tf.function(tff.models.euler_sampling.sample)(
        dim=1,
        drift_fn=drift_fn, volatility_fn=vol_fn,
        random_type=tff.math.random.RandomType.PSEUDO_ANTITHETIC,
        times=times, num_samples=num_samples, seed=42, time_step=0.01)
# Compute exponent to get samples of `S_t`
paths = tf.math.exp(log_paths)
# American put option price for strike 1.1 and expiry 1 (assuming actual day
# count convention and no settlement adjustment)
strike = [1.1]
exercise_times = tf.range(times.shape[-1])
discount_factors = tf.exp(-rate * times)
payoff_fn = make_basket_put_payoff(strike)
basis_fn = make_polynomial_basis(10)
least_square_mc(paths, exercise_times, payoff_fn, basis_fn,
                discount_factors=discount_factors)
# Expected value: [0.397]
# European put option price
tff.black_scholes.option_price(volatilities=[1], strikes=strikes,
                               expiries=[1], spots=[1.],
                               discount_factors=discount_factors[-1],
                               is_call_options=False,
                               dtype=tf.float64)
# Expected value: [0.379]
```
#### References

[1] Longstaff, F.A. and Schwartz, E.S., 2001. Valuing American options by
simulation: a simple least-squares approach. The review of financial studies,
14(1), pp.113-147.

#### Args:


* <b>`sample_paths`</b>: A `Tensor` of either shape `[num_samples, num_times, dim]` or
  `[batch_size, num_samples, num_times, dim]`, the sample paths of the
  underlying ito process of dimension `dim` at `num_times` different points.
  The `batch_size` allows multiple options to be valued in parallel.
* <b>`exercise_times`</b>: An `int32` `Tensor` of shape `[num_exercise_times]`.
  Contents must be a subset of the integers `[0,...,num_times - 1]`,
  representing the time indices at which the option may be exercised.
* <b>`payoff_fn`</b>: Callable from a `Tensor` of shape `[num_samples, S, dim]`
  (where S <= num_times) to a `Tensor` of shape `[num_samples, batch_size]`
  of the same dtype as `samples`. The output represents the payout resulting
  from exercising the option at time `S`. The `batch_size` allows multiple
  options to be valued in parallel.
* <b>`basis_fn`</b>: Callable from a `Tensor` of the same shape and `dtype` as
  `sample_paths` and a positive integer `Tenor` (representing a current
  time index) to a `Tensor` of shape `[batch_size, basis_size, num_samples]`
  of the same dtype as `sample_paths`. The result being the design matrix
  used in regression of the continuation value of options.
* <b>`discount_factors`</b>: A `Tensor` of shape `[num_exercise_times]` or of rank 3
  and compatible with `[num_samples, batch_size, num_exercise_times]`.
  The `dtype` should be the same as of `samples`.
  Default value: `None` which maps to a one-`Tensor` of the same `dtype`
    as `samples` and shape `[num_exercise_times]`.
* <b>`num_calibration_samples`</b>: An optional integer less or equal to `num_samples`.
  The number of sampled trajectories used for the LSM regression step.
  Note that only the last`num_samples - num_calibration_samples` of the
  sampled paths are used to determine the price of the option.
  Default value: `None`, which means that all samples are used for
    regression and option pricing.
* <b>`dtype`</b>: Optional `dtype`. Either `tf.float32` or `tf.float64`. The `dtype`
  If supplied, represents the `dtype` for the input and output `Tensor`s.
  Default value: `None`, which means that the `dtype` inferred by TensorFlow
  is used.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
  Default value: `None` which is mapped to the default name
  'least_square_mc'.


#### Returns:

A `Tensor` of shape `[num_samples, batch_size]` of the same dtype as
`samples`.
