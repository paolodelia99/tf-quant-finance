<!--
This file is generated by a tool. Do not edit directly.
For open-source contributions the docs will be updated automatically.
-->

*Last updated: 2023-03-16.*

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.black_scholes.variance_swap_fair_strike" />
<meta itemprop="path" content="Stable" />
</div>

# tf_quant_finance.black_scholes.variance_swap_fair_strike

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/paolodelia99/tf-quant-finance/blob/main/tf_quant_finance/black_scholes/variance_swaps.py">View source</a>



Calculates the fair value strike for a variance swap contract.

```python
tf_quant_finance.black_scholes.variance_swap_fair_strike(
    put_strikes, put_volatilities, call_strikes, call_volatilities, expiries,
    discount_rates, spots, reference_strikes, validate_args=False, dtype=None,
    name=None
)
```



<!-- Placeholder for "Used in" -->

This implements the approach in Appendix A of Demeterfi et al (1999), where a
variance swap is defined as a forward contract on the square of annualized
realized volatility (though the approach assumes continuous sampling). The
variance swap payoff is, then:

`notional * (realized_volatility^2 - variance_strike)`

The method calculates the weight of each European option required to
approximately replicate such a payoff using the discrete range of strike
prices and implied volatilities of European options traded on the market. The
fair value `variance_strike` is that which is expected to produce zero payoff.

#### Example

```python
dtype = tf.float64
call_strikes = tf.constant([[100, 105, 110, 115], [1000, 1100, 1200, 1300]],
  dtype=dtype)
call_vols = 0.2 * tf.ones((2, 4), dtype=dtype)
put_strikes = tf.constant([[100, 95, 90, 85], [1000, 900, 800, 700]],
  dtype=dtype)
put_vols = 0.2 * tf.ones((2, 4), dtype=dtype)
reference_strikes = tf.constant([100.0, 1000.0], dtype=dtype)
expiries = tf.constant([0.25, 0.25], dtype=dtype)
discount_rates = tf.constant([0.05, 0.05], dtype=dtype)
variance_swap_price(
  put_strikes,
  put_vols,
  call_strikes,
  put_vols,
  expiries,
  discount_rates,
  reference_strikes,
  reference_strikes,
  dtype=tf.float64)
# [0.03825004, 0.04659269]
```

#### References

[1] Demeterfi, K., Derman, E., Kamal, M. and Zou, J., 1999. More Than You Ever
  Wanted To Know About Volatility Swaps. Goldman Sachs Quantitative Strategies
  Research Notes.

#### Args:


* <b>`put_strikes`</b>: A real `Tensor` of shape  `batch_shape + [num_put_strikes]`
  containing the strike values of traded puts. This must be supplied in
  **descending** order, and its elements should be less than or equal to the
  `reference_strike`.
* <b>`put_volatilities`</b>: A real `Tensor` of shape  `batch_shape +
  [num_put_strikes]` containing the market volatility for each strike in
  `put_strikes. The final value is unused.
* <b>`call_strikes`</b>: A real `Tensor` of shape  `batch_shape + [num_call_strikes]`
  containing the strike values of traded calls. This must be supplied in
  **ascending** order, and its elements should be greater than or equal to
  the `reference_strike`.
* <b>`call_volatilities`</b>: A real `Tensor` of shape  `batch_shape +
  [num_call_strikes]` containing the market volatility for each strike in
  `call_strikes`. The final value is unused.
* <b>`expiries`</b>: A real `Tensor` of shape compatible with `batch_shape` containing
  the time to expiries of the contracts.
* <b>`discount_rates`</b>: A real `Tensor` of shape compatible with `batch_shape`
  containing the discount rate to be applied.
* <b>`spots`</b>: A real `Tensor` of shape compatible with `batch_shape` containing the
  current spot price of the asset.
* <b>`reference_strikes`</b>: A real `Tensor` of shape compatible with `batch_shape`
  containing an arbitrary value demarcating the atm boundary between liquid
  calls and puts. Typically either the spot price or the (common) first
  value of `put_strikes` or `call_strikes`.
* <b>`validate_args`</b>: Python `bool`. When `True`, input `Tensor`s are checked for
  validity. The checks verify the the matching length of strikes and
  volatilties. When `False` invalid inputs may silently render incorrect
  outputs, yet runtime performance will be improved.
  Default value: False.
* <b>`dtype`</b>: `tf.Dtype`. If supplied the dtype for the input and output `Tensor`s.
  Default value: None, leading to the default value inferred by Tensorflow.
* <b>`name`</b>: Python str. The name to give to the ops created by this function.
  Default value: `None` which maps to 'variance_swap_price'.


#### Returns:

A `Tensor` of shape `batch_shape` containing the fair value of variance for
each item in the batch. Note this is on the decimal rather than square
percentage scale.
