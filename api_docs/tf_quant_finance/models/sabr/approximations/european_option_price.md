<!--
This file is generated by a tool. Do not edit directly.
For open-source contributions the docs will be updated automatically.
-->

*Last updated: 2023-03-16.*

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.models.sabr.approximations.european_option_price" />
<meta itemprop="path" content="Stable" />
</div>

# tf_quant_finance.models.sabr.approximations.european_option_price

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/paolodelia99/tf-quant-finance/blob/main/tf_quant_finance/models/sabr/approximations/european_options.py">View source</a>



Computes the approximate European option price under the SABR model.

```python
tf_quant_finance.models.sabr.approximations.european_option_price(
    *, strikes, expiries, forwards, is_call_options, alpha, beta, volvol, rho,
    shift=0.0, volatility_type=tf_quant_finance.models.sabr.approximations.SabrImpli
    edVolatilityType.LOGNORMAL, approximation_type=tf_quant_finance.models.sabr.appr
    oximations.SabrApproximationType.HAGAN, dtype=None, name=None
)
```



<!-- Placeholder for "Used in" -->

For a review of the SABR model and the conventions used, please see the
docstring for `implied_volatility`.

#### Example
```python
import tf_quant_finance as tff
import tensorflow as tf

prices = tff.models.sabr.approximations.european_option_price(
  strikes=np.array([90.0, 100.0]),
  expiries=np.array([0.5, 1.0]),
  forwards=np.array([100.0, 110.0]),
  is_call_options=np.array([True, False]),
  alpha=3.2,
  beta=0.2,
  volvol=1.4,
  rho=0.0005,
  dtype=tf.float64)

# Expected: [10.41244961, 1.47123225]

```

#### Args:


* <b>`strikes`</b>: Real `Tensor` of arbitrary shape, specifying the strike prices.
  Values must be strictly positive.
* <b>`expiries`</b>: Real `Tensor` of shape compatible with that of `strikes`,
  specifying the corresponding time-to-expiries of the options. Values must
  be strictly positive.
* <b>`forwards`</b>: Real `Tensor` of shape compatible with that of `strikes`,
  specifying the observed forward prices of the underlying. Values must be
  strictly positive.
* <b>`is_call_options`</b>: Boolean `Tensor` of shape compatible with that of
  `forward`, indicating whether the option is a call option (true) or put
  option (false).
* <b>`alpha`</b>: Real `Tensor` of shape compatible with that of `strikes`, specifying
  the initial values of the stochastic volatility. Values must be strictly
  positive.
* <b>`beta`</b>: Real `Tensor` of shape compatible with that of `strikes`, specifying
  the model exponent `beta`. Values must satisfy 0 <= `beta` <= 1.
* <b>`volvol`</b>: Real `Tensor` of shape compatible with that of `strikes`,
  specifying the model vol-vol multipliers. Values must satisfy
  `0 <= volvol`.
* <b>`rho`</b>: Real `Tensor` of shape compatible with that of `strikes`, specifying
  the correlation factors between the Wiener processes modeling the forward
  and the volatility. Values must satisfy -1 < `rho` < 1.
* <b>`shift`</b>: Optional `Tensor` of shape compatible with that of `strkies`,
  specifying the shift parameter(s). In the shifted model, the process
  modeling the forward is modified as: dF = sigma * (F + shift) ^ beta * dW.
  With this modification, negative forward rates are valid as long as
  F > -shift.
  Default value: 0.0
* <b>`volatility_type`</b>: Either SabrImpliedVolatility.NORMAL or LOGNORMAL.
  Default value: `LOGNORMAL`.
* <b>`approximation_type`</b>: Instance of `SabrApproxmationScheme`.
  Default value: `HAGAN`.
* <b>`dtype`</b>: Optional: `tf.DType`. If supplied, the dtype to be used for
  converting values to `Tensor`s.
  Default value: `None`, which means that the default dtypes inferred from
    `strikes` is used.
* <b>`name`</b>: str. The name for the ops created by this function.
  Default value: 'sabr_approx_eu_option_price'.


#### Returns:

A real `Tensor` of the same shape as `strikes`, containing the
corresponding options price.
