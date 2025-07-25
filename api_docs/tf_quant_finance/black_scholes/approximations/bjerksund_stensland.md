<!--
This file is generated by a tool. Do not edit directly.
For open-source contributions the docs will be updated automatically.
-->

*Last updated: 2023-03-16.*

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.black_scholes.approximations.bjerksund_stensland" />
<meta itemprop="path" content="Stable" />
</div>

# tf_quant_finance.black_scholes.approximations.bjerksund_stensland

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/paolodelia99/tf-quant-finance/blob/main/tf_quant_finance/black_scholes/approximations/american_option.py">View source</a>



Computes prices of a batch of American options using Bjerksund-Stensland.

```python
tf_quant_finance.black_scholes.approximations.bjerksund_stensland(
    *, volatilities, strikes, expiries, spots=None, forwards=None,
    discount_rates=None, dividend_rates=None, discount_factors=None,
    is_call_options=None, modified_boundary=True, dtype=None, name=None
)
```



<!-- Placeholder for "Used in" -->

#### Example

```python
  import tf_quant_finance as tff
  # Price a batch of 5 american call options.
  volatilities = [0.2, 0.2, 0.2, 0.2, 0.2]
  forwards = [80.0, 90.0, 100.0, 110.0, 120.0]
  # Strikes will automatically be broadcasted to shape [5].
  strikes = np.array([100.0])
  # Expiries will be broadcast to shape [5], i.e. each option has strike=100
  # and expiry = 0.25.
  expiries = 0.25
  discount_rates = 0.08
  dividend_rates = 0.12
  computed_prices = tff.black_scholes.approximations.bjerksund_stensland(
      volatilities=volatilities,
      strikes=strikes,
      expiries=expiries,
      discount_rates=discount_rates,
      dividend_rates=dividend_rates,
      forwards=forwards,
      is_call_options=True
      modified_boundary=True)
# Expected print output of computed prices:
# [ 0.03931201  0.70745419  4.01937524 11.31429842 21.20602005]
```

#### References:
[1] Bjerksund, P. and Stensland G., Closed Form Valuation of American Options,
    2002
    https://core.ac.uk/download/pdf/30824897.pdf

#### Args:


* <b>`volatilities`</b>: Real `Tensor` of any shape and real dtype. The volatilities to
  expiry of the options to price.
* <b>`strikes`</b>: A real `Tensor` of the same dtype and compatible shape as
  `volatilities`. The strikes of the options to be priced.
* <b>`expiries`</b>: A real `Tensor` of same dtype and compatible shape as
  `volatilities`. The expiry of each option. The units should be such that
  `expiry * volatility**2` is dimensionless.
* <b>`spots`</b>: A real `Tensor` of any shape that broadcasts to the shape of the
  `volatilities`. The current spot price of the underlying. Either this
  argument or the `forwards` (but not both) must be supplied.
* <b>`forwards`</b>: A real `Tensor` of any shape that broadcasts to the shape of
  `volatilities`. The forwards to maturity. Either this argument or the
  `spots` must be supplied but both must not be supplied.
* <b>`discount_rates`</b>: An optional real `Tensor` of same dtype as the
  `volatilities` and of the shape that broadcasts with `volatilities`.
  If not `None`, discount factors are calculated as e^(-rT),
  where r are the discount rates, or risk free rates. At most one of
  discount_rates and discount_factors can be supplied.
  Default value: `None`, equivalent to r = 0 and discount factors = 1 when
  discount_factors also not given.
* <b>`dividend_rates`</b>: An optional real `Tensor` of same dtype as the
  `volatilities`. The continuous dividend rate on the underliers. May be
  negative (to indicate costs of holding the underlier).
  Default value: `None`, equivalent to zero dividends.
* <b>`discount_factors`</b>: An optional real `Tensor` of same dtype as the
  `volatilities`. If not `None`, these are the discount factors to expiry
  (i.e. e^(-rT)). Mutually exclusive with discount_rate and cost_of_carry.
  If neither is given, no discounting is applied (i.e. the undiscounted
  option price is returned). If `spots` is supplied and `discount_factors`
  is not `None` then this is also used to compute the forwards to expiry.
  At most one of discount_rates and discount_factors can be supplied.
  Default value: `None`, which maps to e^(-rT) calculated from
  discount_rates.
* <b>`is_call_options`</b>: A boolean `Tensor` of a shape compatible with
  `volatilities`. Indicates whether the option is a call (if True) or a put
  (if False). If not supplied, call options are assumed.
* <b>`modified_boundary`</b>: Python `bool`. Indicates whether the Bjerksund-Stensland
  1993 algorithm (single boundary) if False or Bjerksund-Stensland 2002
  algorithm (modified boundary) if True, is to be used.
* <b>`dtype`</b>: Optional `tf.DType`. If supplied, the dtype to be used for conversion
  of any supplied non-`Tensor` arguments to `Tensor`.
  Default value: `None` which maps to the default dtype inferred by
    TensorFlow.
* <b>`name`</b>: str. The name for the ops created by this function.
  Default value: `None` which is mapped to the default name `option_price`.


#### Returns:

A `Tensor` of the same shape as `forwards`.



#### Raises:


* <b>`ValueError`</b>: If both `forwards` and `spots` are supplied or if neither is
  supplied.
* <b>`ValueError`</b>: If both `discount_rates` and `discount_factors` is supplied.