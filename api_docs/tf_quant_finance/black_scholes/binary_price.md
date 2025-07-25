<!--
This file is generated by a tool. Do not edit directly.
For open-source contributions the docs will be updated automatically.
-->

*Last updated: 2023-03-16.*

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.black_scholes.binary_price" />
<meta itemprop="path" content="Stable" />
</div>

# tf_quant_finance.black_scholes.binary_price

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/paolodelia99/tf-quant-finance/blob/main/tf_quant_finance/black_scholes/vanilla_prices.py">View source</a>



Computes the Black Scholes price for a batch of binary call or put options.

```python
tf_quant_finance.black_scholes.binary_price(
    *, volatilities, strikes, expiries, spots=None, forwards=None,
    discount_rates=None, dividend_rates=None, discount_factors=None,
    is_call_options=None, is_normal_volatility=False, dtype=None, name=None
)
```



<!-- Placeholder for "Used in" -->

The binary call (resp. put) option priced here is that which pays off a unit
of cash if the underlying asset has a value greater (resp. smaller) than the
strike price at expiry. Hence the binary option price is the discounted
probability that the asset will end up higher (resp. lower) than the
strike price at expiry.

#### Example

```python
  # Price a batch of 5 binary call options.
  volatilities = np.array([0.0001, 102.0, 2.0, 0.1, 0.4])
  forwards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
  # Strikes will automatically be broadcasted to shape [5].
  strikes = np.array([3.0])
  # Expiries will be broadcast to shape [5], i.e. each option has strike=3
  # and expiry = 1.
  expiries = 1.0
  computed_prices = tff.black_scholes.binary_price(
      volatilities=volatilities,
      strikes=strikes,
      expiries=expiries,
      forwards=forwards)
# Expected print output of prices:
# [0.         0.         0.15865525 0.99764937 0.85927418]
```

#### References:

[1] Hull, John C., Options, Futures and Other Derivatives. Pearson, 2018.
[2] Wikipedia contributors. Binary option. Available at:
https://en.wikipedia.org/w/index.php?title=Binary_option

#### Args:


* <b>`volatilities`</b>: Real `Tensor` of any shape and dtype. The volatilities to
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
  `volatilities` and of the shape that broadcasts with `volatilities`.
  Default value: `None`, equivalent to q = 0.
* <b>`discount_factors`</b>: An optional real `Tensor` of same dtype as the
  `volatilities`. If not None, these are the discount factors to expiry
  (i.e. e^(-rT)). If None, no discounting is applied (i.e. the undiscounted
  option price is returned). If `spots` is supplied and `discount_factors`
  is not None then this is also used to compute the forwards to expiry.
  Default value: None, equivalent to discount factors = 1.
* <b>`is_call_options`</b>: A boolean `Tensor` of a shape compatible with
  `volatilities`. Indicates whether the option is a call (if True) or a put
  (if False). If not supplied, call options are assumed.
* <b>`is_normal_volatility`</b>: An optional Python boolean specifying whether the
  `volatilities` correspond to lognormal Black volatility (if False) or
  normal Black volatility (if True).
  Default value: False, which corresponds to lognormal volatility.
* <b>`dtype`</b>: Optional `tf.DType`. If supplied, the dtype to be used for conversion
  of any supplied non-`Tensor` arguments to `Tensor`.
  Default value: None which maps to the default dtype inferred by TensorFlow
    (float32).
* <b>`name`</b>: str. The name for the ops created by this function.
  Default value: None which is mapped to the default name `binary_price`.


#### Returns:


* <b>`binary_prices`</b>: A `Tensor` of the same shape as `forwards`. The Black
Scholes price of the binary options.


#### Raises:


* <b>`ValueError`</b>: If both `forwards` and `spots` are supplied or if neither is
  supplied.
* <b>`ValueError`</b>: If both `discount_rates` and `discount_factors` is supplied.