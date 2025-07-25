<!--
This file is generated by a tool. Do not edit directly.
For open-source contributions the docs will be updated automatically.
-->

*Last updated: 2023-03-16.*

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.rates.analytics.swap.equity_swap_price" />
<meta itemprop="path" content="Stable" />
</div>

# tf_quant_finance.rates.analytics.swap.equity_swap_price

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/paolodelia99/tf-quant-finance/blob/main/tf_quant_finance/rates/analytics/swap.py">View source</a>



Computes prices of a batch of equity swaps.

```python
tf_quant_finance.rates.analytics.swap.equity_swap_price(
    rate_leg_coupon_rates, equity_leg_forward_prices, equity_leg_spots,
    rate_leg_notional, equity_leg_notional, rate_leg_daycount_fractions,
    rate_leg_discount_factors, equity_leg_discount_factors, equity_dividends=None,
    is_equity_receiver=None, dtype=None, name=None
)
```



<!-- Placeholder for "Used in" -->

The swap consists of an equity and interest rate legs.

#### Example
```python
rate_leg_coupon_rates = [[0.1, 0.2, 0.05], [0.1, 0.05, 0.2]]
# Two cashflows of 4 and 3 payments, respectively
forward_prices = [[110, 120, 140, 150], [210, 220, 240, 0]]
spots = [100, 200]
notional = 1000
pay_leg_daycount_fractions = 0.5
rate_leg_daycount_fractions = [[0.5, 0.5, 0.5], [0.4, 0.5, 0.6]]
rate_leg_discount_factors = [[0.95, 0.9, 0.85], [0.98, 0.92, 0.88]]
equity_leg_discount_factors = [[0.95, 0.9, 0.85, 0.8],
                               [0.98, 0.92, 0.88, 0.0]]

equity_swap_price(
    rate_leg_coupon_rates=rate_leg_coupon_rates,
    equity_leg_forward_prices=forward_prices,
    equity_leg_spots=spots,
    rate_leg_notional=notional,
    equity_leg_notional=notional,
    rate_leg_daycount_fractions=rate_leg_daycount_fractions,
    rate_leg_discount_factors=rate_leg_discount_factors,
    equity_leg_discount_factors=equity_leg_discount_factors,
    is_equity_receiver=[True, False],
    dtype=tf.float64)
# Expected: [216.87770563, -5.00952381]
forward_rates(df_start_dates, df_end_dates, daycount_fractions,
              dtype=tf.float64)
```

#### Args:


* <b>`rate_leg_coupon_rates`</b>: A real `Tensor` of shape
  `batch_shape + [num_rate_cashflows]`, where `num_rate_cashflows` is the
  number of cashflows for each batch element. Coupon rates for the
  interest rate leg.
* <b>`equity_leg_forward_prices`</b>: A `Tensor` of the same `dtype` as
  `rate_leg_coupon_rates` and of shape
  `batch_shape + [num_equity_cashflows]`, where `num_equity_cashflows` is
  the number of cashflows for each batch element. Equity leg forward
  prices.
* <b>`equity_leg_spots`</b>: A `Tensor` of the same `dtype` as
  `equity_leg_forward_prices` and of shape compatible with `batch_shape`.
  Spot prices for each batch element of the equity leg.
* <b>`rate_leg_notional`</b>: A `Tensor` of the same `dtype` as `rate_leg_coupon_rates`
  and of compatible shape. Notional amount for each cashflow.
* <b>`equity_leg_notional`</b>: A `Tensor` of the same `dtype` as
  `equity_leg_forward_prices` and of compatible shape.  Notional amount for
  each cashflow.
* <b>`rate_leg_daycount_fractions`</b>: A `Tensor` of the same `dtype` as
  `rate_leg_coupon_rates` and of compatible shape.  Year fractions for the
  coupon accrual.
* <b>`rate_leg_discount_factors`</b>: A `Tensor` of the same `dtype` as
  `rate_leg_coupon_rates` and of compatible shape. Discount factors for each
  cashflow of the rate leg.
* <b>`equity_leg_discount_factors`</b>: A `Tensor` of the same `dtype` as
  `equity_leg_forward_prices` and of compatible shape. Discount factors for
  each cashflow of the equity leg.
* <b>`equity_dividends`</b>: A `Tensor` of the same `dtype` as
  `equity_leg_forward_prices` and of compatible shape. Dividends paid at the
  leg reset times.
  Default value: None which maps to zero dividend.
* <b>`is_equity_receiver`</b>: A boolean `Tensor` of shape compatible with
  `batch_shape`. Indicates whether the swap holder is equity holder or
  receiver.
  Default value: None which means that all swaps are equity reiver swaps.
* <b>`dtype`</b>: `tf.Dtype`. If supplied the dtype for the input and output `Tensor`s.
  Default value: None which maps to the default dtype inferred from
  `rate_leg_coupon_rates`.
* <b>`name`</b>: Python str. The name to give to the ops created by this function.
  Default value: None which maps to 'equity_swap_price'.


#### Returns:

A `Tensor` of the same `dtype` as `rate_leg_coupon_rates` and of shape
`batch_shape`. Present values of the equity swaps.
