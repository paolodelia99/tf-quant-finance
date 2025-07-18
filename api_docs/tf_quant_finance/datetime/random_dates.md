<!--
This file is generated by a tool. Do not edit directly.
For open-source contributions the docs will be updated automatically.
-->

*Last updated: 2023-03-16.*

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.datetime.random_dates" />
<meta itemprop="path" content="Stable" />
</div>

# tf_quant_finance.datetime.random_dates

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/paolodelia99/tf-quant-finance/blob/main/tf_quant_finance/datetime/date_tensor.py">View source</a>



Generates random dates between the supplied start and end dates.

```python
tf_quant_finance.datetime.random_dates(
    *, start_date, end_date, size=1, seed=None, name=None
)
```



<!-- Placeholder for "Used in" -->

Generates specified number of random dates between the given start and end
dates. The start and end dates are supplied as `DateTensor` objects. The dates
uniformly distributed between the start date (inclusive) and end date
(exclusive). Note that the dates are uniformly distributed over the calendar
range, i.e. no holiday calendar is taken into account.

#### Args:


* <b>`start_date`</b>: DateTensor of arbitrary shape. The start dates of the range from
  which to sample. The start dates are themselves included in the range.
* <b>`end_date`</b>: DateTensor of shape compatible with the `start_date`. The end date
  of the range from which to sample. The end dates are excluded from the
  range.
* <b>`size`</b>: Positive scalar int32 Tensor. The number of dates to draw between the
  start and end date.
  Default value: 1.
* <b>`seed`</b>: Optional seed for the random generation.
* <b>`name`</b>: Optional str. The name to give to the ops created by this function.
  Default value: 'random_dates'.


#### Returns:

A DateTensor of shape [size] + dates_shape where dates_shape is the common
broadcast shape for (start_date, end_date).


#### Example

```python
# Note that the start and end dates need to be of broadcastable shape (though
# not necessarily the same shape).
# In this example, the start dates are of shape [2] and the end dates are
# of a compatible but non-identical shape [1].
start_dates = tff.datetime.dates_from_tuples([
  (2020, 5, 16),
  (2020, 6, 13)
])
end_dates = tff.datetime.dates_from_tuples([(2021, 5, 21)])
size = 3  # Generate 3 dates for each pair of (start, end date).
sample = tff.datetime.random_dates(start_date=start_dates, end_date=end_dates,
                            size=size)
# sample is a DateTensor of shape [3, 2]. The [3] is from the size and [2] is
# the common broadcast shape of start and end date.
```