<!--
This file is generated by a tool. Do not edit directly.
For open-source contributions the docs will be updated automatically.
-->

*Last updated: 2023-03-16.*

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.datetime.daycount_actual_365_fixed" />
<meta itemprop="path" content="Stable" />
</div>

# tf_quant_finance.datetime.daycount_actual_365_fixed

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/paolodelia99/tf-quant-finance/blob/main/tf_quant_finance/datetime/daycounts.py">View source</a>



Computes the year fraction between the specified dates.

```python
tf_quant_finance.datetime.daycount_actual_365_fixed(
    *, start_date, end_date, schedule_info=None, dtype=None, name=None
)
```



<!-- Placeholder for "Used in" -->

The actual/365 convention specifies the year fraction between the start and
end date as the actual number of days between the two dates divided by 365.

Note that the schedule info is not needed for this convention and is ignored
if supplied.

#### For more details see:


https://en.wikipedia.org/wiki/Day_count_convention#Actual/365_Fixed

#### Args:


* <b>`start_date`</b>: A `DateTensor` object of any shape.
* <b>`end_date`</b>: A `DateTensor` object of compatible shape with `start_date`.
* <b>`schedule_info`</b>: The schedule info. Ignored for this convention.
* <b>`dtype`</b>: The dtype of the result. Either `tf.float32` or `tf.float64`. If not
  supplied, `tf.float32` is returned.
* <b>`name`</b>: Python `str` name prefixed to ops created by this function. If not
  supplied, `actual_365_fixed` is used.


#### Returns:

A real `Tensor` of supplied `dtype` and shape of `start_date`. The year
fraction between the start and end date as computed by Actual/365 fixed
convention.
