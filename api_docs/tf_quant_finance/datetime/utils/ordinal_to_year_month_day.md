<!--
This file is generated by a tool. Do not edit directly.
For open-source contributions the docs will be updated automatically.
-->

*Last updated: 2023-03-16.*

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.datetime.utils.ordinal_to_year_month_day" />
<meta itemprop="path" content="Stable" />
</div>

# tf_quant_finance.datetime.utils.ordinal_to_year_month_day

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/paolodelia99/tf-quant-finance/blob/main/tf_quant_finance/datetime/date_utils.py">View source</a>



Calculates years, months and dates Tensor given ordinals Tensor.

```python
tf_quant_finance.datetime.utils.ordinal_to_year_month_day(
    ordinals
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`ordinals`</b>: Tensor of int32 type. Each element is number of days since 1 Jan
 0001. 1 Jan 0001 has `ordinal = 1`.


#### Returns:

Tuple (years, months, days), each element is an int32 Tensor of the same
shape as `ordinals`. `months` and `days` are one-based.
