<!--
This file is generated by a tool. Do not edit directly.
For open-source contributions the docs will be updated automatically.
-->

*Last updated: 2023-03-16.*

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.math.qmc.utils.exp2" />
<meta itemprop="path" content="Stable" />
</div>

# tf_quant_finance.math.qmc.utils.exp2

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/paolodelia99/tf-quant-finance/blob/main/tf_quant_finance/math/qmc/utils.py">View source</a>



Returns the point-wise base-2 exponentiation of a given `Tensor`.

```python
tf_quant_finance.math.qmc.utils.exp2(
    value
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`value`</b>: Positive scalar `Tensor` of integers.

#### Examples

```python
import tensorflow as tf
import tf_quant_finance as tff

# Example: Computing the base-2 exponentiation of a range.

tff.math.qmc.utils.exp2(tf.range(0, 5))
# ==> tf.Tensor([1, 2, 4, 8, 16], shape=(5,), dtype=int32)
```

#### Returns:

`Tensor` with the same `shape` and `dtype` as `value` equal to `1 << value`.
