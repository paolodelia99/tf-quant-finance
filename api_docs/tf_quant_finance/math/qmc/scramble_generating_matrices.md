<!--
This file is generated by a tool. Do not edit directly.
For open-source contributions the docs will be updated automatically.
-->

*Last updated: 2023-03-16.*

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.math.qmc.scramble_generating_matrices" />
<meta itemprop="path" content="Stable" />
</div>

# tf_quant_finance.math.qmc.scramble_generating_matrices

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/paolodelia99/tf-quant-finance/blob/main/tf_quant_finance/math/qmc/digital_net.py">View source</a>



Scrambles a generating matrix.

```python
tf_quant_finance.math.qmc.scramble_generating_matrices(
    generating_matrices, scrambling_matrices, num_digits, validate_args=False,
    dtype=None, name=None
)
```



<!-- Placeholder for "Used in" -->

#### Examples

```python
import tf_quant_finance as tff

# Example: Scrambling the 2D Sobol generating matrices.

dim = 2
num_results = 1000
num_digits = 10
seed = (2, 3)

tff.math.qmc.scramble_generating_matrices(
    tff.math.qmc.sobol_generating_matrices(dim, num_results, num_digits),
    tff.math.qmc.random_scrambling_matrices(dim, num_digits, seed=seed),
    num_digits)
# ==> tf.Tensor([
#             [586, 505, 224, 102,  34,  31,  13,   6,   2,   1],
#             [872, 695, 945, 531, 852, 663, 898, 568, 875, 693],
#         ], shape=(2, 10), dtype=int32)
```

#### Args:


* <b>`generating_matrices`</b>: Positive scalar `Tensor` of integers.
* <b>`scrambling_matrices`</b>: Positive Scalar `Tensor` of integers with the same
  `shape` as `generating_matrices`.
* <b>`num_digits`</b>: Positive scalar `Tensor` of integers with rank 0. The base-2
  precision of the points which can be sampled from `generating_matrices`.
* <b>`validate_args`</b>: Python `bool` indicating whether to validate arguments.
  Default value: `False`.
* <b>`dtype`</b>: Optional `dtype`. The `dtype` of the output `Tensor` (either `int32`
  or `int64`).
  Default value: `None` which maps to `generating_matrices.dtype`.
* <b>`name`</b>: Python `str` name prefixed to ops created by this function.
  Default value: `None` which maps to `scramble_generating_matrices`.


#### Returns:

A `Tensor` with the same `shape` and `dtype` as `generating_matrices`.
