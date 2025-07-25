<!--
This file is generated by a tool. Do not edit directly.
For open-source contributions the docs will be updated automatically.
-->

*Last updated: 2023-03-16.*

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.math.qmc.digital_net_sample" />
<meta itemprop="path" content="Stable" />
</div>

# tf_quant_finance.math.qmc.digital_net_sample

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/paolodelia99/tf-quant-finance/blob/main/tf_quant_finance/math/qmc/digital_net.py">View source</a>



Constructs a digital net from a generating matrix.

```python
tf_quant_finance.math.qmc.digital_net_sample(
    generating_matrices, num_results, num_digits, sequence_indices=None,
    scrambling_matrices=None, digital_shift=None, apply_tent_transform=False,
    validate_args=False, dtype=None, name=None
)
```



<!-- Placeholder for "Used in" -->

#### Examples

```python
import tf_quant_finance as tff

# Example: Sampling 1,000 points from 2D Sobol generating matrices.

dim = 2
num_results = 1000
num_digits = 10

tff.math.qmc.digital_net_sample(
    tff.math.qmc.sobol_generating_matrices(dim, num_results, num_digits),
    num_results,
    num_digits)
# ==> tf.Tensor([
#             [0.,         0.        ],
#             [0.5,        0.5       ],
#             [0.25,       0.75      ],
#             ...
#             [0.65527344, 0.9736328 ],
#             [0.40527344, 0.7236328 ],
#             [0.90527344, 0.22363281],
#         ], shape=(1000, 2), dtype=float32)
```

#### Args:


* <b>`generating_matrices`</b>: Positive scalar `Tensor` of integers with rank 2. The
  matrix from which to sample points.
* <b>`num_results`</b>: Positive scalar `Tensor` of integers with rank 0. The maximum
  number of points to sample from `generating_matrices`.
* <b>`num_digits`</b>: Positive scalar `Tensor` of integers with rank 0. the base-2
  precision of the points sampled from `generating_matrices`.
* <b>`sequence_indices`</b>: Optional positive scalar `Tensor` of integers with rank 1.
  The elements of the sequence to return specified by their position in the
  sequence.
  Default value: `None` which corresponds to the `[0, num_results)` range.
* <b>`scrambling_matrices`</b>: Optional positive scalar `Tensor` of integers with the
  same shape as `generating_matrices`. The left matrix scramble to apply to
  the generating matrices.
  Default value: `None`.
* <b>`digital_shift`</b>: Optional positive scalar `Tensor` of integers with shape
  (`dim`) where `dim = tf.shape(generating_matrices)[0]`. The digital shift
  to apply to all the sampled points via a bitwise xor.
  Default value: `None`.
* <b>`apply_tent_transform`</b>: Python `bool` indicating whether to apply a tent
  transform to the sampled points.
  Default value: `False`.
* <b>`validate_args`</b>: Python `bool` indicating whether to validate arguments.
  Default value: `False`.
* <b>`dtype`</b>: Optional `dtype`. The `dtype` of the output `Tensor` (either
  `float32` or `float64`).
  Default value: `None` which maps to `float32`.
* <b>`name`</b>: Python `str` name prefixed to ops created by this function.
  Default value: `None` which maps to `sample_digital_net`.


#### Returns:

A `Tensor` of samples from  the Sobol sequence with `shape`
`(num_samples, dim)` where `num_samples = min(num_results,
size(sequence_indices))` and `dim = tf.shape(generating_matrices)[0]`.
