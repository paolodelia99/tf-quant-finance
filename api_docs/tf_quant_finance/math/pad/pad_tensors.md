<!--
This file is generated by a tool. Do not edit directly.
For open-source contributions the docs will be updated automatically.
-->

*Last updated: 2023-03-16.*

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.math.pad.pad_tensors" />
<meta itemprop="path" content="Stable" />
</div>

# tf_quant_finance.math.pad.pad_tensors

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/paolodelia99/tf-quant-finance/blob/main/tf_quant_finance/math/pad.py">View source</a>



Pads the innermost dimension of `Tensor`s to a common shape.

```python
tf_quant_finance.math.pad.pad_tensors(
    tensors, pad_values=None, dtype=None, name=None
)
```



<!-- Placeholder for "Used in" -->

Given a list of `Tensor`s of the same `dtype` and with shapes
`batch_shape_i + [n_i]`, pads the innermost dimension of each tensor to
`batch_shape_i + [max(n_i)]`. For each tensor `t`, the padding is done with
values `t[..., -1]`.

### Example. Pad with the terminal value
```python
x = [[1, 2, 3, 9], [2, 3, 5, 2]]
y = [4, 5, 8]
pad_tensors([x, y])
# Expected: [array([[1, 2, 3, 9], [2, 3, 5, 2]], array([4, 5, 8, 8])]
```

### Example. Pad with user-supplied values
```python
x = [[1, 2, 3, 9], [2, 3, 5, 2]]
y = [4, 5, 8]
pad_tensors([x, y], pad_values=10)
# Expected: [array([[1, 2, 3, 9], [2, 3, 5, 2]], array([4, 5, 8, 10])]
```

#### Args:


* <b>`tensors`</b>: A list of tensors of the same `dtype` and shapes
  `batch_shape_i + [n_i]`.
* <b>`pad_values`</b>: An optional scalar `Tensor` or a list of `Tensor`s of shapes
  broadcastable with the corresponding `batch_shape_i` and the same
  `dtype` as `pad_values`. Corresponds to the padded values used for
  `tensors`.
* <b>`dtype`</b>: The default dtype to use when converting values to `Tensor`s.
  Default value: `None` which means that default dtypes inferred by
    TensorFlow are used.
* <b>`name`</b>: Python string. The name to give to the ops created by this class.
  Default value: `None` which maps to the default name `pad_tensors`.

#### Returns:

A list of `Tensor`s of shape `batch_shape_i + [max(n_i)]`.



#### Raises:


* <b>`ValueError`</b>: If input is not an instance of a list or a tuple.