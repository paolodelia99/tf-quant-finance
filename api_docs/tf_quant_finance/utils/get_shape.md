<!--
This file is generated by a tool. Do not edit directly.
For open-source contributions the docs will be updated automatically.
-->

*Last updated: 2023-03-16.*

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.utils.get_shape" />
<meta itemprop="path" content="Stable" />
</div>

# tf_quant_finance.utils.get_shape

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/paolodelia99/tf-quant-finance/blob/main/tf_quant_finance/utils/shape_utils.py">View source</a>



Returns static shape of `x` if it is fully defined, or dynamic, otherwise.

```python
tf_quant_finance.utils.get_shape(
    x, name=None
)
```



<!-- Placeholder for "Used in" -->

####Example
```python
import tensorflow as tf
import tf_quant_finance as tff

x = tf.zeros([5, 2])
prefer_static_shape(x)
# Expected: [5, 2]

#### Args:


* <b>`x`</b>: A tensor of any shape and `dtype`
* <b>`name`</b>: Python string. The name to give to the ops created by this function.
  Default value: `None` which maps to the default name
  `get_shape`.


#### Returns:

A shape of `x` which a list, if the shape is fully defined, or a `Tensor`
for dynamically shaped `x`.
