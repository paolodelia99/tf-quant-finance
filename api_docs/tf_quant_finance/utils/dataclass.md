<!--
This file is generated by a tool. Do not edit directly.
For open-source contributions the docs will be updated automatically.
-->

*Last updated: 2023-03-16.*

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.utils.dataclass" />
<meta itemprop="path" content="Stable" />
</div>

# tf_quant_finance.utils.dataclass

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/paolodelia99/tf-quant-finance/blob/main/tf_quant_finance/utils/dataclass.py">View source</a>



Creates a data class object compatible with `tf.function`.

```python
tf_quant_finance.utils.dataclass(
    cls
)
```



<!-- Placeholder for "Used in" -->

Modifies dunder methods of an input class with typed attributes to work as an
input/output to `tf.function`, as well as a loop variable of
`tf.while_loop`.

An intended use case for this decorator is on top of a simple class definition
with type annotated arguments like in the example below. It is not guaranteed
that this decorator works with an arbitrary class.


#### Examples

```python
import tensorflow as tf
import tf_quant_finance as tff

@tff.utils.dataclass
class Coords:
  x: tf.Tensor
  y: tf.Tensor

@tf.function
def fn(start_coords: Coords) -> Coords:
  def cond(it, _):
    return it < 10
  def body(it, coords):
    return it + 1, Coords(x=coords.x + 1, y=coords.y + 2)
  return tf.while_loop(cond, body, loop_vars=(0, start_coords))[1]

start_coords = Coords(x=tf.constant(0), y=tf.constant(0))
fn(start_coords)
# Expected Coords(a=10, b=20)
```

#### Args:


* <b>`cls`</b>: Input class object with type annotated arguments. The class should not
  have an init method defined. Class fields are treated as ordered in the
  same order as they appear in the class definition.



#### Returns:

Modified class that can be used as a `tf.function` input/output as well
as a loop variable of `tf.function`. All typed arguments of the original
class are treated as ordered in the same order as they appear in the class
definition. All untyped arguments are ignored. Modified class modifies
`len` and `iter` methods defined for the  class instances such that `len`
returns the number of arguments, and `iter`  creates an iterator for the
ordered argument values.
