<!--
This file is generated by a tool. Do not edit directly.
For open-source contributions the docs will be updated automatically.
-->

*Last updated: 2023-03-16.*

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.math.pde.steppers.composite_stepper.composite_scheme_step" />
<meta itemprop="path" content="Stable" />
</div>

# tf_quant_finance.math.pde.steppers.composite_stepper.composite_scheme_step

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/paolodelia99/tf-quant-finance/blob/main/tf_quant_finance/math/pde/steppers/composite_stepper.py">View source</a>



Composes two time marching schemes.

```python
tf_quant_finance.math.pde.steppers.composite_stepper.composite_scheme_step(
    first_scheme_steps, first_scheme, second_scheme
)
```



<!-- Placeholder for "Used in" -->

Applies a step of parabolic PDE solver using `first_scheme` if number of
performed steps is less than `first_scheme_steps`, and using `second_scheme`
otherwise.

#### Args:


* <b>`first_scheme_steps`</b>: A Python integer. Number of steps to apply
  `first_scheme` on.
* <b>`first_scheme`</b>: First time marching scheme (see `time_marching_scheme`
  argument of `parabolic_equation_step`).
* <b>`second_scheme`</b>: Second time marching scheme (see `time_marching_scheme`
  argument of `parabolic_equation_step`).


#### Returns:

Callable to be used in finite-difference PDE solvers (see fd_solvers.py).
