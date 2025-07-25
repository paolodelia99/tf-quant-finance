<!--
This file is generated by a tool. Do not edit directly.
For open-source contributions the docs will be updated automatically.
-->

*Last updated: 2023-03-16.*

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.math.pde.steppers.implicit.implicit_step" />
<meta itemprop="path" content="Stable" />
</div>

# tf_quant_finance.math.pde.steppers.implicit.implicit_step

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/paolodelia99/tf-quant-finance/blob/main/tf_quant_finance/math/pde/steppers/implicit.py">View source</a>



Creates a stepper function with implicit time marching scheme.

```python
tf_quant_finance.math.pde.steppers.implicit.implicit_step()
```



<!-- Placeholder for "Used in" -->

Given a space-discretized equation

```
du/dt = A(t) u(t) + b(t)
```
(here `u` is a value vector, `A` and `b` are the matrix and the vector defined
by the PDE), the implicit time marching scheme approximates the right-hand
side with its value after the time step:

```
(u(t2) - u(t1)) / (t2 - t1) = A(t2) u(t2) + b(t2)
```
This scheme is stable, but is only first order accurate.
Usually, Crank-Nicolson or Extrapolation schemes are preferable.

More details can be found in `weighted_implicit_explicit.py` describing the
weighted implicit-explicit scheme - implicit scheme is a special case
with `theta = 0`.

#### Returns:

Callable to be used in finite-difference PDE solvers (see fd_solvers.py).
