<!--
This file is generated by a tool. Do not edit directly.
For open-source contributions the docs will be updated automatically.
-->

*Last updated: 2023-03-16.*

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.math.pde.steppers.extrapolation.extrapolation_step" />
<meta itemprop="path" content="Stable" />
</div>

# tf_quant_finance.math.pde.steppers.extrapolation.extrapolation_step

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/paolodelia99/tf-quant-finance/blob/main/tf_quant_finance/math/pde/steppers/extrapolation.py">View source</a>



Creates a stepper function with Extrapolation time marching scheme.

```python
tf_quant_finance.math.pde.steppers.extrapolation.extrapolation_step()
```



<!-- Placeholder for "Used in" -->

Extrapolation scheme combines two half-steps and the full time step to obtain
desirable properties. See more details below in `extrapolation_scheme`.

It is slower than Crank-Nicolson scheme, but deals better with value grids
that have discontinuities. Consider also `oscillation_damped_crank_nicolson`,
an efficient combination of Crank-Nicolson and Extrapolation schemes.

#### Returns:

Callable to be used in finite-difference PDE solvers (see fd_solvers.py).
