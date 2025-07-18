<!--
This file is generated by a tool. Do not edit directly.
For open-source contributions the docs will be updated automatically.
-->

*Last updated: 2023-03-16.*

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.math.pde.steppers.douglas_adi.douglas_adi_step" />
<meta itemprop="path" content="Stable" />
</div>

# tf_quant_finance.math.pde.steppers.douglas_adi.douglas_adi_step

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/paolodelia99/tf-quant-finance/blob/main/tf_quant_finance/math/pde/steppers/douglas_adi.py">View source</a>



Creates a stepper function with Crank-Nicolson time marching scheme.

```python
tf_quant_finance.math.pde.steppers.douglas_adi.douglas_adi_step(
    theta=0.5
)
```



<!-- Placeholder for "Used in" -->

Douglas ADI scheme is the simplest time marching scheme for solving parabolic
PDEs with multiple spatial dimensions. The time step consists of several
substeps: the first one is fully explicit, and the following `N` steps are
implicit with respect to contributions of one of the `N` axes (hence "ADI" -
alternating direction implicit). See `douglas_adi_scheme` below for more
details.

#### Args:


* <b>`theta`</b>: positive Number. `theta = 0` corresponds to fully explicit scheme.
The larger `theta` the stronger are the corrections by the implicit
substeps. The recommended value is `theta = 0.5`, because the scheme is
second order accurate in that case, unless mixed second derivative terms are
present in the PDE.

#### Returns:

Callable to be used in finite-difference PDE solvers (see fd_solvers.py).
