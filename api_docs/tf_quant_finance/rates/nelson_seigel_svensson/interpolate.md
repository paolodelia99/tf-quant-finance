<!--
This file is generated by a tool. Do not edit directly.
For open-source contributions the docs will be updated automatically.
-->

*Last updated: 2023-03-16.*

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.rates.nelson_seigel_svensson.interpolate" />
<meta itemprop="path" content="Stable" />
</div>

# tf_quant_finance.rates.nelson_seigel_svensson.interpolate

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/paolodelia99/tf-quant-finance/blob/main/tf_quant_finance/rates/nelson_seigel_svensson/nelson_seigel_svensson_interpolation.py">View source</a>



Performs Nelson Seigel Svensson interpolation for supplied points.

```python
tf_quant_finance.rates.nelson_seigel_svensson.interpolate(
    interpolation_times, svensson_parameters, dtype=None, name=None
)
```



<!-- Placeholder for "Used in" -->

Given a set of interpolation times and the parameters for the nelson seigel
svensson model, this function returns the interpolated values for the yield
curve. We assume that the parameters are already computed using a fitting
technique.
```None
r(T) = beta_0 +
       beta_1 * (1-exp(-T/tau_1))/(T/tau_1) +
       beta_2 * ((1-exp(-T/tau_1))/(T/tau_1) - exp(-T/tau_1)) +
       beta_3 * ((1-exp(-T/tau_2))/(T/tau_2) - exp_(-T/tau_2))
```

Where `T` represents interpolation times and
`beta_i`'s and `tau_i`'s are paramters for the model.

#### Example
```python
import tf_quant_finance as tff
interpolation_times = [5., 10., 15., 20.]
svensson_parameters =
tff.rates.nelson_svensson.interpolate.SvenssonParameters(
      beta_0=0.05, beta_1=-0.01, beta_2=0.3, beta_3=0.02,
      tau_1=1.5, tau_2=20.0)
result = interpolate(interpolation_times, svensson_parameters)
# Expected_result
# [0.12531, 0.09667, 0.08361, 0.07703]
```

#### References:
  [1]: Robert Müller. A technical note on the Svensson model as applied to
  the Swiss term structure.
  BIS Papers No 25, Mar 2015.
  https://www.bis.org/publ/bppdf/bispap25l.pdf

#### Args:


* <b>`interpolation_times`</b>: The times at which interpolation is desired. A N-D
  `Tensor` of real dtype where the first N-1 dimensions represent the
  batching dimensions.
* <b>`svensson_parameters`</b>: An instance of `SvenssonParameters`. All parameters
  within should be real tensors.
* <b>`dtype`</b>: Optional tf.dtype for `interpolation_times`. If not specified, the
  dtype of the inputs will be used.
* <b>`name`</b>: Python str. The name prefixed to the ops created by this function. If
  not supplied, the default name 'nelson_svensson_interpolation' is used.


#### Returns:

A N-D `Tensor` of real dtype with the same shape as `interpolations_times`
  containing the interpolated yields.
