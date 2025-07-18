<!--
This file is generated by a tool. Do not edit directly.
For open-source contributions the docs will be updated automatically.
-->

*Last updated: 2023-03-16.*

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.experimental.svi.total_variance_from_raw_svi_parameters" />
<meta itemprop="path" content="Stable" />
</div>

# tf_quant_finance.experimental.svi.total_variance_from_raw_svi_parameters

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/paolodelia99/tf-quant-finance/blob/main/tf_quant_finance/experimental/svi/parameterizations.py">View source</a>



Computes modeled total variance using raw SVI parameters.

```python
tf_quant_finance.experimental.svi.total_variance_from_raw_svi_parameters(
    *, svi_parameters, log_moneyness=None, forwards=None, strikes=None, dtype=None,
    name=None
)
```



<!-- Placeholder for "Used in" -->

The SVI volatility model parameterizes an option's total implied variance,
defined as w(k,t) := sigmaBS(k,t)^2 * t, where k := log(K/F) is the options's
log-moneyness, t is the time to expiry, and sigmaBS(k,t) is the Black-Scholes
market implied volatility. For a fixed timeslice (i.e. given expiry t), the
raw SVI parameterization consists of 5 parameters (a,b,rho,m,sigma), and
the model approximation formula for w(k,t) as a function of k is (cf.[1]):
```None
w(k) = a + b * (rho * (k - m) + sqrt{(k - m)^2 + sigma^2)}
```
The raw parameters have the following interpretations (cf.[2]):
a      vertically shifts the variance graph
b      controls the angle between the left and right asymptotes
rho    controls the rotation of the variance graph
m      horizontally shifts the variance graph
sigma  controls the graph smoothness at the vertex (ATM)

#### Example

```python
import numpy as np
import tensorflow as tf
import tf_quant_finance as tff

svi_parameters = np.array([-0.1825, 0.3306, -0.0988, 0.0368, 0.6011])

forwards = np.array([2402.])
strikes = np.array([[1800., 2000., 2200., 2400., 2600., 2800., 3000.]])

total_var = tff.experimental.svi.total_variance_from_raw_svi_parameters(
    svi_parameters=svi_parameters, forwards=forwards, strikes=strikes)

# Expected: total_var tensor (rounded to 4 decimal places) should contain
# [[0.0541, 0.0363, 0.02452, 0.0178, 0.0153, 0.0161, 0.0194]]
```

#### References:
[1] Gatheral J., Jaquier A., Arbitrage-free SVI volatility surfaces.
https://arxiv.org/pdf/1204.0646.pdf
[2] Gatheral J, A parsimonious arbitrage-free implied volatility
parameterization with application to the valuation of volatility derivatives.
http://faculty.baruch.cuny.edu/jgatheral/madrid2004.pdf

#### Args:


* <b>`svi_parameters`</b>: A rank 2 real `Tensor` of shape [batch_size, 5]. The raw SVI
  parameters for each volatility skew.
* <b>`log_moneyness`</b>: A rank 2 real `Tensor` of shape [batch_size, num_strikes].
  The log-moneyness of the option.
* <b>`forwards`</b>: A rank 2 real `Tensor` of shape [batch_size, num_strikes]. The
  forward price of the option at expiry.
* <b>`strikes`</b>: A rank 2 real `Tensor` of shape [batch_size, num_strikes]. The
  option's strike price.
* <b>`dtype`</b>: Optional `tf.Dtype`. If supplied, the dtype for the input and output
  `Tensor`s will be converted to this.
  Default value: `None` which maps to the dtype inferred from
    `log_moneyness`.
* <b>`name`</b>: Python str. The name to give to the ops created by this function.
  Default value: `None` which maps to `svi_total_variance`.


#### Returns:

A rank 2 real `Tensor` of shape [batch_size, num_strikes].



#### Raises:


* <b>`ValueError`</b>: If exactly one of `forwards` and `strikes` is supplied.
* <b>`ValueError`</b>: If both `log_moneyness' and `forwards` are supplied or if
neither is supplied.