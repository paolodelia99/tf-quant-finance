<!--
This file is generated by a tool. Do not edit directly.
For open-source contributions the docs will be updated automatically.
-->

*Last updated: 2023-03-16.*

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.models.hjm.CalibrationResult" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__ge__"/>
<meta itemprop="property" content="__gt__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__iter__"/>
<meta itemprop="property" content="__le__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="__lt__"/>
<meta itemprop="property" content="__ne__"/>
</div>

# tf_quant_finance.models.hjm.CalibrationResult

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/paolodelia99/tf-quant-finance/blob/main/tf_quant_finance/models/hjm/calibration.py">View source</a>



Collection of calibrated QuasiGaussianHJM parameters.

```python
tf_quant_finance.models.hjm.CalibrationResult(
    mean_reversion, volatility, corr_matrix
)
```



<!-- Placeholder for "Used in" -->

For a review of the HJM model and the conventions used, please see the
docstring for `QuasiGaussianHJM`, or for `calibration_from_swaptions` below.

#### Attributes:

* <b>`mean_reversion`</b>: Rank-1 `Tensor` specifying the mean-reversion parameter.
* <b>`volatility`</b>: Rank-1 `Tensor` specifying the volatility parameter.
* <b>`corr_matrix`</b>: Rank-1 `Tensor` specifying the correlation matrix parameter.

## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

```python
__eq__(
    other
)
```

Method generated by attrs for class CalibrationResult.


<h3 id="__ge__"><code>__ge__</code></h3>

```python
__ge__(
    other
)
```

Method generated by attrs for class CalibrationResult.


<h3 id="__gt__"><code>__gt__</code></h3>

```python
__gt__(
    other
)
```

Method generated by attrs for class CalibrationResult.


<h3 id="__iter__"><code>__iter__</code></h3>

<a target="_blank" href="https://github.com/paolodelia99/tf-quant-finance/blob/main/tf_quant_finance/utils/dataclass.py">View source</a>

```python
__iter__()
```




<h3 id="__le__"><code>__le__</code></h3>

```python
__le__(
    other
)
```

Method generated by attrs for class CalibrationResult.


<h3 id="__len__"><code>__len__</code></h3>

<a target="_blank" href="https://github.com/paolodelia99/tf-quant-finance/blob/main/tf_quant_finance/utils/dataclass.py">View source</a>

```python
__len__()
```




<h3 id="__lt__"><code>__lt__</code></h3>

```python
__lt__(
    other
)
```

Method generated by attrs for class CalibrationResult.


<h3 id="__ne__"><code>__ne__</code></h3>

```python
__ne__(
    other
)
```

Method generated by attrs for class CalibrationResult.




