# TensorFlow Quantitative Finance.

This library provides high-performance components leveraging the hardware acceleration support and automatic differentiation of TensorFlow. The library will provide TensorFlow support for foundational mathematical methods, mid-level methods, and specific pricing models. 

## Installation

First and foremost install and upgrade your tensorflow version

    pip install --upgrade tensorflow tf_keras

then install the library itself:

    pip install tf-q-finance

## Modules

[`black_scholes`](./black_scholes.md) module: TensorFlow Quantitative Finance volatility surfaces and vanilla options.

[`datetime`](./datetime.md) module: Date-related utilities.

[`math`](./math.md) module: TensorFlow Quantitative Finance general math functions.

[`models`](./models.md) module: TensorFlow Quantitative Finance tools to build Diffusion Models.

[`rates`](./rates.md) module: Functions to handle rates.

[`types`](./types.md) module: Types module.

[`utils`](./utils.md) module: Utilities module.

## Examples

For practical usage and demonstrations, please see the [examples folder in the repository](https://github.com/paolodelia99/tf-quant-finance/tree/main/tf_quant_finance/examples). It contains Jupyter notebooks and scripts showcasing various features and use cases of TensorFlow Quantitative Finance.