# sfc-arch-fv
Forecast-Validation for optimal ARCH/GARCH model selection in Python

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Workflow](#workflow)
- [Usage Example](#usage-example)
- [Models](#models)
- [Extra Modules](#extra-modules)
- [Requirements](#requirements)
- [References](#references)
- [License](#license)

## Overview
This repository develops a forecast validation framework in Python for comparing multiple univariate conditional volatility forecast models based on their out-of-sample forecast losses.

## Installation
```bash
git clone https://github.com/donn-majidi/sfclubunibo-arch-fv.git
cd sfclubunibo-arch-fv
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

See [Requirements](#requirements) for the list of required packages.

## Workflow
| Step | Stage | Code | Produces |
|---|---|---|---|
| 1 | Prepare the return series | from data pipeline (e.g. `np.log(prices).diff() * 100`) | a demeaned/log return series, free of `NaN`s |
| 2 | Build candidate models | `arch_model(...)`, or `ZeroMean`/`ARX` + a volatility process + a distribution, from the [`arch`](https://bashtage.github.io/arch/) package | an object array of `ARCHModel` instances |
| 3 | Rolling-window estimation & forecasting | [`Validator(endog, models).validate(...)`](#models) | index-aligned forecasts, `mse_loss`/`qlike_loss`, VaR/ES, `std_residuals`, `model_fits` |
| 4 | Multiple-comparison testing (optional) | `arch.bootstrap.StepM` / `MCS`, fed `Validator`'s loss series, with block size from [`bootstrap_block_size`](#bootstrap_block_size) | superior models (StepM) / the model confidence set (MCS) |
| 5 | *Ex-post* diagnostics (optional) | [`cusum_supf_test`](#cusum_supf_test), [`GenParetoMLE`](#class-genparetomle) + [`tail_index_ci`](#tail_index_ci), [`jb_test`](#jb_test), fed `Validator`'s `std_residuals` | moment-stability test, tail-index estimate & CI, normality test |

## Usage Example
```python
import numpy as np
import pandas as pd
from src.models.validator import Validator

## The following is an illustrative example using data fetched from yahoo finance.
# pip install yfinance #run this line if yfinance is not already installed.
import yfinance as yf
sp500 = yf.Ticker('^GSPC').history(start='2022-01-01')
returns = np.log(sp500['Close']).diff() * 100 ## Always a good idea to multiply returns by 100
returns = returns.dropna()

## Import arch model constructor
from arch import arch_model
mod1 = arch_model(returns, vol='GARCH', p=1, o=0, q=1, dist='gaussian') ## GARCH(1,1) with Gaussian likelihood function.
mod2 = arch_model(returns, vol='GARCH', p=1, o=1, q=1, dist='studentst') ## GJR_GARCH(1,1,1) with Studend-t likelihood function.

## Create array of model instances
models = np.asarray([mod1, mod2], dtype=object) ## The dtype of the array has to be explicitly set to object.

## Create and instance of the Validator class
model_validator = Validator(endog=returns, models=models)

## Set the input parameters to feed to the validate() method
ws = 252  # Window size
uf = 21   # Update frequency
fh = 1    # Forecast horizon
alpha = [0.01,0.05]  # Significance levels for Value at Risk and Expected Shortfall estimation. This parameter is optional.
align = 'target'  # Index align method for out-of-sample forecasts and corresponding losses. If not passed, the default align method will be used: 'origin'.

model_validator.validate(window_size=ws, horizon=fh, update_frequency=uf, alpha = alpha, align=align)
```
> [!NOTE]
> 1. Value at Risk and Expected Shortfall forecasts will only be computed if a value for alpha is passed to the validate() method.
>
> 2. The default index align method for the out-of-sample forecasts is 'origin' as in the default behavior of the forecast() method in the arch package. Setting the align method to 'target' can ease direct comparison with the data as no further index alignment would be required. Compare:

| index: align = 'origin'  |  h.1 | h.2  | index: align = 'target'  | h.1  | h.2 |
|--------|------|----- | ------ | ---- | --- |
| 2026-08-03 | `1.0310` | `1.0315` | 2026-08-03 | NaN | NaN |
| 2026-08-04 | `1.2406` | 1.2337 | 2026-08-04 | `1.0310` | NaN |
| 2026-08-05 | 1.1138 | 1.1113 | 2026-08-05 | `1.2406` | `1.0315` |

```python
## Out-of-sample forecasts along with forecast losses can be accessed from class properties.
mv_forecasts = model_validator.forecasts  ## DataFrame containing index aligned out-of-sample forecasts
mv_mse = model_validator.mse_loss  ## DataFrame containing squared forecast errors
mv_qlike = model_validator.qlike_loss  ## DataFrame containing quasi-likelihood scores

## If alpha is passed, the class instance will also contain Value at Risk and Expected Shortfall forecasts
## which can be accessed from class properties
mv_var = model_validator.value_at_risk
mv_exp = model_validator.expected_shortfall

## Standardized residuals obtained from each iteration can be accessed via:
mv_resid = model_validator.std_residuals
```
>[!NOTE]
>Regardless of the index alignment method chosen for the out-of-sample forecasts, the indices of the standardized residuals are always 'origin' aligned, as they have to correspond to the last observation used to fit the model.

```python
## Finally, the model fit results from the last iteration can be accessed from the model_results property
mod1_fit = model_validator.model_fits[0]
mod2_fit = model_validator.model_fits[1]
```
## Models

### `class Validator`
```python
class Validator(endog: np.ndarray,
                models: np.ndarray)
```
Model validator class for rolling-window forecast loss evaluations.
#### Parameters
- `endog`: Return series (e.g. demeaned log returns) the models were built on.
- `models`: 1-D array of model instances.

#### Methods
```python
validate(window_size: int,
          horizon: int,
          update_frequency: int = 1,
          alpha: np.ndarray | None = None,
          align: str | None = 'origin')
  ```
  
  Run the validation process for given input parameters.

  #### Parameters:
  - `window_size`: Number of observations in each rolling estimation window.
  - `horizon`: Forecast horizon h to evaluate.
  - `update_frequency`: Update frequency for re-fitting the models.
  - `alpha`: Array of significance levels for Value at Risk and Expected Shortfall forecasting. This parameter is optional, if not passed, VaR and ES forecasts will not be computed.
  - `align`: Index alignment method: 'origin' or 'target'. Default is 'origin'.

> [!NOTE]
  > 1. Value at Risk and Expected Shortfall forecasts will only be computed if a value for alpha is passed to the validate() method.
  >
  > 2. The default index align method for the out-of-sample forecasts is 'origin' as in the default behavior of the forecast() method in the arch package. Setting the align method to 'target' can   ease direct comparison with the data as no further index alignment would be required. Compare:

  | index: align = 'origin'  |  h.1 | h.2  | index: align = 'target'  | h.1  | h.2 |
  |--------|------|----- | ------ | ---- | --- |
  | 2026-08-03 | `1.0310` | `1.0315` | 2026-08-03 | NaN | NaN |
  | 2026-08-04 | `1.2406` | 1.2337 | 2026-08-04 | `1.0310` | NaN |
  | 2026-08-05 | 1.1138 | 1.1113 | 2026-08-05 | `1.2406` | `1.0315` |

```python
compute_loss(forecasts: np.ndarray,
              window_size: int,
              horizon: int,
              loss_function:  'mse' | 'MSE', 'mae', 'MAE', 'qlike', 'QLIKE')
  ```
  Compute the desired loss function given input parameters.
  #### Parameters:
  - `forecasts`: n-Dimensional array of model forecasts.
  - `window_size`: Number of observations in each rolling estimation window.
  - `horizon`: Forecast horizon h to evaluate.
  - `loss_function`: String indicating the loss function to compute. Must be one of `('mse', 'MSE', 'mae', 'MAE', 'qlike', 'QLIKE')`

#### Properties
- `forecasts`: Dataframe of h-step-ahead conditional variance forecast per model.
- `mse_loss`: Dataframe of h-step-ahead conditional variance squared error loss per model.
- `mae_loss`: Dataframe of h-step-ahead conditional variance absolute error loss per model.
- `qlike_loss`: Dataframe of h-step-ahead conditional variance quasi-likelihood score per model.
- `value_at_risk`: Dataframe of h-step-ahead conditional value at risk forecast per model. The columns are multi-indexed per model per significance level.
- `expected_shortfall`: Dataframe of h-step-ahead conditional expected shortfall per model. The columns are multi-indexed per model per significance level.
- `std_residuals`: Standardized residuals obtained from the last observation in each estimation window.
- `model_fits`: Array containing fitted parameters from the last estimation loop.

## Extra Modules

### `Class LossContainer`
```python
class LossContainer(observations: np.ndarray,
                    forecasts: np.ndarray,
                    forecast_horizon: int = None)
```
Generic class for loss function calculations. It takes as input the array of model forecasts and the index-aligned observations and it calculates the MSE, MAE, and QLIKE loss of each forecast. The estimated loss series are stored as properties. The class can be optionally instantiated with `forecast_horizon` which will include the forecast horizon in the summary results.
observations: 1-Dimensional array of observations. Must be in the same units as the forecasts.

#### Parameters
- `observations`: 1-Dimensional array of observations. Must be in the same units as the forecasts.
- `forecasts`: 1-Dimensional or 2-dimensional array of model forecasts. Index must be 'target' aligned.
- `forecast_horizon`: Integer determining the forecast horizon.

#### Properties
- `mse_loss`: Series of h-step-ahead conditional variance squared error loss per model.
- `mae_loss`: Series of h-step-ahead conditional variance absolute error loss per model.
- `qlike_loss`: Series of h-step-ahead conditional variance quasi-likelihood score per model.

> [!NOTE]
> The Validator class automatically computes the loss series and handles index-alignment internally. Only use the LossContainer class if using an alternative forecasting scheme.
```python
from src.modules.loss_container import LossContainer

## assuming that the array of out-of-sample forecasts is stored in forecasts
## and the univariate array of observations (variance proxies in the case of volatility forecasts) is stored in observations
## then the class should be loaded as:
lc = LossContainer(observations, forecasts, forecast_horizon=1)
lc_mse = lc.mse_loss
print(lc_mse)

## average forecast losses
print(lc)
```
>[!NOTE]
>Make sure that the forecasts and the observations are in the same units and that the index of the forecasts is 'target' aligned.

Continuing the example from section [Usage Example](#usage-example):
```python
mv_index = mv_forecasts.index
observations = returns.loc[mv_index]**2
lc2 = LossContainer(observations, mv_forecasts, forecast_horizon=fh)
lc2_mse = lc2.mse_loss
print(lc2_mse)

## average forecast losses
print(lc2)
```

### `bootstrap_block_size`
This function implements the automatic block-length selection procedure of Politis & White (2004) / Patton, Politis & White (2009).
- It takes as input a dataframe or a series containing the estimated model losses and calculates the optimal block-size for each column.

Parameters:
- `x`: 1-Dimensional or 2-dimensional array of input time-series.

Returns:
- `pd.DataFrame` containing the estimated optimal block size for each of the input series per bootstrapping algorithm ('Stationary Bootstrap', 'Circular Bootstrap', 'Moving-Blocks Bootstrap')
```python
from src.modules.bootstrap_params import bootstrap_block_size
opt_bs = bootstrap_block_size(mv_qlike_loss)
print(opt_bs)
```
### `cusum_supf_test`
This function implements Bruce Hansen's CUSUM of Squares SupF test.

Parameters:
- `x`: 1-Dimensional array of standardized residuals.
- `moment`: Moment order being tested. Has to be either 2 or 4.
- `alpha`: The size of the test. Default is 0.05.
- `trim`: Trimming fraction for trimming the CUSUM path.
- `bandwidth`: Kernel bandwidth for variance estimation.
- `ax`: plt.Axes canvas on which to plot the graph of the CUSUM path along with the confidence bands.

Returns:
- `dict()` containing:
    - the SupF statistic
    - Chi2 test statistic
    - pvalue
    - breakpoint (where SupF exceeds the confidence bands.)

```python
from src.modules.standard_diagnostics import cusum_supf_test
zs_0 = mv_resid[0] ## Univariate array of standardized residuals

import matplotlib.pyplot as plt
fig, ax = plt.subplots()

cusum_results = cusum_supf_test(zs_0, moment=2, ax=ax)
print(cusum_results)
plt.show()
```
### `class GenParetoMLE`
This is a child class of statsmodels GenericLikelihoodModel to estimate the shape and scale parameters of a Generalized Pareto Distribution.
```python
class GenParetoMLE(endog: np.ndarray)
```
#### Parameters
- `endog`: Tail observations for GPD parameters estimation. Must already be centered.

#### Methods
- `fit()`:
  Fit the model. Method return type is GenericLikelihoodModelResults.
  Among the properties of the GenericLikelihoodModelResults are the estimated parameters which can be accessed via:
  
### Properties of GenericLikelihoodModelResults
- `params`: Estimated parameters.
  - `params[0]` contains the estimated shape parameter.
  - `params[1]` contains the estimated scale parameter.

```python
from src.modules.standard_diagnostics import GenParetoMLE
thresh = np.percentile(zs_0, 95)
ex_0 = zs_0[zs_0 > thresh] - thresh
gp_model = GenParetoMLE(ex_0)
gp_fit = gp_model.fit()
gp_fit.summary()

xi_hat = gp.params[0]
sigma_hat = gp.params[1]
```

### `tail_index_ci`
This function computes the standard error of the estimated tail index and the corresponding confidence intervals at given percentiles. The MLE of the shape parameter $\hat{\xi}$ is asymptotically normally distributed as:
```math
                        \sqrt{m} (\hat{\xi} - \xi_0) \sim \mathcal{N}(0, (1+\xi_0)^2), 
```
where $m$ is the number of extreme observations, i.e., the size of the series passed to the function, and $\xi_0$ is the hypothesized value of the tail index. Using the definition above, the standard error of the tail index estimator can be computed as:
```math
                      \hat{\nu} = \frac{ 1 + \hat{\xi} }{\sqrt{m}}.
```
Hence the confidence intervals at significance level $\alpha$ are computed as:
```math
                    CI = \left[  \hat{\xi} - \Phi^{-1}(1-\alpha/2) \hat{\nu} \, , \, \hat{\xi} + \Phi^{-1}(1-\alpha/2) \hat{\nu}  \right].
```
Trivially, moments of order $r$ and lower exist if and only if $r < 1/\xi_0$.

>[!NOTE]
>The maximum likelihood estimator is asymptotically normal only in the region $\xi \in (-0.5,\infty)$. For values of the estimated tail index less than -0.5 the MLE is no longer asymptotically normal, and thus confidence bands cannot be computed.

Parameters:
- `z`: Univariate array of centered exceedances above threshold. Must be the same array fed to GenParetoMLE.
- `xi_hat`: MLE estimate of the shape parameter.
- `sigma_hat`: MLE estimate of the scale parameter.
- `alpha`: Significance level for the confidence interval (e.g. 0.05 for a 95% CI). Must be strictly between 0 and 1. The default is 0.05.
- `bandwidth`: Optional. Number of bins to plot the histogram.
- `trim_quantile`: Optional | Default = 0.99. The cut-off quantile on the plot. By trimming the extreme values makes the plot tidier.
- `ax: Optional`: plt.Axes canvas on which to plot the empirical pdf of the input data along with the theoretical pdf of the Generalized Pareto Distribution with given shape and scale parameters.

Returns:
- `dict()` containing:
  - xi_hat
  - Std Error
  - CI Lower
  - CI Upper
```python
from src.modules.standard_diagnostics import tail_index_ci

## Pickands tail index estimator
fig, ax = plt.subplots()
tail_indx = tail_index_ci(ex_0, xi_hat = xi_hat, sigma_hat = sigma_hat, alpha = 0.05, ax=ax)
print(tail_indx)
plt.show()
```
### `jb_test`
This function implements the Jarque-Bera test for asymptotic normality of the standardized residuals. It takes as input the sample data and returns the Jarque-Bera test statistic along with its associated critical value and the p-value of the test.

The Jarque-Bera test statistic is defined as
```math
                JB = T/6 (\hat{S}^2 + \frac{(\hat{K} - 3)^2}{4} ),
```
where $\hat{S}$ and $\hat{K}$ are sample estimates of Skewness and Kurtosis of the sample data. Under the null hypothesis that the sample data is normally distributed, the JB test statistic has an asymptotic chi-squared distribution with two degrees of freedom:
```math
                JB \sim \mathcal{\chi}^2(2).
```
Parameters:
- `z`: Univariate array of standardized random variables.
Returns:
- `dict()` containing:
  - test statistic
  - critical value
  - p-value of the test

```python
from src.modules.standard_diagnostics import jb_test
jb_results = jb_test(zs_0)
print(jb_results)
```
## Requirements
- [`numpy>=2.3.0`](https://numpy.org/)
- [`pandas>=2.3.0`](https://pandas.pydata.org/)
- [`scipy>=1.16.0`](https://scipy.org/)
- [`matplotlib>=3.10.0`](https://matplotlib.org/)
- [`seaborn>=0.13.0`](https://seaborn.pydata.org/)
- [`scikit-learn>=1.8.0`](https://scikit-learn.org/)
- [`statsmodels>=0.14.0`](https://www.statsmodels.org/)
- [`arch>=8.0.0`](https://bashtage.github.io/arch/)
- [`yfinance>=1.6.0`](https://github.com/ranaroussi/yfinance)
## References
- Balkema, A. A., & de Haan, L. (1974). Residual Life Time at Great Age. *The Annals of Probability*, 2(5), 792–804.
- Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327.
- Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation. *Econometrica*, 50(4), 987–1007.
- Hansen, B. E. (1997). Approximate Asymptotic P Values for Structural-Change Tests. *Journal of Business & Economic Statistics*, 15(1), 60–67.
- Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The Model Confidence Set. *Econometrica*, 79(2), 453–497.
- Jarque, C. M., & Bera, A. K. (1980). Efficient Tests for Normality, Homoscedasticity and Serial Independence of Regression Residuals. *Economics Letters*, 6(3), 255–259.
- Patton, A. J. (2011). Volatility Forecast Comparison Using Imperfect Volatility Proxies. *Journal of Econometrics*, 160(1), 246–256.
- Patton, A., Politis, D. N., & White, H. (2009). Correction to "Automatic Block-Length Selection for the Dependent Bootstrap" by D. Politis and H. White. *Econometric Reviews*, 28(4), 372–375.
- Pickands III, J. (1975). Statistical Inference Using Extreme Order Statistics. *The Annals of Statistics*, 3(1), 119–131.
- Politis, D. N., & White, H. (2004). Automatic Block-Length Selection for the Dependent Bootstrap. *Econometric Reviews*, 23(1), 53–70.
- Romano, J. P., & Wolf, M. (2005). Stepwise Multiple Testing as Formalized Data Snooping. *Econometrica*, 73(4), 1237–1282.
- Sheppard, K. (2025). *bashtage/arch* (Version 8.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.593254

## License
This product is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.
