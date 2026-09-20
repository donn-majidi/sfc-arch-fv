import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import perf_counter
from datetime import timedelta

plt.rc('figure', figsize=(16,6))
plt.rc('savefig', dpi=90)
sns.set_style('darkgrid')

'''
Load Data from LSEG Workspace
'''
load_start = perf_counter()
spx = pd.read_excel('../data/sp500.xlsx', header=3)
load_elapsed = perf_counter() - load_start
print(f'Time elapsed: {timedelta(seconds=load_elapsed)}')
spx.rename(columns={'Name':'Date'}, inplace=True)
spx = spx.iloc[2:,:]
spx.Date = pd.to_datetime(spx.Date, format='%Y-%m-%d')
spx = spx.set_index('Date', drop=True)
spx = spx.astype(dtype=float)

## Use Regex to find company names
import re
azioni= spx.columns

r = re.compile('.*morgan', flags=re.I)
list(filter(r.match, azioni))

stx = spx['CISCO SYSTEMS']
#rs = np.log(stx).diff().dropna() * 100
rs = stx.pct_change().dropna() * 100

## Drop all rows where returns are 0
rs = rs[rs!=0]

start_date = '2016-03-01'
rs = rs.loc[start_date:]

## Import the desired Mean Model, Volatility Processes and Distributions from arch
from arch.univariate import ZeroMean
from arch.univariate import GARCH, EGARCH, APARCH
from arch.univariate.distribution import Normal, StudentsT, SkewStudent

#### Define volatility processes 
vol_GARCH_1 = GARCH(1,0,1)
vol_GARCH_2 = GARCH(2,0,2)
vol_AVGARCH_1 = GARCH(1,0,1,power=1)
vol_AVGARCH_2 = GARCH(2,0,2,power=1)

vol_GJR_1 = GARCH(1,1,1)
vol_GJR_2 = GARCH(2,1,2)
vol_GJR_3 = GARCH(2,2,2)
vol_TARCH_1 = GARCH(1,1,1,power=1)
vol_TARCH_2 = GARCH(2,1,2,power=1)
vol_TARCH_3 = GARCH(2,2,2,power=1)

vol_EGARCH_1 = EGARCH(1,0,1)
vol_EGARCH_2 = EGARCH(2,0,2)
vol_EGARCH_3 = EGARCH(1,1,1)
vol_EGARCH_4 = EGARCH(2,1,2)
vol_EGARCH_5 = EGARCH(2,2,2)

vol_APARCH_1 = APARCH(1,0,1)
vol_APARCH_2 = APARCH(2,0,2)
vol_APARCH_3 = APARCH(1,1,1)
vol_APARCH_4 = APARCH(2,1,2)
vol_APARCH_5 = APARCH(2,2,2)

''' 
Custom volatility processes are supported as well
They can be specified via arch VolatilityProcess abstract class:
    from arch.univariate.volatility import VolatilityProcess
'''

## collect processes in a single list

vol_processes = [vol_GARCH_1, vol_GARCH_2, vol_AVGARCH_1, vol_AVGARCH_2,
                 vol_GJR_1, vol_GJR_2, vol_GJR_3, vol_TARCH_1, vol_TARCH_2, vol_TARCH_3,
                 vol_EGARCH_1, vol_EGARCH_2, vol_EGARCH_3, vol_EGARCH_4, vol_EGARCH_5,
                 vol_APARCH_1, vol_APARCH_2, vol_APARCH_3, vol_APARCH_4, vol_APARCH_5]

#### Initiate distribution classes
dist_normal = Normal(seed=1776)
dist_studst = StudentsT(seed=1776)
dist_skewt = SkewStudent(seed=1776)

dists = [dist_normal, dist_studst, dist_skewt]

#### Now we construct all 60 models from all possible unique combinations
#### of vol processes and distributions
from itertools import product

proc_combs = set(product(vol_processes, dists))

#### Create all model instances
models = []
for _vol, _dist in proc_combs:
    _mod = ZeroMean(y=rs, volatility=_vol, distribution=_dist)
    models.append(_mod)

models = np.asarray(models, dtype=object)

#### Initiate Forecast-Validation of model instances via the Validator class
from src.models.validator import Validator

model_validator = Validator(endog=rs, models=models)

## Set the parameters to feed to the validate method
ws = 2016               ## Train on the first 8 years of data
fh = 1                  ## Produce 1-step ahead out of sample forecasts
uf = 21                 ## Update the models every 21 days
alpha = [0.01, 0.05]    ## Significance level for VaR and Expected Shortfall forecasts
align = 'target'        ## Align out-of-sample forecast indices to target date

validate_start = perf_counter()
model_validator.validate(window_size=ws, horizon=fh, update_frequency=uf, 
                         alpha=alpha, align=align)
validate_elapsed = perf_counter() - validate_start
print(f'Time elapsed: {timedelta(seconds=validate_elapsed)}')

## Make deep copies of the parameters
## These deep copies prove to be useful as some models might be eliminated 
## before running the StepM and MCS procedures without altering the class attributes.
md_forecasts = model_validator.forecasts.copy(deep=True)
md_residuals = model_validator.std_residuals.copy(deep=True)
md_mse = model_validator.mse_loss.copy(deep=True)
md_qlike = model_validator.qlike_loss.copy(deep=True)
md_var = model_validator.value_at_risk.copy(deep=True)
md_exp = model_validator.expected_shortfall.copy(deep=True)

## Align the indices of the test set and the forecasts for data visualization
rs_test = rs.loc[md_forecasts.index]

## Data Visualization : Returns vs. Forecasted Volatility Envelopes
fig, ax = plt.subplots()
ax.plot(rs_test, alpha=0.8, label='returns')
ax.plot(np.sqrt(md_forecasts), alpha=0.8)
ax.plot(-1*np.sqrt(md_forecasts), alpha=0.8)
ax.legend()
ax.set_title('Returns vs. Forecasted Volatility Envelopes')

#### StepM and MCS Comparison Porcedures
'''
To run multiple comparison procedures, first need to estiamte the optimal bootstrap
block sizes for the target loss series.

Multiple comparisons are based on QLIKE loss series, of course, any of the other
loss functions MSE or MAE may be used instead.
from src.modules.bootstrap_params import bootstrap_block_size
'''
from src.modules.bootstrap_params import bootstrap_block_size
opt_bs = bootstrap_block_size(md_qlike).max()
print(opt_bs)
opt_sb = np.ceil(opt_bs['Stationary Bootstrap'])
opt_cb = np.ceil(opt_bs['Circular Bootstrap'])
opt_mb = np.ceil(opt_bs['Moving-Blocks Bootstrap'])

######## StepM
#### For StepM, the benchmark is always the GARCH(1,1) specification
#### with Gaussian likelihood function.

## First need to find the model index corresponding to the GARCH(1,1) + Gaussian specification
bm = next(
    md for md in models
    if md.volatility == vol_GARCH_1 and md.distribution == dist_normal
    )
bm_index = np.where(models == bm)[0][0]

bm_losses = md_qlike.loc[:, bm_index]
alt_losses = md_qlike.loc[:, md_qlike.columns != bm_index]

from arch.bootstrap import StepM
stepm = StepM(benchmark=bm_losses, models=alt_losses, size=0.05, reps=10000,
              block_size=opt_sb, bootstrap='stationary', seed=1776)
stepm.compute()

######## MCS
from arch.bootstrap import MCS
mcs = MCS(losses=md_qlike, size=0.05, reps=10000,
          block_size=opt_sb, bootstrap='stationary', seed = 1776)
mcs.compute()

######## Visualizations
avg_qlike = pd.DataFrame(md_qlike.mean(), columns=["Average loss"])
fig = avg_qlike.plot(style=['o'])
fig.set_xlabel('Model Number')
fig.set_ylabel('QLIKE Score')

stepm_models = pd.concat([alt_losses.mean(), alt_losses.mean(), alt_losses.mean()], axis=1)
stepm_models.columns = ["Same or worse", "Superior", "Benchmark"]
sup = stepm_models.index.isin(stepm.superior_models)
worse = np.logical_not(sup)
stepm_models.loc[sup, "Same or worse"] = np.nan
stepm_models.loc[worse, "Superior"] = np.nan
stepm_models.loc[:, "Benchmark"] = np.nan
## Add benchmark back to the data frame
stepm_models.loc[bm_index, "Benchmark"] = bm_losses.mean()
fig = stepm_models.plot(style=["o", "s", "*"])
fig.set_xlabel('Model Number')
fig.set_ylabel('QLIKE Score')

status = pd.DataFrame(
    [md_qlike.mean(0), md_qlike.mean(0)], index=["Excluded", "Included"]
).T
status.loc[status.index.isin(mcs.included), "Excluded"] = np.nan
status.loc[status.index.isin(mcs.excluded), "Included"] = np.nan
fig = status.plot(style=["o", "s"])
fig.set_xlabel('Model Number')
fig.set_ylabel('QLIKE Score')

########## Diagnostic tests
test_size = 0.05        # size of the tests

#### Moment Conditions via The SupF CUSUM Test
from src.modules.standard_diagnostics import cusum_supf_test
moments = [2,4]           
moment_conditions = np.zeros(shape=(len(models), len(moments)))
for i in range(len(models)):
    for j in range(len(moments)):
        _test = cusum_supf_test(md_residuals[i], moments[j])
        pvalue = _test['P-value']
        if pvalue > test_size:
            moment_conditions[i,j] = 1
        else:
            moment_conditions[i,j] = 0

mcol_str = 'Moment Order_{0:0' + 'd}'
mcols = [mcol_str.format(i) for i in moments]
moment_conditions = pd.DataFrame(moment_conditions, columns=mcols)
print(moment_conditions)

#### Hill's Tail Index Estimator
from src.modules.standard_diagnostics import GenParetoMLE, hill_test
tail_indx = np.zeros(shape=(len(models), 2))
for i in range(len(models)):
    zs = md_residuals[i]
    thresh = np.percentile(zs, 95)
    exs = zs[zs > thresh] - thresh
    gp = GenParetoMLE(exs)
    gp_fit = gp.fit()
    xi_hat = gp_fit.params[0]
    sigma_hat = gp_fit.params[1]
    
    if xi_hat < 0:
        tail_indx[i,0] = np.round(xi_hat, decimals=2)
        tail_indx[i,1] = -1
        continue
    r = max(np.ceil(1/xi_hat), 4)
    tail_indx[i,0] = r
    
    _test = hill_test(exs, r, xi_hat, sigma_hat)
    pvalue = _test['P-value']
    if pvalue > test_size:
        tail_indx[i,1] = 1
    else:
        tail_indx[i,1] = 0

tcols = ['Tail Index', 'Hill Test']
tail_indx = pd.DataFrame(tail_indx, columns=tcols)
print(tail_indx)

#### Jarque-Bera Test
## This test can only be carried out for specifications with Normal distribution
from src.modules.standard_diagnostics import jb_test
md_indx = [i for i, md in enumerate(models) if md.distribution == dist_normal]

jb_norm = np.zeros(shape=(len(md_indx),1))
for i in range(len(md_indx)):
    zs = md_residuals[md_indx[i]]
    _test = jb_test(zs)
    pvalue = _test['P-value']
    if pvalue > test_size:
        jb_norm[i] = 1
    else:
        jb_norm[i] = 0
        
jbcols = ['JB Test']
jb_norm = pd.DataFrame(jb_norm, index=md_indx, columns=jbcols)
print(jb_norm)

######## VaR and ES Forecasts
