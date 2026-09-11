import warnings
import numpy as np
import pandas as pd
from arch.univariate.base import ARCHModel as _ARCHModel
from statsmodels.tools.validation import (string_like,
                                          array_like,
                                          bool_like,
                                          float_like,
                                          int_like,
                                          )
class Validator:
    '''
    Rolling-window forecast-validation framework for one or more ARCH/GARCH-class
    conditional-volatility models.

    Instantiate with the return series and candidate models, then call
    ``validate()`` to run the rolling-window estimation/forecasting loop. Each
    model is re-estimated on a rolling window of ``window_size`` observations,
    produces a ``horizon``-step-ahead conditional variance forecast at each
    origin, and is evaluated against the realized squared return at the
    target date via the MSE, MAE, and QLIKE loss functions. If ``alpha`` is
    passed to ``validate()``, Value at Risk and Expected Shortfall forecasts
    are additionally computed for each window and each significance level in
    ``alpha``, using the model's own conditional distribution.

    Parameters
    ----------
    endog : np.ndarray
        Return series (e.g. demeaned log returns) the models were built on.
        Must not contain NaN values. If a ``pd.Series`` or ``pd.DataFrame``,
        its index is preserved and used to label the output.
    models : np.ndarray
        1-D object array of model instances (as returned by ``arch_model()``
        or constructed via ``ZeroMean``/``ARX`` plus a volatility process and
        a distribution). Each element must be an instance of
        ``arch.univariate.base.ARCHModel``.

    Attributes
    ----------
    forecasts : pd.DataFrame
        h-step-ahead conditional variance forecast per model, one column per
        model. ``None`` until ``validate()`` is called.
    std_residuals : pd.DataFrame
        Standardized residuals from the last observation in each estimation
        window, one column per model. Always 'origin' aligned. ``None``
        until ``validate()`` is called.
    model_fits : np.ndarray
        Object array holding, for each model, the fitted ``ARCHModelResult``
        from the last rolling window only. ``None`` until ``validate()`` is
        called. If ``update_frequency`` was passed to ``validate()`` and the
        last window was not a re-estimation window, this holds the
        ``ARCHModelFixedResult`` produced by ``fix()`` for that window
        instead (same parameters as the last actual fit, just re-anchored to
        the final window).
    mse_loss : pd.DataFrame
        Squared-error loss series, (forecast - endog^2)^2, per model.
        ``None`` until ``validate()`` is called.
    mae_loss : pd.DataFrame
        Absolute-error loss series, |forecast - endog^2|, per model. ``None``
        until ``validate()`` is called.
    qlike_loss : pd.DataFrame
        Quasi-likelihood loss series, log(forecast) + endog^2/forecast, per
        model. ``None`` until ``validate()`` is called.
    value_at_risk : pd.DataFrame
        h-step-ahead conditional Value at Risk forecast. If ``alpha`` was
        passed to ``validate()``, columns are a ``(model, alpha)``
        ``pd.MultiIndex``, one column per model/significance-level
        combination. If ``alpha`` was not passed, all-zero with one column
        per model. ``None`` until ``validate()`` is called.
    expected_shortfall : pd.DataFrame
        h-step-ahead conditional Expected Shortfall forecast. Same column
        structure as ``value_at_risk``. ``None`` until ``validate()`` is
        called.

    Raises
    ------
    ValueError
        If ``endog`` contains NaN values.
    Exception
        If any element of ``models`` is not an instance of ``ARCHModel``.
    '''

    def __init__(self, endog: np.ndarray, models: np.ndarray):

        self._endog = array_like(endog, 'endog', ndim=2)
        self._models = array_like(models, 'models', ndim=1, dtype=object)
        self._nobs = endog.shape[0]
        self._nmodels = len(models)
        
        if isinstance(endog, pd.Series) or isinstance(endog, pd.DataFrame):
            self._index = endog.index
        else:
            self._index = np.arange(len(endog))
        
        ## Input validation
        if np.any(np.isnan(endog)):
            raise ValueError('Input data contains NaN values.')
            
        if not np.all([isinstance(md, _ARCHModel) for md in models]):
            raise Exception('Model inputs should be instances of the ARCHModel class '
                            'initiated by the arch_model() model constructor.')
        
        ## Initiate class params
        self.forecasts = None
        self.std_residuals = None
        self.model_fits = None
        self.mse_loss = None
        self.mae_loss = None
        self.qlike_loss = None
        self.value_at_risk = None
        self.expected_shortfall = None
        
    def __str__(self):
        '''Returns a compact summary of average MSE, MAE, and QLIKE loss per model.'''
        columns = [f'model_{i} ({md.volatility.name})' for i, md in enumerate(self._models)]

        summary = pd.DataFrame(
            {
                'MSE': np.asarray(self.mse_loss.mean(axis=0)),
                'MAE': np.asarray(self.mae_loss.mean(axis=0)),
                'QLIKE': np.asarray(self.qlike_loss.mean(axis=0)),
            },
            index=columns,
        ).round(6)

        header = (
            f"Validator: {len(self.forecasts)} evaluated window(s), "
            f"{self._nmodels} model(s)."
        )
        return f"{header}\n{summary.to_string()}"

    def __repr__(self):
        '''Returns an unambiguous string representation of the instance's shape parameters.'''
        return (
            f"{self.__class__.__name__}(nobs={self._nobs}, nmodels={self._nmodels})"
        )

    def validate(self, window_size: int, horizon: int, update_frequency: int = 1,
                 alpha: np.ndarray | None = None, align: str | None = 'origin'):
        '''
        Runs the rolling-window estimation/forecasting loop for each model.

        For each model, warm-starts the fit from the previous window's
        converged parameters (falling back to a cold start with a larger
        iteration budget on non-convergence), produces a ``horizon``-step-
        ahead conditional variance forecast, and, if ``alpha`` is passed, the
        corresponding Value at Risk and Expected Shortfall at every
        significance level in ``alpha``. Populates ``forecasts``,
        ``std_residuals``, ``value_at_risk``, ``expected_shortfall``,
        ``model_fits``, and the MSE/MAE/QLIKE loss series (via
        ``compute_loss``), converting all of them to index-aligned
        ``pd.DataFrame``s (via ``_to_pandas``) before returning.

        Parameters
        ----------
        window_size : int
            Number of observations in each rolling estimation window. Must
            be strictly less than the number of observations in ``endog``.
        horizon : int
            Forecast horizon h to evaluate. ``window_size + horizon`` must
            be strictly less than the number of observations in ``endog``.
        update_frequency : int, optional
            Re-estimate (via ``md.fit()``) only every ``update_frequency``-th
            window; on the windows in between, the model is re-anchored to
            the new window via ``md.fix()`` using the parameters from the
            most recent actual fit, at zero optimization cost — window 0 is
            always fit regardless of this setting, so a fresh set of
            parameters always seeds the first forecast. This trades staleness
            of the parameter estimates for speed: forecasts and standardized
            residuals still advance one window at a time (via ``fix()``'s
            ``first_obs``/``last_obs``), only the coefficients themselves are
            held fixed between re-estimations. Must be a positive integer.
            The default is 1, which re-estimates every window.
        alpha : float | array_like | None, optional
            Significance level(s) for Value at Risk and Expected Shortfall
            forecasting. May be a single float or a 1-D array/list of
            floats, each strictly between 0 and 1. If not passed,
            ``value_at_risk``/``expected_shortfall`` are left as all-zero,
            one column per model. If passed, VaR/ES are computed at every
            level and ``value_at_risk``/``expected_shortfall`` gain a
            ``(model, alpha)`` ``pd.MultiIndex`` column structure — one
            column per model/significance-level combination. The default is
            None.
        align : {'origin', 'target'}, optional
            Index alignment method for ``forecasts``, ``value_at_risk``,
            ``expected_shortfall``, and the loss series. ``'origin'`` labels
            each row by the last observation used to produce that forecast
            (the default behavior of the ``forecast()`` method in the
            ``arch`` package); ``'target'`` labels each row by the date
            being forecasted, letting the output be compared directly
            against ``endog`` without further index shifting. Standardized
            residuals are always ``'origin'`` aligned regardless of this
            setting, since they correspond to the last observation used to
            fit the model. The default is 'origin'.

        Raises
        ------
        ValueError
            If ``window_size`` or ``window_size + horizon`` is not strictly
            less than the number of observations in ``endog``, if any value
            in ``alpha`` is not strictly between 0 and 1, if ``align`` is
            not 'origin' or 'target', or if ``update_frequency`` is not a
            positive integer.
        '''
        models = self._models
        nobs = self._nobs
        nmodels = self._nmodels
        
        ## TODO: Add option for chossing between rolling, fixed, and one time window

        window_size = int_like(window_size, 'window_size')
        horizon = int_like(horizon, 'horizon')
        alpha = array_like(alpha, 'alpha', ndim=1, optional=True)
        align = string_like(align, 'align')
        update_frequency = int_like(update_frequency, 'update_frequency')

        if update_frequency < 1:
            raise ValueError('update_frequency must be a positive integer.')

        if window_size >= nobs:
            raise ValueError('The length of the window must be strictly less than the '
                             'number of observations.')
            
        if window_size + horizon >= nobs:
            raise ValueError('window_size + forecast_horizon must be strictly less than '
                             'the number of observations.')
            
        if alpha is not None and (np.any(alpha >= 1) or np.any(alpha <= 0)):
            raise ValueError('The values of alpha should be strictly between 0 and 1.')
            
        if align not in ('target', 'origin'):
            raise ValueError(f'Invalid index align input: "{align}". Choose either "origin" '
                             'for "origin" index alignment or "target" for target index alignment. '
                             'For more information refer to the docstring or the repo README.md file.')

        ## Effective range: T - W - h + 1
        eff_range = nobs - window_size - horizon + 1
        
        forecasts = np.zeros((eff_range, nmodels))
        std_residuals = np.zeros((eff_range, nmodels ))
        
        if alpha is not None:
            nalphas = alpha.shape[0]
            value_at_risk = np.zeros((eff_range, nmodels, nalphas))
            expected_shortfall = np.zeros((eff_range, nmodels, nalphas))
        else:
            value_at_risk = np.zeros((eff_range, nmodels))
            expected_shortfall = np.zeros((eff_range, nmodels))
        
        model_fits = []
        
        for j, md in enumerate(models):
            _method = 'analytic'
            _fit = None
            _warned_convergence = False

            for i in range(eff_range):
                ## Window 0 always gets a real fit, since there is no prior estimate
                ## to re-anchor to via fix(). Otherwise, only re-estimate every
                ## update_frequency-th window; in between, hold the parameters fixed
                ## at their last-estimated values and just re-anchor the model to the
                ## new window (via fix()'s first_obs/last_obs) at zero optimization cost.
                if i % update_frequency == 0:
                    _start_params = _fit.params if _fit is not None else None
                    _start_params = None

                    _fit = md.fit(first_obs=i, last_obs=window_size + i,
                                  starting_values=_start_params, disp=False)

                    ## A warm start from the previous window's params can put the optimizer
                    ## in a bad spot right after a sharp volatility swing, so it may fail to
                    ## converge. Refit from the model's default starting values with a larger
                    ## iteration budget instead of carrying a bad estimate forward as the next
                    ## window's warm start.
                    if _fit.convergence_flag != 0:
                        if not _warned_convergence:
                            warnings.warn(
                                f'Warm-started fit did not converge for model {j} '
                                f'({md.volatility.name}) at window ending index '
                                f'{window_size + i - 1}; refitting from default starting values.',
                                stacklevel=2,
                            )
                            _warned_convergence = True
                        _fit = md.fit(first_obs=i, last_obs=window_size + i,
                                      starting_values=None, options={'maxiter': 500}, disp=False)
                else:
                    _fit = md.fix(params=_fit.params, first_obs=i, last_obs=window_size + i)

                ## Try/Except handle for when analytical forecasts for horizon>1
                ## do not exist. I prefer the bootstrap method, maybe later change
                ## so that user can choose.
                try:
                    _fc = _fit.forecast(horizon=horizon, method=_method)
                except ValueError:
                    warnings.warn(
                        'Failed to produce analytic forecasts. Falling back to '
                        f'boot strap forecasts for model {md.volatility.name} at '
                        f'horizon={horizon}.',
                        stacklevel=2
                        )
                    _method = 'bootstrap'
                    _fc = _fit.forecast(horizon=horizon, method=_method)
                
                _fh = _fc.variance.iloc[0,-1]
                _std = _fit.std_resid.loc[_fit.std_resid.last_valid_index()]
                
                forecasts[i,j] = _fh
                std_residuals[i,j] = _std
                
                ## VaR and Expected Shortfall estimation
                if alpha is not None:
                    ## Frist retrieve distribution parameters for VaR and ES estimation
                    if md.distribution.num_params == 0:
                        dist_params = None
                    else:
                        dist_nparams = md.distribution.num_params
                        dist_params = _fit.params[-dist_nparams:]
                    
                    for k, a in enumerate(alpha):
                        _z_alpha = md.distribution.ppf(a, parameters=dist_params)
                        _var = _z_alpha * np.sqrt(_fh)
                        _es = md.distribution.partial_moment(1, _z_alpha, parameters=dist_params) / a * np.sqrt(_fh)
                
                        value_at_risk[i,j,k] = _var
                        expected_shortfall[i,j,k] = _es
                
            ## Save the fitted model at the last iteration
            model_fits.append(_fit)
            
        self.forecasts = forecasts
        self.std_residuals = std_residuals
        self.value_at_risk = value_at_risk
        self.expected_shortfall = expected_shortfall
        self.model_fits = np.asarray(model_fits, dtype=object)
        self.mse_loss = self.compute_loss(forecasts, window_size, horizon, 'mse')
        self.mae_loss = self.compute_loss(forecasts, window_size, horizon, 'mae')
        self.qlike_loss = self.compute_loss(forecasts, window_size, horizon, 'qlike')
        
        ## Prepare the output
        self._to_pandas(window_size, horizon, align, alpha)
        
    def compute_loss(self, forecasts: np.ndarray, window_size: int,
                     horizon: int, loss_function: str):
        '''
        Computes a single loss series between ``forecasts`` and the
        target-aligned realized squared returns.

        Parameters
        ----------
        forecasts : np.ndarray
            h-step-ahead conditional variance forecast, one column per
            model.
        window_size : int
            Window size used to produce ``forecasts``. Used to locate the
            first target-aligned observation in ``endog``.
        horizon : int
            Forecast horizon used to produce ``forecasts``. Used to locate
            the first target-aligned observation in ``endog``.
        loss_function : {'mse', 'MSE', 'mae', 'MAE', 'qlike', 'QLIKE'}
            Loss function to compute.

        Returns
        -------
        np.ndarray
            Loss series, same shape as ``forecasts``.

        Raises
        ------
        ValueError
            If ``loss_function`` is not one of the supported values.
        '''
        forecasts = array_like(forecasts, 'forecasts', ndim=2)
        window_size = int_like(window_size, 'window_size')
        horizon = int_like(horizon, 'horizon')
        loss_function = string_like(loss_function, 'loss_function')

        valid_loss_functions = ('mse', 'MSE', 'mae', 'MAE', 'qlike', 'QLIKE')

        if loss_function not in valid_loss_functions:
            raise ValueError('Unsupported loss function. Must be one of: '
                             f'{valid_loss_functions}')

        ## Align observations and forecasts for correct loss evaluation
        ## The first index that corresponds to the first target forecast is 
        ## Window_Size + Forecast_horizon - 1
        first_indx = window_size + horizon - 1
        endog = self._endog[first_indx:]

        if loss_function in ('mse', 'MSE'):
            loss = (forecasts - endog**2)**2
        elif loss_function in ('mae', 'MAE'):
            loss = np.abs(forecasts - endog**2)
        elif loss_function in ('qlike', 'QLIKE'):
            loss = ( np.log(forecasts) + endog**2/forecasts )
        else:
            raise NotImplementedError('Loss function not implemented.')
        
        return loss

    def _to_pandas(self, window_size: int, horizon: int, align: str,
                   alpha: np.ndarray | None = None):
        '''
        Converts the raw ndarray results from ``validate``/``compute_loss``
        into index-aligned ``pd.DataFrame``s, per the ``align`` setting.

        Parameters
        ----------
        window_size : int
            Window size used by ``validate()``, for index alignment.
        horizon : int
            Forecast horizon used by ``validate()``, for index alignment.
        align : {'origin', 'target'}
            Index alignment method; see ``validate()``.
        alpha : float | array_like | None, optional
            Significance level(s) used by ``validate()``. If passed,
            ``value_at_risk``/``expected_shortfall`` are reshaped from
            ``(eff_range, nmodels, nalphas)`` into a DataFrame with a
            ``(model, alpha)`` ``pd.MultiIndex`` column structure; if not
            passed, they are left as a plain one-column-per-model
            DataFrame. The default is None.
        '''
        index = self._index
        nobs = self._nobs
        nmodels = self._nmodels
        window_size = int_like(window_size, 'window_size')
        horizon = int_like(horizon, 'horizon')
        align = string_like(align, 'align')
        alpha = array_like(alpha, 'alpha', ndim=1, optional=True)
        
        ### Index alignment
        if align == 'origin':
            fc_index = index[ (window_size - 1) : -horizon]
        else:
            fc_index = index[ (window_size - 1 + horizon) :]
        
        std_index = index[window_size - 1: -horizon]
        
        ## Forecasts
        df = pd.DataFrame(self.forecasts, index=fc_index)
        self.forecasts = df
        
        ## Standardized residuals
        df = pd.DataFrame(self.std_residuals, index=std_index)
        self.std_residuals = df

        ## Value at Risk and Expected Shortfall
        var = self.value_at_risk
        exp = self.expected_shortfall
        if alpha is not None:
            eff_range = nobs - window_size - horizon + 1
            multi_cols = pd.MultiIndex.from_product([range(nmodels), alpha], names=['Model', 'alpha'])
                
            #### Value at Risk
            df = pd.DataFrame(var.reshape(eff_range, -1), columns=multi_cols, index=fc_index)
            self.value_at_risk = df
            
            #### Expected Shortfall
            df = pd.DataFrame(exp.reshape(eff_range, -1), columns=multi_cols, index=fc_index)
            self.expected_shortfall = df
            
        else:
            df = pd.DataFrame(var, index=fc_index)
            self.value_at_risk = df
            
            df = pd.DataFrame(exp, index=fc_index)
            self.expected_shortfall = df
        
        ## MSE
        df = pd.DataFrame(self.mse_loss, index=fc_index)
        self.mse_loss = df
        
        ## MAE
        df = pd.DataFrame(self.mae_loss, index=fc_index)
        self.mae_loss = df
    
        ## QLIKE
        df = pd.DataFrame(self.qlike_loss, index=fc_index)
        self.qlike_loss = df
