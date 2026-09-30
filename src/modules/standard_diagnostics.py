import scipy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2 as _chi2, norm as _norm
from scipy.stats import skew as _skew, kurtosis as _kurtosis
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.tools.validation import (string_like,
                                          array_like,
                                          float_like,
                                          int_like,
                                          )

## Set global parameters
plt.rc('figure', figsize=(16,6))
sns.set_style('darkgrid')

## Moment conditions test via the CUSUM of Squares test
'''
The time-dependent test statistic for the CUSUM test is:
    phi(t) = 1/np.sqrt(T*v_hat) \Sum_{i=1}^t (x_i^2 - \mu_hat_2,T),
    where \mu_hat_2,T is the sample variance using the full sample,
    and v_hat is an estimate of the asymptotic variance of
    1/np.sqrt(T) \Sum_{i=1}^t (x_i^2 - \mu_hat_2,T).
    
    phi(t) is asymptotically normally distributed as N(0, t/T(1-t/T))
    
Therefore, phi(t) is essentially being modeled as a Brownian motion. We will use 
the variance of the asymptotic distribution to construct the confidence intervals
at any time t.
'''
## Moment conditions test via the trimmed SupF critical values of
## Andrews (1993), approximated following Hansen (1997)
#
# Approximate asymptotic p-value coefficients for the SupF distribution with
# m = 1 tested parameter, at each trimming fraction pi_0 tabulated by Hansen:
#     p(SupF) = 1 - chi2( theta0 + theta1 * SupF | eta )
# Source: Hansen, B. E. (1997), Table 2 (m = 1 row).
__HANSEN_SUPF_M1__ = {
    0.01: (-1.79, 1.17, 4.5),
    0.05: (-1.39, 1.07, 3.6),
    0.15: (-0.99, 1.02, 3.0),
    0.25: (-0.73, 0.98, 2.5),
    0.35: (-0.50, 0.96, 2.1),
}   # trim -> (theta0, theta1, eta)

def cusum_supf_test(x: np.ndarray, moment: int, alpha: float | None = 0.05,
                trim: float | None = 0.15, bandwidth: int | None = 8,
                ax: plt.Axes | None = None):
    '''
    Parameters
    ----------
    x : np.ndarray
        Series to test (e.g. demeaned returns). Must be 1-dimensional.
    moment : int
        Which moment to test: 2 or 4.
    alpha : float | None, optional
        Significance level. Must be strictly between 0 and 1. The default is 0.05.
    trim : float | None, optional
        Trimming fraction pi_0 defining the search range
        [trim*T, (1-trim)*T]. Must be one of the values tabulated by Hansen
        (1997): 0.01, 0.05, 0.15, 0.25, 0.35. The default is 0.15.
    bandwidth : int | None, optional
        Bandwidth for the kernel variance estimator. The default is 8.
    ax : plt.Axes | None, optional
        If provided, plots the standardized path phi(t) together with the
        time-varying boundary implied by the SupF critical value. The default is None.

    Raises
    ------
    ValueError
        If ``moment`` is not 2 or 4, ``trim`` is not one of the tabulated
        values, ``alpha`` is not in (0, 1), or ``trim`` leaves no valid
        search range for the given sample size.

    Returns
    -------
    dict
        Dictionary with keys ``'SupF'`` (the SupF statistic), ``'Test Statistic'``
        (the Hansen-approximated chi-squared statistic), ``'P-value'`` (the Hansen
        approximate p-value), and ``'Break Point'`` (the index at which the SupF
        statistic is maximized).

    References
    ----------
    Andrews, D. W. K. (1993). Tests for parameter instability and structural
        change with unknown change point. Econometrica, 61(4), 821-856.
    Hansen, B. E. (1997). Approximate asymptotic p values for
        structural-change tests. Journal of Business & Economic Statistics,
        15(1), 60-67.

    '''
    
    x = array_like(x, 'x', ndim=1)
    moment = int_like(moment, 'moment')
    alpha = float_like(alpha, 'alpha')
    trim = float_like(trim, 'trim')
    bandwidth = int_like(bandwidth, 'bandwidth')
    
    if moment not in (2,4):
        raise ValueError('The moment conditions can only be tested for the second '
                         f'and fourth order moments and not for the moment order: {moment}')
    
    if alpha <= 0 or alpha >=1:
        raise ValueError('The significance level alpha has to be strictly between 0 and 1.')
        
    if trim not in __HANSEN_SUPF_M1__:
        raise ValueError(f'Unsupported value for trim={trim}. Has to be chosen from '
                         f'{sorted(__HANSEN_SUPF_M1__)}, the fractions tabulated in '
                         'Hansen, 1997, Table 2.')
        
    x = x**2 if moment == 4 else x
    nobs = x.shape[0]
    
    ## First estimate the sample variane
    mu_2T = 1/nobs * np.sum(x**2)
    
    ## Run loop to calculate the cumulative sum of squares
    phi = np.zeros(shape=nobs)
    for i in range(nobs):
        ws = np.sum(x[:i+1]**2 - mu_2T)
        phi[i] = 1/np.sqrt(nobs) * ws
        
    ## Now estimate the asymptotic variance v_hat
    eps = x**2 - mu_2T
    gamma_0 = np.sum(eps @ eps) / nobs
    lr_acv = gamma_0
    for i in range(1, bandwidth+1):
        gamma_i = np.sum( eps[i:] @ eps[:nobs -i] ) / nobs
        lr_acv += 2 * (1 - i / (bandwidth + 1)) * gamma_i
        
    phi *= 1/np.sqrt(lr_acv)
    
    
    '''
    We now compute the following test statistic to test for moment stability

        z = theta0 + theta1 * SupF, where

        SupF = sup_{trim*T <= t <= (1-trim)*T}  phi(t)^2 / (t/T * (1 - t/T))
        
    Under the null hypothesis, the asymptotic distribution of the test statistic is
    chi-squared with eta degrees of freedom:
        
                                    z ~ X^2(eta)

    the values for (theta0, theta1, eta) are tabulated by trimming fraction in Hansen
    (1997), Table 2.
    
    The p-value for the test statistic is hence given by:
        
        p(SupF) = 1 - chi2( theta0 + theta1 * SupF | eta )
    '''
    
    ## Restrict the serach to the rimmed range [trim*T, (1-trim)*T]
    lo = max(1, int(np.ceil(trim * nobs)))
    hi = min(nobs, int(np.floor( (1 - trim) * nobs) ) )
    ## Sanity check
    if lo > hi:
        raise ValueError(f'The trimming fraction trim={trim} leaves no valid search range '
                         f'for nobs={nobs}. Select a smaller value.')
        
    t_grid = np.arange(lo, hi + 1)
    frac = t_grid / nobs
    F_t = phi[t_grid -1]**2 / (frac * (1 - frac))
    
    ## Now find the Sup of F_t
    arg_local = int(np.argmax(F_t))
    supF = float(F_t[arg_local])
    breakpoint_time = int(t_grid[arg_local])
    
    theta0, theta1, eta = __HANSEN_SUPF_M1__[trim]
    z = theta0 + theta1 * supF
    critvalue = _chi2.cdf(z, df=eta)
    pvalue = float(np.clip(1 - critvalue, 0.0, 1.0))
        
    ## Return the plot of phi along with the confidence bands
    if ax is not None:
        supF_crit = (_chi2.ppf( 1 - alpha, df = eta ) - theta0) / theta1
        grid = np.arange(nobs+1)
        cv = np.sqrt(supF_crit * grid/nobs * (1 - grid/nobs))
        ax.plot(phi, color='blue')
        ax.plot(grid, cv, color='black', linestyle='--', alpha=0.75,
                label = f'{1-alpha:.0%} SupF boundary')
        ax.plot(grid, -cv, color='black', linestyle='--', alpha=0.75)
        ax.legend()
        ax.set_title(f'CUSUM SupF Test - Moment Order : {moment}')
        
    return {'SupF': supF,
            'Test Statistic': z,
            'P-value': pvalue,
            'Break Point': breakpoint_time}

## Tail Index Estimation via the MLE of a Type I Generalized Pareto Distribution
'''
According to the Pickands–Balkema–De Haan theorem, the asymptotic tail distribution
of a random variable with an unkown distribution is given by the Generalized Pareto Distribution:
    G_{k,sigma} (y) = 1 - exp(-y/sigma)                 if xi = 0
    G_{k,sigma} (y) = 1 - (1 + xi * y/sigma)^(-1/xi)     otherwise.
Note that the random variable in question y is centered by its threshold: u.
'''
class GenParetoMLE(GenericLikelihoodModel):
    '''
    Maximum-likelihood estimator for the shape (xi) and scale (sigma) parameters
    of a Generalized Pareto Distribution, subclassing
    ``statsmodels.base.model.GenericLikelihoodModel``.

    Parameters
    ----------
    endog : np.ndarray
        Exceedances over a threshold u, already centered (i.e. y = x - u for
        x > u). Must be 1-dimensional.

    References
    ----------
    Pickands, J. (1975). Statistical inference using extreme order statistics.
        The Annals of Statistics, 3(1), 119-131.
    '''

    def __init__(self, endog):
        '''
        Parameters
        ----------
        endog : np.ndarray
            Centered exceedances passed to the model. Must be 1-dimensional.
        '''
        self._endog = array_like(endog, 'endog', ndim=1)
        super().__init__(endog)

        ## Indicator controll for the fit method
        self._fit_called = False

    def nloglikeobs(self, params):
        '''
        Per-observation negative log-likelihood of the Generalized Pareto
        Distribution, evaluated at the given parameters.

        Parameters
        ----------
        params : array_like
            Length-2 array ``[xi, sigma]`` with the shape and scale parameters.

        Returns
        -------
        np.ndarray
            Array of shape ``(nobs,)`` with the negative log-density at each
            observation. Must stay per-observation (not summed) since
            ``GenericLikelihoodModel``'s ``score_obs``/OPG-covariance machinery
            differentiates it element-wise.
        '''
        endog = self._endog
        xi = params[0]
        sigma = params[1]
        ll = scipy.stats.genpareto.logpdf(endog, xi, loc=0, scale=sigma)
        return -ll

    def fit(self, start_params = None, maxiter: float | None = 1e2, **kwargs):
        '''
        Fit the Generalized Pareto Distribution via maximum likelihood.

        Parameters
        ----------
        start_params : array_like | None, optional
            Length-2 array ``[xi, sigma]`` used as the optimizer's starting
            point. Defaults to ``[0.5, 1]``, a reasonable start for a Type I
            GPD.
        maxiter : float | None, optional
            Maximum number of optimizer iterations. The default is ``1e2``.
        **kwargs
            Forwarded to ``GenericLikelihoodModel.fit`` (e.g. ``disp``,
            ``cov_type``).

        Returns
        -------
        statsmodels results instance
            The fitted model, as returned by ``GenericLikelihoodModel.fit``.
        '''
        if start_params is None:
            ## Reasonable start_params for Type I GPD
            start_params = [0.5, 1]
        self._fit_called = True
        return super().fit(start_params=start_params, maxiter=maxiter, **kwargs)
    
    ## Set paramter names for model output
    exog_names = ['xi_hat', 'sigma_hat']
    

def tail_index_test(z: np.ndarray, moment_order: float, xi_hat: float, sigma_hat: float,
              bandwidth: int | None = 10, trim_quantile: float | None = 0.99,
              ax: plt.Axes | None = None):
        '''
        Tests whether the moment condition of a specified order holds for the
        estimated tail index, via the asymptotic normality of the GPD shape MLE.

        Parameters
        ----------
        z : np.ndarray
            Centered exceedances (the same series used to fit ``GenParetoMLE``).
            Must be 1-dimensional.
        moment_order : float
            Moment order r being tested; implies the hypothesized shape
            xi = 1/r under the null. Must be strictly positive.
        xi_hat : float
            MLE-estimated GPD shape parameter (e.g. from
            ``GenParetoMLE(z).fit()``). Must satisfy the Fisher regularity
            condition ``xi_hat >= -0.5``.
        sigma_hat : float
            MLE-estimated GPD scale parameter, used only for the density plot.
        bandwidth : int | None, optional
            Number of histogram bins for the empirical density plot. The
            default is 10.
        trim_quantile : float | None, optional
            Empirical quantile of ``z`` used as the plot's x-axis cutoff, so a
            few rare extreme exceedances don't stretch the axis and squash the
            bulk of the mass near 0. Must be strictly between 0 and 1. The
            default is 0.99.
        ax : plt.Axes | None, optional
            If provided, plots the empirical density of the exceedances
            against the fitted Generalized Pareto density. The default is None.

        Raises
        ------
        ValueError
            If ``moment_order`` is not strictly positive, or ``trim_quantile``
            is not in (0, 1).
        Exception
            If ``xi_hat`` is less than -0.5, violating the Fisher regularity
            condition required for the asymptotic normality result below.

        Returns
        -------
        dict
            Dictionary with keys ``'Test Statistic'`` (the w statistic),
            ``'Critical Value'`` (the normal CDF at w), and ``'P-value'``.

        References
        ----------
        Pickands, J. (1975). Statistical inference using extreme order
            statistics. The Annals of Statistics, 3(1), 119-131.
        '''

        z = array_like(z, 'z', ndim=1)
        moment_order = float_like(moment_order, 'moment_order')
        xi_hat = float_like(xi_hat, 'xi_hat')
        sigma_hat = float_like(sigma_hat, 'sigma_hat')
        bandwidth = int_like(bandwidth, 'bandwidth', optional=True) # Kernel bandwidth for computing the histogram of the exceedances
        trim_quantile = float_like(trim_quantile, 'trim_quantile') # Quantile of z used as the plot's x-axis cutoff

        if moment_order <= 0:
            raise ValueError('Moment order must be strictly positive.')

        if trim_quantile <= 0 or trim_quantile > 1:
            raise ValueError('trim_quantile has to be strictly between 0 and 1.')
            
        if xi_hat < -0.5:
            raise Exception('The value of the estimated shape parameter is less than -0.5. '
                            'This violates the Fisher regularity conditions. Cannot run the test.')
                    
        '''
        The asymptotic distribution of the shape parameter xi_hat minus its hypothesized value
        xi = 1/r, where r is the moment order being tested is normal with variance (1+xi)^2:
            
                        sqrt(m) * (xi_hat - xi) ~ N(0,(1+xi)^2),
        
        where m is the number of exceedances, ovvero the size of the series passed to the function.
        This motivates the test statistic:
            
                        w = sqrt(m) * (xi_hat - xi) / (1+xi) ~ N(0,1)
        '''
        nexcess = z.shape[0]
        xi = 1 / moment_order
        w = np.sqrt(nexcess) * (xi_hat - xi) / (1+xi)
        
        critvalue = _norm.cdf(w)
        pvalue = float(np.clip(1 - critvalue, 0.0, 1.0))
        
        if ax is not None:
            ## Cap the displayed range at a high empirical quantile of z rather than
            ## z.max(), since extreme exceedances are rare and let a single outlier
            ## stretch the axis, squashing the bulk of the mass into a sliver near 0.
            ## This only affects the view (via set_xlim) -- the histogram/density
            ## values themselves are still computed over the full sample.
            cutoff = np.percentile(z, trim_quantile * 100)
            grid = np.linspace(0, cutoff, 200)
            gp_density = scipy.stats.genpareto.pdf(grid, xi_hat, loc=0, scale=sigma_hat)
            if xi_hat > 0:
                pd_type = 'Type I'
            elif xi_hat == 0:
                pd_type = 'Exponential'
            else:
                pd_type = 'Type II'

            ax.hist(z, bins=bandwidth, density=True, label='Empirical Density of Exceedances')
            ax.plot(grid, gp_density, label=f'Pareto {pd_type} Density - $\\hat{{\\xi}}$ = {xi_hat:.2f}')
            ax.set_xlim(0, cutoff)
            ax.legend()
            ax.set_title('Empirical Density vs. Generalized Pareto Density')
        
        return {'Test Statistic': w,
                'Critical Value': critvalue,
                'P-value': pvalue}    

## Test di Jarque-Bera
def jb_test(z: np.ndarray):
    '''
    Jarque-Bera test of normality, based on the sample skewness and kurtosis.

    Parameters
    ----------
    z : np.ndarray
        Series to test (e.g. model residuals or returns). Must be
        1-dimensional with at least two observations.

    Raises
    ------
    ValueError
        If ``z`` has fewer than two observations.

    Returns
    -------
    dict
        Dictionary with keys ``'Test Statistic'`` (the JB statistic),
        ``'Critical Value'`` (the chi-squared(2) CDF at the JB statistic), and
        ``'P-value'``.

    References
    ----------
    Jarque, C. M., & Bera, A. K. (1980). Efficient tests for normality,
        homoscedasticity and serial independence of regression residuals.
        Economics Letters, 6(3), 255-259.
    '''

    z = array_like(z, 'z', ndim=1)
    nobs = z.shape[0]
    if nobs < 2:
        raise ValueError('Input data must contain at least two observations.')
    
    sk = _skew(z, axis=0)
    kt = 3 + _kurtosis(z, axis=0)
    
    '''
    The Jarque-Bera test statistic is defined as:
        
            JB = T/6 * (S_hat^2 + (K_hat - 3)^2/4),
    
    where S_hat and K_hat are the sample estimated Skewness and Kurtosis of the sample data.
    Under the null hypothesis that the sample data is normally distributed, the JB test
    statistic has an asymptotic chi-squared distribution with two degrees of freedom:
        
                                        JB ~ X^2(2).
    '''
    ## Compute the JB test statistic
    jb = nobs/6 * (sk**2 + (kt - 3)**2/4 )
    critvalue = _chi2.cdf(jb, df=2)
    pvalue = float(np.clip( 1 - critvalue, 0.0, 1.0 ))
    
    return {'Test Statistic':jb,
            'Critical Value': critvalue,
            'P-value': pvalue}