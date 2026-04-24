"""Demonstrate how to calculate various linear regression estimates.

Copyright (C) 2019 Mikko Pitkanen

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.odr
import pystan
import statsmodels.formula.api as smf
import scipy as sc
from sklearn.decomposition import PCA


def bayes_ols_fit(xi, yi):
    """Perform non-weighted OLS regression using Bayesian inference.

    This model does not take into account the uncertainties in the data, for a
    model that properly takes into account the uncertainties see, for example,
    Stan manual
    https://mc-stan.org/docs/2_19/stan-users-guide/bayesian-measurement-error-model.html
    (valid on 2019-05-18)
    """
    # define stan model
    model = """
        data {
        int<lower=0> N;      // number of cases
        vector[N] x;
        vector[N] y;         // outcome (variate)
        real<lower=0> sigma; // outcome noise
        }

        parameters {
        real intercept;
        real slope;
        }

        model {
        y ~ normal(intercept + slope * x, sigma);
        }
        """

    n = len(xi)
    ind = np.arange(n)

    # formalize input data
    data = {
        'N': n,
        'x': xi[ind],
        'y': yi[ind],
        'sigma': np.std(yi)}

    sm_OLS = pystan.StanModel(model_code=model)

    # make OLS fit to get initial guesses for slope and intercept
    slope_ols, intercept_ols = np.polyfit(x, y, 1)

    fit = sm_OLS.sampling(
        data=data,
        iter=1000,
        chains=4,
        init=lambda: {
            'x': xi[ind],
            'y': yi[ind],
            'slope': slope_ols,
            'intercept': intercept_ols},
        algorithm="NUTS",
        n_jobs=4)

    # find the index for maximum a posteriori (MAP) values
    samples = fit.extract(permuted=True)
    lp = samples['lp__']
    MAPindex = np.argmax(lp)

    # use MAP values for slope and intercept
    slope = samples['slope'][MAPindex]
    intercept = samples['intercept'][MAPindex]

    return slope, intercept


def bivariate_fit(xi, yi, dxi, dyi, ri=0.0, b0=1.0, maxIter=1e6):
    """Perform bivariate regression by York et al. 2004.

    This is an implementation of the line fitting algorithm presented in:
    York, D et al., Unified equations for the slope, intercept, and standard
    errors of the best straight line, American Journal of Physics, 2004, 72,
    3, 367-375, doi = 10.1119/1.1632486

    See especially Section III and Table I. The enumerated steps below are
    citations to Section III

    Parameters:
    xi, yi      np.array, x and y values
    dxi, dyi    np.array, errors for the data points xi, yi
    ri          float, correlation coefficient for the weights
    b0          float, initial guess for slope
    maxIter     float, maximum allowed number of iterations. this is to escape
                possible non-converging iteration loops

    Returns:
    b           slope estimate
    a           intercept estimate
    S           goodness-of-fit estimate
    cov         covariance matrix of the estimated slope and intercept values

    """
    # (1) Choose an approximate initial value of b
    # make OLS fit to get the initial guesses for slope
    slope_ols, intercept_ols = np.polyfit(x, y, 1)
    b = slope_ols

    # (2) Determine the weights wxi, wyi, for each point.
    wxi = 1.0 / dxi**2.0
    wyi = 1.0 / dyi**2.0

    alphai = (wxi * wyi)**0.5
    b_diff = 999.0

    # tolerance for the fit, when b changes by less than tol for two
    # consecutive iterations, fit is considered found
    tol = 1.0e-8

    # iterate until b changes less than tol
    iIter = 1
    while (abs(b_diff) >= tol) & (iIter <= maxIter):

        b_prev = b

        # (3) Use these weights wxi, wyi to evaluate Wi for each point.
        Wi = (wxi * wyi) / (wxi + b**2.0 * wyi - 2.0*b*ri*alphai)

        # (4) Use the observed points (xi ,yi) and Wi to calculate x_bar and
        # y_bar, from which Ui and Vi , and hence betai can be evaluated for
        # each point
        x_bar = np.sum(Wi * xi) / np.sum(Wi)
        y_bar = np.sum(Wi * yi) / np.sum(Wi)

        Ui = xi - x_bar
        Vi = yi - y_bar

        betai = Wi * (Ui / wyi + b*Vi / wxi - (b*Ui + Vi) * ri / alphai)

        # (5) Use Wi, Ui, Vi, and betai to calculate an improved estimate of b
        b = np.sum(Wi * betai * Vi) / np.sum(Wi * betai * Ui)

        # (6) Use the new b and repeat steps (3), (4), and (5) until successive
        # estimates of b agree within some desired tolerance tol
        b_diff = b - b_prev

        iIter += 1

    # (7) From this final value of b, together with the final x_bar and y_bar,
    # calculate a from
    a = y_bar - b * x_bar

    # Goodness of fit
    S = np.sum(Wi * (yi - b*xi - a)**2.0)

    # (8) For each point (xi, yi), calculate the adjusted values xi_adj
    xi_adj = x_bar + betai

    # (9) Use xi_adj, together with Wi, to calculate xi_adj_bar and thence ui
    xi_adj_bar = np.sum(Wi * xi_adj) / np.sum(Wi)
    ui = xi_adj - xi_adj_bar

    # (10) From Wi , xi_adj_bar and ui, calculate sigma_b, and then sigma_a
    # (the standard uncertainties of the fitted parameters)
    sigma_b = np.sqrt(1.0 / np.sum(Wi * ui**2))
    sigma_a = np.sqrt(1.0 / np.sum(Wi) + xi_adj_bar**2 * sigma_b**2)

    # calculate covariance matrix of slope and intercept (York et al.,
    # Section II)
    cov = -xi_adj_bar * sigma_b**2
    # [[var(b), cov], [cov, var(a)]]
    cov_matrix = np.array(
        [[sigma_b**2, cov], [cov, sigma_a**2]])

    if iIter <= maxIter:
        return b, a, S, cov_matrix
    else:
        print("bivariate_fit() exceeded maximum number of iterations, " +
              "maxIter = {:}".format(maxIter))
        return np.nan, np.nan, np.nan, np.nan


def deming_fit(xi, yi):
    """Perform Deming regression.

    Nomenclature follows:
    Francq, Bernard G., and Bernadette B. Govaerts. 2014. "Measurement Methods
    Comparison with Errors-in-Variables Regressions. From Horizontal to
    Vertical OLS Regression, Review and New Perspectives." Chemometrics and
    Intelligent Laboratory Systems. Elsevier.
    doi:10.1016/j.chemolab.2014.03.006.

    Parameters:
    xi, yi      np.array, x and y values

    Returns:
    slope       regression slope estimate
    intercept   regression intercept estimate

    """
    Sxx = np.sum((xi - np.mean(xi)) ** 2)
    Syy = np.sum((yi - np.mean(yi)) ** 2)
    Sxy = np.sum((xi - np.mean(xi)) * (yi - np.mean(yi)))
    lambda_xy = (np.var(yi) / np.size(yi)) / (np.var(xi) / np.size(xi))

    slope = (Syy - lambda_xy * Sxx +
             np.sqrt((Syy - lambda_xy * Sxx) ** 2 +
                     4 * lambda_xy * (Sxy ** 2))) / (2 * Sxy)

    intercept = np.mean(yi) - slope * np.mean(xi)

    return slope, intercept


def odr_fit(xi, yi, dxi, dyi):
    """Perform weighted orthogonal distance regression.

    https://docs.scipy.org/doc/scipy/reference/odr.html (valid on 2019-04-16)

    Parametes:
    xi, yi      np.array, x and y values
    dxi, dxy    np.array, x and y errors

    Returns:
    slope       regression slope estimate
    intercept   regression intercept estimate
    """
    def f(B, x):
        """Define linear function y = a * x + b for ODR.

        Parameters:
        B   [slope, intercept]
        x   x values
        """
        return B[0] * x + B[1]

    # define the model for ODR
    linear = scipy.odr.Model(f)

    # formalize the data
    data = scipy.odr.RealData(
        xi,
        yi,
        sx=dxi,
        sy=dyi)

    # make OLS fit to get initial guesses for slope and intercept
    slope_ols, intercept_ols = np.polyfit(x, y, 1)

    # instantiate ODR with your data, model and initial parameter estimate
    # use OLS regression coefficients as initial guess
    odr = scipy.odr.ODR(
        data,
        linear,
        beta0=[slope_ols, intercept_ols])

    # run the fit
    output = odr.run()
    slope, intercept = output.beta

    return slope, intercept


def pca_fit(xi, yi):
    """Estimate principal component regression fit to xi, yi data.

    See eg. https://shankarmsy.github.io/posts/pca-vs-lr.html (valid on
    2019-04-17)

    Parameters:
      xi, yi      x and y data points

    Returns:
      a           y-intercept, y = a + bx
      b           slope

    Example:
    [slope, intercept] = pca_fit( xi, yi, dxi, dyi, ri, b0 )
    """
    #

    xy = np.array([xi, yi]).T

    pca = PCA(n_components=1)
    xy_pca = pca.fit_transform(xy)

    xy_n = pca.inverse_transform(xy_pca)
    slope = (xy_n[0, 1] - xy_n[1, 1])/(xy_n[0, 0] - xy_n[1, 0])
    intercept = xy_n[0, 1] - slope * xy_n[0, 0]

    return slope, intercept


def quantile_fit(xi, yi, q=0.5):
    """Perform quantile regression.

    See for instance:
    https://www.statsmodels.org/dev/examples/notebooks/generated/quantile_regression.html
    (valid on 2091-04-16)

    Parametes:
    xi, yi      np.array, x and y values

    Returns:
    slope       regression slope estimate
    intercept   regression intercept estimate
    """
    data = {'xi': xi, 'yi': yi}

    df = pd.DataFrame.from_dict(data=data)

    mod = smf.quantreg('yi ~ xi', df)
    res = mod.fit(q=q)

    # return slope, intercept, covariance_matrix
    return res.params['xi'], res.params['Intercept'], res.cov_params().values


if __name__ == "__main__":
    """Demonstrate statistical inference using linear regression line fitting.

    Purpose: make linear regression with different estimators on x and y data
    with uncertainties. The correct linear model is y = 1.5x - 2.0. Try and
    answer:
    - which of the estimators is the best and why?
    - which of the methods consider x and y uncertainty and which ones don't?

    Also, try to replace x, y, dx and dy with your own data.

    Please note, that some methods, like OLS, have limitations that make it
    unsuitable/not optimal for this particular task!

    The script has been tested in a conda environment (see
    https://www.anaconda.com/distribution/#download-section for more info on
    that):
    conda create --name ito python=3.7 numpy==1.16.2 matplotlib==3.0.3 pandas==0.24.2 ipython==7.4.0 pystan==2.17.1.0 scipy==1.2.1 scikit-learn==0.20.3 statsmodels==0.9.0

    On linux you can run this from the command line by calling:
    python regression_estimators.py

    You can run this in Ipython by calling:
    run regression_estimators.py
    """
    # define absolute and relative standard uncertainties for x and y data sets
    sigma_x_abs = 0.1
    sigma_x_rel = 0.05
    sigma_y_abs = 0.1
    sigma_y_rel = 0.05

    # define the real regression parameters
    slope_true=1.5
    intercept_true=-2

    # define test data set points. These were generated with
    # 1. create normally distributed and independent random x values
    # 2. y = slope_true * x + intercept_true
    # 3. add normally distributed and independent random noise to x and y
    x = np.array([
        7.02490389, 5.84673882, 6.22362901, 5.89447501, 6.50522957,
        5.80298616, 6.17497626, 6.57451761, 6.47010046, 6.04077582,
        5.90118102, 6.77270208, 6.43396724, 6.41136767, 6.12493598,
        5.90716534, 6.32037763, 7.39491283, 6.36049059, 5.9670787 ,
        6.85141919, 6.26910599, 6.20254179, 6.9836126 , 6.63848388,
        6.21000692, 6.23215349, 6.2068118 , 6.39700798, 5.68460809,
        6.0957604 , 5.93433827, 6.92329796, 6.87485541, 6.64441035,
        6.5876272 , 6.21395565, 6.97018765, 5.8405509 , 6.68689768,
        6.55696236, 5.91300654, 5.77200607, 6.18620691, 6.46252992,
        5.84408498, 5.72175502, 6.28586177, 6.1426537 , 5.97624839,
        7.2909262 , 6.26629957, 6.35857082, 6.00486819, 5.96392117,
        6.79158893, 6.88007737, 5.79147038, 6.32788946, 5.89282374,
        5.246736  , 6.79574812, 6.57403906, 6.14307375, 7.00910025,
        5.7563269 , 6.351342  , 6.53075042, 5.71545834, 6.30847149,
        7.02490349, 6.40364356, 6.16509938, 6.4619477 , 6.70890128,
        6.51323415, 6.99526207, 5.98790113, 5.92062987, 6.07047262,
        7.05354862, 5.71384054, 6.60230794, 7.0169052 , 6.36480226,
        6.31785604, 5.61791288, 6.85937139, 5.75865116, 5.72959174,
        5.90952266, 6.42005849, 6.93056586, 6.01429019, 6.9796715 ,
        6.94304459, 6.75550702, 5.66799426, 6.98226771, 6.04554234])

    y = np.array([
        8.40544072, 7.23576875, 7.36491546, 6.0339046 , 7.97447602,
        7.15119055, 6.54041287, 7.17955333, 7.61995614, 7.45436687,
        6.51994675, 6.86866468, 7.93039991, 7.96141096, 6.74008807,
        6.24162408, 7.56592469, 8.90243085, 7.93080636, 7.69373893,
        8.1495254 , 7.31618462, 7.38623682, 8.27756635, 7.26490068,
        7.62419581, 8.09272363, 6.83289432, 7.00903454, 7.32198232,
        7.76544704, 7.86794507, 7.34049199, 7.16680021, 7.28097398,
        7.1300533 , 7.56470235, 8.53067913, 6.5722756 , 8.35793814,
        7.85134993, 6.28578289, 6.78504232, 7.46187614, 7.63509705,
        7.14787352, 7.76011323, 7.73277699, 6.61017633, 7.04707694,
        8.11976918, 7.57491045, 7.6502606 , 7.81891365, 7.5169907 ,
        7.10958076, 8.10664908, 6.41070742, 7.42201405, 7.1440822 ,
        6.71524939, 8.29542569, 7.40644049, 6.88359516, 8.10013957,
        6.17323241, 6.89164089, 8.18856187, 6.43704836, 7.1734189 ,
        7.33072932, 8.21214643, 7.73751715, 7.73084165, 8.5996884 ,
        8.08276146, 7.83624525, 7.24484867, 6.62742944, 5.95489133,
        8.05221471, 6.09695074, 8.934238  , 8.4620742 , 7.03271364,
        6.62512029, 7.76597935, 7.76624445, 6.84164444, 7.15060009,
        7.05616176, 7.62173155, 8.63441307, 6.77385575, 7.61571327,
        7.87055929, 8.07943385, 6.48806751, 7.88899205, 7.61359413])

    # estimate total standard uncertanties for each data point
    dx = np.sqrt((x * sigma_x_rel)**2 + sigma_x_abs**2)
    dy = np.sqrt((y * sigma_y_rel)**2 + sigma_y_abs**2)

    # calculate total relative uncertainty [%]
    u_x = np.mean(dx / x) * 100
    u_y = np.mean(dy / y) * 100

    parameters = dict()

    label = 'Truth: y={:1.2f}x{:+1.2f}'.format(slope_true, intercept_true)
    parameters = {label: (intercept_true, slope_true, np.nan, np.nan)}

    # bivariate, York et al 2004
    slope, intercept, S, cov = bivariate_fit(
        x, y, dx, dy, b0=0.0)
    label = 'York, 2004 :y={:1.2f}x{:+1.2f}'.format(slope, intercept)
    parameters.update({label: (intercept, slope, S, cov)})

    # OLS
    slope, intercept = np.polyfit(x, y, 1)
    S = np.nan
    cov = np.nan
    label = 'OLS: y={:1.2f}x{:+1.2f}'.format(slope, intercept)
    parameters.update({label: (intercept, slope, S, cov)})

    # Bayes OLS.
    slope, intercept = bayes_ols_fit(x, y)
    S = np.nan
    cov = np.nan
    label = 'Bayes OLS: y={:1.2f}x+{:1.2f}'.format(slope, intercept)
    parameters.update({label: (intercept, slope, S, cov)})

    # Deming
    slope, intercept = deming_fit(x, y)
    S = np.nan
    cov = np.nan
    label = 'Deming: y={:1.2f}x{:+1.2f}'.format(slope, intercept)
    parameters.update({label: (intercept, slope, S, cov)})

    # ODR, weighted orthogonal distance regression
    slope, intercept = odr_fit(x, y, dx, dy)
    S = np.nan
    cov = np.nan
    label = 'ODR: y={:1.2f}x{:+1.2f}'.format(slope, intercept)
    parameters.update({label: (intercept, slope, S, cov)})

    # quantile
    slope, intercept, cov = quantile_fit(x, y)
    S = np.nan
    label = 'Quantile: y={:1.2f}x+{:1.2f}'.format(slope, intercept)
    parameters.update({label: (intercept, slope, S, cov)})

    # pca
    slope, intercept = pca_fit(x, y)
    S = np.nan
    cov = np.nan
    label = 'PCA: y={:1.2f}x{:+1.2f}'.format(slope, intercept)
    parameters.update({label: (intercept, slope, S, cov)})

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)

    # plot error bar
    ax.errorbar(
        x, y,
        xerr=dx, yerr=dy,
        fmt='o',
        errorevery=5,
        linestyle='None',
        marker='.',
        ecolor='k',
        elinewidth=0.5,
        barsabove=True,
        label=None)

    xlim = np.array(ax.get_xlim())
    ylim = np.array(ax.get_ylim())

    for label, (intercept, slope, S, cov) in parameters.items():
        ax.plot(xlim, slope*xlim + intercept, '-', label=label)

    plt.suptitle('ITO Homework Fig. 1, synthetic data\nThe correct relationship is y={:1.2f}x{:+1.2f}'.format(slope_true, intercept_true), fontsize=16)

    ax.set_xlabel('x, total uncertainty {:1.0f}%'.format(u_x))
    ax.set_ylabel('y, total uncertainty {:1.0f}%'.format(u_y))

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(b=True)
    ax.legend(loc='lower right')
    plt.show()
