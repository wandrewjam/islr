import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.nonparametric.smoothers_lowess import lowess
from patsy import dmatrices


def main():
    boston = sm.datasets.get_rdataset('Boston', 'MASS')
    boston_df = boston.data

    y, X = dmatrices('medv ~ lstat', data=boston_df, return_type='dataframe')
    mod = sm.OLS(y, X)
    res = mod.fit()

    plot_single_regression(res)
    plot_regression_diagnostics(res)


def plot_single_regression(res):
    x_data, y_data = res.model.data.exog[:, 1], res.model.data.endog
    x_plot = np.array([[1, min(x_data)],
                       [1, max(x_data)]])

    fig, ax = plt.subplots()
    ax.scatter(x_data, y_data, edgecolors='k', facecolors='none')
    ax.plot(x_plot[:, 1], res.predict(x_plot), 'r', linewidth=2)
    ax.set_xlabel(res.model.exog_names[1])
    ax.set_ylabel(res.model.endog_names)

    plt.show()


def plot_regression_diagnostics(res):
    fig, ax = plt.subplots(2, 2, figsize=[8, 8])

    # Create the Residuals vs. Fitted plot
    residual_v_fitted(ax[0, 0], res)
    # Create the Q-Q plot
    qq_plot(ax[0, 1], res)
    # Create the Scale-Location plot
    scale_location(ax[1, 0], res)
    # Create the Residuals vs. Leverage plot
    residuals_v_leverage(ax[1, 1], res)

    plt.tight_layout()
    plt.show()


def residual_v_fitted(ax, res):
    residuals = res.resid
    fitted = res.fittedvalues
    smoothed = lowess(residuals, fitted)
    top3 = abs(residuals).sort_values(ascending=False)[:3]

    ax.scatter(fitted, residuals, edgecolors='k', facecolors='none')
    ax.plot(smoothed[:, 0], smoothed[:, 1], color='r')
    ax.axhline(0, color='k', linestyle=':', alpha=.5)
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs Fitted')
    for i in top3.index:
        ax.annotate(i, xy=(fitted[i], residuals[i]))


def qq_plot(ax, res):
    sorted_student_residuals = pd.Series(
        res.get_influence().resid_studentized_internal)
    sorted_student_residuals.index = res.resid.index
    sorted_student_residuals = sorted_student_residuals.sort_values(
        ascending=True)

    df = pd.DataFrame(sorted_student_residuals)
    df.columns = ['sorted_student_residuals']
    df['theoretical_quantiles'] = stats.probplot(
        df['sorted_student_residuals'], dist='norm', fit=False)[0]
    rankings = abs(df['sorted_student_residuals']).sort_values(
        ascending=False)
    top3 = rankings[:3]
    x = df['theoretical_quantiles']
    y = df['sorted_student_residuals']

    ax.scatter(x, y, edgecolors='k', facecolors='none')
    ax.set_title('Normal Q-Q')
    ax.set_xlabel('Theoretical quantiles')
    ax.set_ylabel('Standardized residuals')
    ax.plot([np.min([x, y]), np.max([x, y])],
            [np.min([x, y]), np.max([x, y])], color='r', ls='--')
    for val in top3.index:
        ax.annotate(val, xy=(
            df['theoretical_quantiles'].loc[val],
            df['sorted_student_residuals'].loc[val]
        ))


def scale_location(ax, res):
    fitted = res.fittedvalues
    student_residuals = res.get_influence().resid_studentized_internal
    sqrt_student_residuals = pd.Series(np.sqrt(np.abs(student_residuals)))
    sqrt_student_residuals.index = res.resid.index
    smoothed = lowess(sqrt_student_residuals, fitted)
    top3 = abs(sqrt_student_residuals).sort_values(ascending=False)[:3]

    ax.scatter(fitted, sqrt_student_residuals,
               edgecolors='k', facecolors='none')
    ax.plot(smoothed[:, 0], smoothed[:, 1], color='r')
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('$\\sqrt{|\\mathrm{Standardized \\; residuals}|}$')
    ax.set_title('Scale-Location')
    ax.set_ylim(0, max(sqrt_student_residuals) + 0.1)
    for i in top3.index:
        ax.annotate(i, xy=(fitted[i], sqrt_student_residuals[i]))


def residuals_v_leverage(ax, res):
    student_residuals = pd.Series(
        res.get_influence().resid_studentized_internal)
    student_residuals.index = res.resid.index
    df = pd.DataFrame(student_residuals)
    df.columns = ['student_residuals']
    df['leverage'] = res.get_influence().hat_matrix_diag
    df['cooks_d'] = res.get_influence().cooks_distance[0]
    model_cooks = df['cooks_d'].sort_values(ascending=False)
    top3 = model_cooks[:3]
    smoothed = lowess(df['student_residuals'], df['leverage'])

    x = df['leverage']
    y = df['student_residuals']

    ax.scatter(x, y, edgecolors='k', facecolors='none')
    ax.plot(smoothed[:, 0], smoothed[:, 1], color='r')
    ax.set_title('Residuals vs Leverage')
    ax.set_xlabel('Leverage')
    ax.set_ylabel('Standardized residuals')
    ax.set_xlim()
    ax.set_ylim()

    # top3 = np.flip(np.argsort(res.get_influence().cooks_distance[0]), 0)[:3]
    for val in top3.index:
        ax.annotate(val, xy=(x.loc[val], y.loc[val]))
    plot_cooks_distance(ax, res, x)


def plot_cooks_distance(ax, res, x):
    xpos = max(x) * 1.01
    cooksx = np.linspace(min(x), xpos, 50)
    p = len(res.params)
    poscooks1y = np.sqrt((p * (1 - cooksx)) / cooksx)
    poscooks05y = np.sqrt(0.5 * (p * (1 - cooksx)) / cooksx)
    negcooks1y = -np.sqrt((p * (1 - cooksx)) / cooksx)
    negcooks05y = -np.sqrt(0.5 * (p * (1 - cooksx)) / cooksx)
    ax.plot(cooksx, poscooks1y, label='Cook\'s Distance', ls=':',
            color='r')
    ax.plot(cooksx, poscooks05y, ls=':', color='r')
    ax.plot(cooksx, negcooks1y, ls=':', color='r')
    ax.plot(cooksx, negcooks05y, ls=':', color='r')
    ax.axvline(0, ls=':', alpha=.3, color='k')
    ax.axhline(0, ls=':', alpha=.3, color='k')
    ylo, yhi = ax.get_ylim()
    for (ypos, text) in zip(
            [poscooks1y[-1], poscooks05y[-1], negcooks05y[-1], negcooks1y[-1]],
            ['1.0', '0.5', '0.5', '1.0']
    ):
        if ylo < ypos < yhi:
            ax.annotate(text, xy=(cooksx[-1], ypos), color='r')
    ax.legend()


if __name__ == '__main__':
    main()
