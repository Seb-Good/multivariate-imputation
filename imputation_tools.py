"""
imputation_tools.py
--------------------
This module provides classes and methods for imputing compositional data under the detection limit using
Expectation Maximization.
By: Sebastian D. Goodfellow, Ph.D.
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import numpy as np
import pandas as pd
from skbio.stats.composition import *
import scipy.stats as st


def em_imputation(df, detection_limits, delta=0.65, tolerance=0.0001, max_iter=50):

    """
    Imputes missing compositional data with values below the detection limit using expectation maximization.

    Parameters
    ----------
    df : pandas dataframe
        Original DataFrame of compositional data. Values below detection can be presented in any format. All columns
        must have their data in a common format (%, ppm, etc..). Columns must only include compositional data (No
        hole ID, from, to, etc..).
    detection_limits : pandas dataframe
        Detection limit for each column. Units must match those for input data (df).
    delta : float
        Delta parameter for multiplicative replacement.
    tolerance : float
        Tolerance threshold.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    df_imputed : pandas dataframe
        Original DataFrame including imputed missing values.
    """

    """
    Preliminary Setup
    """
    # Get copy of compositional data
    x = df.copy()

    # Set values below or at detection to NaN
    for col in df.columns:
        x[col][x[col] <= detection_limits.ix[0, col]] = np.nan

    # Get dimensions
    nn = x.shape[0]
    d = x.shape[1]

    # Format dl so that it is a matrix of dimension nn rows and D columns
    dl = pd.DataFrame(np.ones([nn, d]) * detection_limits.values, columns=x.columns)

    # Get the sum for each row excluding NA values
    c = x.sum(axis=1)

    # Check if data is closed
    close = 0
    if all(np.abs(c - np.mean(c)) < np.finfo(np.float64).eps**0.5):
        close = 1

    # Create DataFrame where <=LOD = 1 and >LOD =0
    misspat = pd.DataFrame(np.isnan(x)*1)

    # Concatenate
    misspat = pd.Series(np.array(misspat.astype(str).apply(''.join, axis=1)), dtype='category')

    # Rename categories
    misspat = misspat.cat.rename_categories(np.arange(1, len(np.unique(misspat))+1).tolist())

    """
    Imputation
    """
    # Get the column index of the first column which has no missing values
    pos = pd.Series(
        data=x.columns.get_loc(
            pd.Series(x.isnull().any() == False)[pd.Series(x.isnull().any() == False) == True].index[0]
        ),
        index=[pd.Series(x.isnull().any() == False)[pd.Series(x.isnull().any() == False) == True].index[0]]
    )

    # Check that at least one column has all values
    if not pd.Series(x.isnull().any() == False).any():
        print("lrEM based on alr requires at least one complete column")

    # Create cpoints matrix
    cpoints = np.log(dl.values) - np.log(x.ix[:, pos[0]].values).reshape(x.shape[0], 1) - np.finfo(np.float64).eps
    cpoints = pd.DataFrame(np.delete(cpoints, pos[0], axis=1))

    # Calculate the oblique additive log-ratio (alr) transformation
    x_alr = pd.DataFrame(np.log(x.values) - np.log(x.ix[:, pos].values), columns=x.columns).drop(pos.index[0], axis=1)

    # Get dimensions
    nn = x_alr.shape[0]
    d = x_alr.shape[1]

    # Multiplicative Replacement
    x_mr = multiplicative_replacement(x, dl, delta)

    # Calculate the oblique additive log-ratio (alr) transformation
    x_mr_alr = pd.DataFrame(
        np.log(x_mr.values) - np.log(x_mr.ix[:, pos].values), columns=x_mr.columns
    ).drop(pos.index[0], axis=1)

    # Calculate the mean of each column
    m = x_mr_alr.mean(axis=0)

    # Calculate covariance matrix
    cov = pd.DataFrame(np.cov(x_mr_alr, rowvar=0), index=x_mr_alr.columns, columns=x_mr_alr.columns)

    # Set variables
    iter_again = 1  # Set stopping criterion
    niters = 0  # Iteration count

    # EM imputing iterations
    while iter_again == 1:

        # Update iteration count
        niters += 1
        m_new = m.copy()
        cov_new = cov.copy()
        y = x_alr.copy()
        v = pd.DataFrame(np.zeros([d, d]))  # DataFrame of zeros (elements x elements)

        # Loop through misspat categories other than 1
        for npat in np.arange(2, len(np.unique(misspat))+1):

            # Get indices of misspat where the category is equal to npat
            i = misspat[misspat == npat].index

            # Find column labels without a missing value in x_alr
            columns_finite = pd.Series(np.isfinite(x_alr.ix[i[0], :]))
            varobs = pd.Series(
                data=[x_alr.columns.get_loc(col) for col in columns_finite[columns_finite == True].index],
                index=x_alr.columns[
                    [x_alr.columns.get_loc(col) for col in columns_finite[columns_finite == True].index]
                ]
            )

            # Find column labels with a missing value in x_alr
            columns_infinite = pd.Series(~np.isfinite(x_alr.ix[i[0], :]))
            varmiss = pd.Series(
                data=[x_alr.columns.get_loc(col) for col in columns_infinite[columns_infinite == True].index],
                index=x_alr.columns[
                    [x_alr.columns.get_loc(col) for col in columns_infinite[columns_infinite == True].index]
                ]
            )

            # Check to see if at least one column is finite
            if len(varobs) == 0:
                print("lrEM based on alr requires at least one complete column")

            # Set empty sigmas array
            sigmas = np.zeros(d)

            # Apply sweep
            b, cr = sweep(m, cov, varobs)

            y.ix[i, varmiss] = np.ones([len(i), 1])*b[0] + x_alr.ix[i, varobs].values.dot(b[1:len(varobs)+1])
            sigmas[varmiss] = np.sqrt(np.diag(cr))

            for j in range(len(varmiss)):

                sigma = sigmas[varmiss[j]]
                fd_n01 = st.norm.pdf((cpoints.ix[i, varmiss[j]] - y.ix[i, varmiss[j]]) / sigma)
                fdist_n01 = st.norm.cdf((cpoints.ix[i, varmiss[j]] - y.ix[i, varmiss[j]]) / sigma)
                y.ix[i, varmiss[j]] = y.ix[i, varmiss[j]] - sigma * (fd_n01 / fdist_n01)

            v.ix[varmiss, varmiss] += cr*len(i)

        m = y.mean(axis=0, skipna=False)
        dif = y - np.ones([nn, 1]) * m.values.T
        pc = pd.DataFrame(dif.values.T.dot(dif.values), index=dif.columns, columns=dif.columns)
        cov = (pc + v.values) / (nn - 1)

        # Convergence Check
        m_dif = np.max(np.abs(m - m_new))
        c_dif = np.max(np.max(np.abs(cov - cov_new)))

        # Stopping
        if np.max(np.array([m_dif, c_dif])) < tolerance or niters == max_iter:
            iter_again = 0

    # Compute inverse oblique additive log-ratio (alr) transformation
    y = inverse_alr(y)

    for i in range(nn):

        if x.ix[i, :].isnull().any():

            vbdl = pd.Series(x.ix[i, :].isnull())
            vbdl = vbdl[vbdl == True]
            x.ix[i, vbdl.index] = (x.ix[i, pos].values / y.ix[i, pos].values) * y.ix[i, vbdl.index].values

    # If data was closed, un-close it
    if close == 1:
        x = x.apply(lambda row: row / row.sum() * c[0], axis=1)

    return x


def multiplicative_replacement(df, dl, delta):

    """
    Imputes missing compositional data with values below the detection limit using multiplicative replacement.

    Parameters
    ----------
    df : pandas dataframe
        Original DataFrame of compositional data. Values below detection are presented as NaN. All columns
        must have their data in a common format (%, ppm, etc..). Columns must only include compositional data (No
        hole ID, from, to, etc..).
    dl : pandas dataframe
        Detection limit for each column. Units must match those for input data (df).
    delta : float
        Delta parameter for multiplicative replacement.

    Returns
    -------
    df_imputed : pandas dataframe
        Original DataFrame including imputed missing values.
    """

    # Copy DataFrame
    x_mr = df.copy()

    # Get dimensions of X
    nn = x_mr.shape[0]
    d = x_mr.shape[1]

    # Get sum of each row excluding NA vales
    c = x_mr.sum(axis=1)

    # Check if data is closed
    closed = 0
    if all(np.abs(c - np.mean(c)) < np.finfo(np.float64).eps**0.5):
        closed = 1

    # Set detection limits to matrix
    if len(dl) == 1:
        dl = pd.DataFrame(np.ones([nn, d])*dl.values, columns=x_mr.columns)

    # Set new variable
    y = x_mr.copy()
    x_temp = x_mr.copy()

    # Loop through rows
    for row_id in range(nn):

        # If any missing values in a row
        if x_mr.ix[row_id, :].isnull().any():

            # Get column indices of missing values
            z = x_mr.ix[row_id, :].isnull()[x_mr.ix[row_id, :].isnull() == True].index.tolist()

            # Apply multiplicative replacement
            y.ix[row_id, z] = delta * dl.ix[row_id, z]
            y.drop(z, axis=1).ix[row_id, :] = \
                (1 - np.sum(y.ix[row_id, z]) / c[row_id]) * x_mr.drop(z, axis=1).ix[row_id, :]
            x_mr.ix[row_id, z] = \
                (x_mr.drop(z, axis=1).ix[row_id, :][0] / y.drop(z, axis=1).ix[row_id, :][0]) * y.ix[row_id, z]

    # If data was closed, un-close it
    if closed == 1:
        x_mr = x_mr.apply(lambda row: row / row.sum() * c[0], axis=1)

    # Copy data for output
    df_imputed = x_mr.copy()

    return df_imputed


def inverse_alr(df):

    """
    Compute inverse oblique additive log-ratio (alr) transformation.

    Parameters
    ----------
    df : pandas dataframe
        Additive log-ratio transform of original DataFrame of compositional data.

    Returns
    -------
    df_inverse : pandas dataframe
        Additive log-ratio inverse.
    """

    ad = 1 / (np.sum(np.exp(df.values), axis=1) + 1)
    ax = pd.DataFrame(np.exp(df.values) * ad[:, np.newaxis], columns=df.columns)
    df_inverse = pd.concat([pd.DataFrame(ad, columns=['ad']), ax], axis=1)

    return df_inverse


def sweep(m, c, varobs):

    """
    Parameters
    ----------
    m :
    c :
    varobs :

    Returns
    -------
    b :
    cr :
    """

    # Get dimensions
    d = len(m)
    q = len(varobs)

    i = np.ones(d)
    i[varobs.values] -= 1
    dep = np.where(i != 0)[0]
    ndep = len(dep)

    a = np.zeros([d+1, d+1])
    a[0, 0] = -1

    a[0, 1:d+1] = m.values
    a[1:d+1, 0] = m.values
    a[1:d+1, 1:d+1] = c

    reor = np.append(np.append(np.zeros(1), varobs.values+1), dep+1).astype('int')
    a = a[:, reor]
    a = a[reor, :]

    ind = np.arange(q+1)

    # Get dimensions
    nn = a.shape[0]
    d = a.shape[1]

    s = a.copy()

    for j in ind:

        s[j, j] = -1 / a[j, j]

        for i in np.arange(d):
            if i != j:
                s[i, j] = -a[i, j] * s[j, j]
                s[j, i] = s[i, j]

        for i in np.arange(d):
            if i != j:
                for k in np.arange(d):
                    if k != j:
                        s[i, k] = a[i, k] - s[i, j] * a[j, k]
                        s[k, i] = s[i, k]

        a = s.copy()

    b = a[0:q+1, q+1:d+1]
    cr = a[q+1:d+1, q+1:d+1]

    return b, cr
