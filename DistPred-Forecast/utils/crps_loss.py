import torch
import numpy as np
import properscoring as ps

def move_axis_to_end(x, dim):
    #return np.rollaxis(array, axis, start=array.ndim)
    axis = list(range(len(x.shape)))
    del axis[dim]
    axis.append(dim)
    return x.permute(axis)

def argsort_indices(a, axis=-1):
    """Like argsort, but returns an index suitable for sorting the
    the original array even if that array is multidimensional
    """
    ind = list(np.ix_(*[np.arange(d) for d in a.shape]))
    ind[axis] = a.argsort(axis)
    return tuple(ind)

def _crps_ensemble_vectorized(observations, forecasts, weights=None):
    """
    An alternative but simpler implementation of CRPS for testing purposes

    This implementation is based on the identity:

    .. math::
        CRPS(F, x) = E_F|X - x| - 1/2 * E_F|X - X'|

    where X and X' denote independent random variables drawn from the forecast
    distribution F, and E_F denotes the expectation value under F.

    Hence it has runtime O(n^2) instead of O(n log(n)) where n is the number of
    ensemble members.

    Reference
    ---------
    Tilmann Gneiting and Adrian E. Raftery. Strictly proper scoring rules,
        prediction, and estimation, 2005. University of Washington Department of
        Statistics Technical Report no. 463R.
        https://www.stat.washington.edu/research/reports/2004/tr463R.pdf
    """
    if observations.ndim == forecasts.ndim - 1:
        # sum over the last axis
        assert observations.shape == forecasts.shape[:-1]
        observations = observations.unsqueeze(-1)
        if weights is None:
            results = torch.abs(forecasts - observations)
            score = torch.nanmean(results, -1)
            # forecasts_diff = (forecasts.unsqueeze(-1) - forecasts.unsqueeze(-2))
            # score += -0.5 * torch.nanmean(torch.abs(forecasts_diff), dim=(-2, -1))
            forecasts_diff = (forecasts.unsqueeze(-1) - forecasts.unsqueeze(-2)).abs()
            score += -0.5 * torch.nanmean(forecasts_diff, dim=(-2, -1))
            return score.abs().mean()

        weights = torch.where(~torch.isnan(forecasts), weights, torch.nan)
        weights = weights / torch.nanmean(weights, dim=-1, keepdims=True)
        results = weights * torch.abs(forecasts - observations)
        score = torch.nanmean(results, -1)
        # insert new axes along last and second to last forecast dimensions so
        # forecasts_diff expands with the array broadcasting
        forecasts_diff = (forecasts.unsqueeze(-1) - forecasts.unsqueeze(-2))
        weights_matrix = (weights.unsqueeze(-1) * weights.unsqueeze(-2))
        weight_forecasts = weights_matrix * torch.abs(forecasts_diff)
        score += -0.5 * torch.nanmean(weight_forecasts, dim=(-2, -1))
        return score.abs().mean()
    elif observations.ndim == forecasts.ndim:
        # there is no 'realization' axis to sum over (this is a deterministic
        # forecast)
        return torch.abs(observations - forecasts).mean()


def crps_ensemble(observations, forecasts, weights=None, issorted=False,
                  axis=-1):
    """
    Calculate the continuous ranked probability score (CRPS) for a set of
    explicit forecast realizations.

    The CRPS compares the empirical distribution of an ensemble forecast
    to a scalar observation. Smaller scores indicate better skill.

    CRPS is defined for one-dimensional random variables with a probability
    density $p(x)$,

    .. math::
        CRPS(F, x) = \int_z (F(z) - H(z - x))^2 dz

    where $F(x) = \int_{z \leq x} p(z) dz$ is the cumulative distribution
    function (CDF) of the forecast distribution $F$ and $H(x)$ denotes the
    Heaviside step function, where $x$ is a point estimate of the true
    observation (observational error is neglected).

    This function calculates CRPS efficiently using the empirical CDF:
    http://en.wikipedia.org/wiki/Empirical_distribution_function

    The Numba accelerated version of this function requires time
    O(N * E * log(E)) and space O(N * E) where N is the number of observations
    and E is the size of the forecast ensemble.

    The non-Numba accelerated version much slower for large ensembles: it
    requires both time and space O(N * E ** 2).

    Parameters
    ----------
    observations : float or array_like
        Observations float or array. Missing values (NaN) are given scores of
        NaN.
    forecasts : float or array_like
        Array of forecasts ensemble members, of the same shape as observations
        except for the axis along which CRPS is calculated (which should be the
        axis corresponding to the ensemble). If forecasts has the same shape as
        observations, the forecasts are treated as deterministic. Missing
        values (NaN) are ignored.
    weights : array_like, optional
        If provided, the CRPS is calculated exactly with the assigned
        probability weights to each forecast. Weights should be positive, but
        do not need to be normalized. By default, each forecast is weighted
        equally.
    issorted : bool, optional
        Optimization flag to indicate that the elements of `ensemble` are
        already sorted along `axis`.
    axis : int, optional
        Axis in forecasts and weights which corresponds to different ensemble
        members, along which to calculate CRPS.

    Returns
    -------
    out : np.ndarray
        CRPS for each ensemble forecast against the observations.

    References
    ----------
    Jochen Broecker. Chapter 7 in Forecast Verification: A Practitioner's Guide
        in Atmospheric Science. John Wiley & Sons, Ltd, Chichester, UK, 2nd
        edition, 2012.
        https://drive.google.com/a/climate.com/file/d/0B8AfRcot4nsIYmc3alpTeTZpLWc
    Tilmann Gneiting and Adrian E. Raftery. Strictly proper scoring rules,
        prediction, and estimation, 2005. University of Washington Department of
        Statistics Technical Report no. 463R.
        https://www.stat.washington.edu/research/reports/2004/tr463R.pdf
    Wilks D.S. (1995) Chapter 8 in _Statistical Methods in the
        Atmospheric Sciences_. Academic Press.
    """
    if axis != -1:
        forecasts = move_axis_to_end(forecasts, axis)

    if weights is not None:
        weights = move_axis_to_end(weights, axis)
        if weights.shape != forecasts.shape:
            raise ValueError('forecasts and weights must have the same shape')

    if observations.shape not in [forecasts.shape, forecasts.shape[:-1]]:
        raise ValueError('observations and forecasts must have matching '
                         'shapes or matching shapes except along `axis=%s`'
                         % axis)

    if observations.shape == forecasts.shape:
        if weights is not None:
            raise ValueError('cannot supply weights unless you also supply '
                             'an ensemble forecast')
        return abs(observations - forecasts)

    if not issorted:
        if weights is None:
            forecasts, _ = forecasts.sort(axis=-1)
        else:
            idx = argsort_indices(forecasts, axis=-1)
            forecasts = forecasts[idx]
            weights = weights[idx]




    return _crps_ensemble_vectorized(observations, forecasts, weights)


if __name__ == '__main__':

    ensemble = np.random.RandomState(0).randn(5,6)
    means = np.random.RandomState(0).randn(5)
    weights = np.random.RandomState(1).randn(5, 6)
    print(ps.crps_ensemble(means, ensemble, weights))

    means = torch.tensor(means, requires_grad=False)
    ensemble = torch.tensor(ensemble, requires_grad=True)
    weights = torch.tensor(weights, requires_grad=True)

    score = crps_ensemble(means, ensemble, weights)
    print(score.detach().numpy(), 'xxx')

    loss = score.mean()
    loss.backward()

    print(ensemble.grad)
    print()
    print(weights.grad)