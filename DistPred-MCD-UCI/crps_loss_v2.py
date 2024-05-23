import torch


def crps_ensemble(
    observations,
    forecasts,
    axis: int = -1,
    sorted_ensemble: bool = False,
    estimator: str = "pwm",
):
    r"""Estimate the Continuous Ranked Probability Score (CRPS) for a finite ensemble.

    Parameters
    ----------
    observations: ArrayLike
        The observed values.
    forecasts: ArrayLike
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
    axis: int
        The axis corresponding to the ensemble. Default is the last axis.
    sorted_ensemble: bool
        Boolean indicating whether the ensemble members are already in ascending order.
        Default is False.
    estimator: str
        Indicates the CRPS estimator to be used.
    backend: str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    crps: ArrayLike
        The CRPS between the forecast ensemble and obs.

    Examples
    --------
    >>> from scoringrules import crps
    >>> crps.ensemble(pred, obs)
    """

    if estimator not in ["nrg","pwm","fair"]:
        raise ValueError(f"{estimator} is not a valid estimator. ")

    if axis != -1:
        forecasts = torch.moveaxis(forecasts, axis, -1)

    if not sorted_ensemble and estimator not in ["nrg", "fair"]:
        forecasts, _ = torch.sort(forecasts, axis=-1)

    return ensemble(observations, forecasts, estimator)


def ensemble(obs, fct,estimator: str = "pwm"):
    """Compute the CRPS for a finite ensemble."""
    if estimator == "nrg":
        out = _crps_ensemble_nrg(obs, fct)
    elif estimator == "pwm":
        out = _crps_ensemble_pwm(obs, fct)
    elif estimator == "fair":
        out = _crps_ensemble_fair(obs, fct)
    else:
        raise ValueError("no estimator specified for ensemble!")
    return out.mean()


def _crps_ensemble_fair(obs, fct):
    """Fair version of the CRPS estimator based on the energy form."""
    M: int = fct.shape[-1]
    e_1 = torch.sum(torch.abs(obs[..., None] - fct), axis=-1) / M
    e_2 = torch.sum(
        torch.abs(fct[..., None] - fct[..., None, :]),
        axis=(-1, -2),
    ) / (M * (M - 1))
    return e_1 - 0.5 * e_2


def _crps_ensemble_nrg(obs, fct):
    """CRPS estimator based on the energy form."""
    M: int = fct.shape[-1]
    e_1 = torch.sum(torch.abs(obs[..., None] - fct), axis=-1) / M
    e_2 = torch.sum(torch.abs(fct[..., None] - fct[..., None, :]), (-1, -2)) / (M**2)
    return e_1 - 0.5 * e_2


def _crps_ensemble_pwm(obs, fct):
    """CRPS estimator based on the probability weighted moment (PWM) form."""
    M: int = fct.shape[-1]
    expected_diff = torch.sum(torch.abs(obs[..., None] - fct), axis=-1) / M
    β_0 = torch.sum(fct, axis=-1) / M
    β_1 = torch.sum(fct * torch.arange(M).to(fct.device), axis=-1) / (M * (M - 1.0))
    return expected_diff + β_0 - 2.0 * β_1

