"""Wet-bulb temperature from air temperature and dewpoint, via Stull (2011).

Self-contained: does not import the ``heatextremes`` package, and works
elementwise on plain numpy scalars/arrays or xarray DataArrays (relies only
on ``numpy`` ufuncs, which xarray dispatches over automatically -- same
pattern as the rest of this pipeline, e.g. ``climatology.py``'s use of
``np.isclose``).

Wet-bulb temperature is the temperature an air parcel would reach if cooled
to saturation by evaporating water into it at constant pressure -- it is the
standard way to combine heat and humidity into a single human-heat-stress
metric (e.g. the oft-cited "35 degC wet-bulb" survivability limit). The
thermodynamically exact value is defined implicitly by a psychrometric
energy balance and needs pressure as an input, normally solved for
iteratively (e.g. MetPy's ``wet_bulb_temperature``). ERA5's cache has a
``surface_pressure`` variable that would support that exact calculation
(see ``era5_loader.open_cached_era5``'s docstring), but the AIFS-single-v2
model store currently only has ``2m_temperature``/``2m_dewpoint_temperature``
(see ``aifs_singlev2.py``) -- no pressure -- so an exact calculation isn't
available for the model side without first checking whether the real store
has a pressure variable that just isn't being requested yet.

This module instead uses Stull, R. (2011), "Wet-Bulb Temperature from
Relative Humidity and Air Temperature," J. Appl. Meteor. Climatol., 50,
2267-2269 (https://journals.ametsoc.org/view/journals/apme/50/11/jamc-d-11-0143.1.xml)
-- a closed-form empirical fit needing only temperature and relative
humidity (no pressure), calibrated against a full psychrometric calculation
at standard sea-level pressure (1013.25 hPa). Stull reports ~0.3 degC RMSE
over -20 <= T <= 50 degC and 5 <= RH <= 99%. Because the fit is calibrated
at one fixed pressure, it is a good match for low-elevation regions (e.g.
the Indo-Gangetic plain) but degrades at high elevation, where the exact
pressure-dependent calculation would be needed instead.

Relative humidity itself is derived here from temperature and dewpoint via
the Magnus-Tetens saturation-vapor-pressure approximation (Alduchov &
Eskridge 1996 coefficients), since dewpoint's defining property is that the
saturation vapor pressure at the dewpoint equals the actual (unsaturated)
vapor pressure at the real temperature.
"""

from __future__ import annotations

import numpy as np

KELVIN_TO_CELSIUS_OFFSET = 273.15


def saturation_vapor_pressure(temperature_c):
    """Saturation vapor pressure (hPa) at ``temperature_c`` (degrees C).

    Magnus-Tetens approximation, Alduchov & Eskridge (1996) coefficients --
    accurate to within ~0.1% over -40 to 50 degC:
        e_s(T) = 6.1094 * exp(17.625*T / (T + 243.04))
    """
    return 6.1094 * np.exp(17.625 * temperature_c / (temperature_c + 243.04))


def relative_humidity_from_dewpoint(temperature_c, dewpoint_c, clip=True):
    """Relative humidity (percent, 0-100) from air temperature and dewpoint (degC).

    The actual vapor pressure equals the saturation vapor pressure at the
    dewpoint (that is the definition of dewpoint), so
        RH = 100 * e_s(Td) / e_s(T)

    Parameters
    ----------
    temperature_c, dewpoint_c : array-like or xr.DataArray
        Air temperature and dewpoint temperature, in degrees C. Must already
        be converted from Kelvin if that's the source units -- this function
        does not do unit conversion (see ``wet_bulb_temperature`` below,
        which does).
    clip : bool
        If True (default), clip the result to [0, 100] -- real dewpoint
        should never exceed real temperature, but floating-point noise or
        upstream data artifacts can occasionally push RH fractionally above
        100%, which would push ``wet_bulb_temperature_stull`` outside the
        RH range Stull's fit was calibrated over. Set False to see the raw,
        unclipped value (e.g. for diagnosing suspect input data).
    """
    relative_humidity_pct = 100.0 * saturation_vapor_pressure(dewpoint_c) / saturation_vapor_pressure(temperature_c)
    if clip:
        relative_humidity_pct = np.clip(relative_humidity_pct, 0.0, 100.0)
    return relative_humidity_pct


def wet_bulb_temperature_stull(temperature_c, relative_humidity_pct):
    """Wet-bulb temperature (degC) via Stull (2011)'s empirical formula.

    Parameters
    ----------
    temperature_c : array-like or xr.DataArray
        Air temperature in degrees C.
    relative_humidity_pct : array-like or xr.DataArray
        Relative humidity in percent (0-100 -- not a 0-1 fraction).

    Valid, per Stull, for -20 <= temperature_c <= 50 and
    5 <= relative_humidity_pct <= 99, at standard sea-level pressure; see
    this module's docstring for the sea-level-pressure caveat.
    """
    t = temperature_c
    rh = relative_humidity_pct
    return (
        t * np.arctan(0.151977 * (rh + 8.313659) ** 0.5)
        + np.arctan(t + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh**1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )


def wet_bulb_temperature(temperature, dewpoint, input_units="K"):
    """Wet-bulb temperature from air temperature and dewpoint, via Stull (2011).

    Convenience wrapper combining ``relative_humidity_from_dewpoint`` and
    ``wet_bulb_temperature_stull``, handling the Kelvin-to-Celsius
    conversion both AIFS-single-v2 and ERA5 need (both store
    ``2m_temperature``/``2m_dewpoint_temperature`` in Kelvin -- see the
    Kelvin/Celsius conversion in ``_build_notebook.py``'s Step 2 cell).

    Parameters
    ----------
    temperature, dewpoint : array-like or xr.DataArray
        Air temperature and dewpoint temperature, in the units given by
        ``input_units``.
    input_units : {"K", "degC"}
        Units ``temperature``/``dewpoint`` are already in. Default "K",
        matching the raw AIFS-single-v2/ERA5 store convention.

    Returns
    -------
    Same type as the inputs (array-like or xr.DataArray), in degrees C. If
    the inputs are xr.DataArray, the result is named "wet_bulb_temperature"
    with ``attrs["units"] = "degC"`` set (plain arithmetic drops attrs by
    default, same issue as the temperature Kelvin-conversion bug fixed
    elsewhere in this pipeline).
    """
    if input_units == "K":
        temperature_c = temperature - KELVIN_TO_CELSIUS_OFFSET
        dewpoint_c = dewpoint - KELVIN_TO_CELSIUS_OFFSET
    elif input_units == "degC":
        temperature_c = temperature
        dewpoint_c = dewpoint
    else:
        raise ValueError(f"input_units must be 'K' or 'degC', got {input_units!r}")

    relative_humidity_pct = relative_humidity_from_dewpoint(temperature_c, dewpoint_c)
    wet_bulb_c = wet_bulb_temperature_stull(temperature_c, relative_humidity_pct)

    if hasattr(wet_bulb_c, "attrs"):
        wet_bulb_c = wet_bulb_c.rename("wet_bulb_temperature")
        wet_bulb_c.attrs["units"] = "degC"
    return wet_bulb_c
