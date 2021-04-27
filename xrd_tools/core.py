# -*- coding: utf-8 -*-
# pylint:disable=logging-format-interpolation
"""Calculation happens here"""
import pymatgen
import hashlib
import pytojcamp
from func_timeout import func_set_timeout
from pymatgen.core import Lattice, Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pytojcamp import from_dict

from . import __version__, pattern_cache, laue_cache, logger
from .laue_predictor import LatticeXRDCalculator


@func_set_timeout(60)
def calculate_pattern(
    filecontent, extension="cif", wavelength="CuKa", return_jcamp=False
):
    """Use pymatgen to calculate a PXRD pattern based on a pymatgen structure"""

    m = hashlib.md5()
    m.update(filecontent.encode("utf-8"))
    input_hash = m.hexdigest()
    response = None
    try:
        response = pattern_cache.get(input_hash)
    except KeyError:
        pass
    logger.debug("Response from cache for key {} is {}".format(input_hash, response))
    if response is not None:
        logger.info("Returning from cache")
        return response

    structure = Structure.from_str(filecontent, fmt=extension)
    calculator = XRDCalculator(wavelength=wavelength)
    pattern = calculator.get_pattern(structure)
    two_theta = list(pattern.x)
    intensity = list(pattern.y)
    hkls = pattern.hkls

    output_dict = {
        "x": two_theta,
        "y": intensity,
        "hkls": hkls,
        "apiVersion": __version__,
    }

    if return_jcamp:

        jcamp = from_dict(
            {
                "x": {
                    "data": output_dict["x"],
                    "unit": "°",
                    "type": "INDEPENDENT",
                    "name": "2 theta",
                },
                "y": {
                    "data": output_dict["y"],
                    "type": "DEPENDENT",
                    "unit": "",
                    "name": "intensity",
                },
            },
            data_type="Predicted PXRD pattern",
            origin=f"Pymatgen version {pymatgen.__version__}\
                        converted to JCAMP with pytojcamp version\
                             {pytojcamp.__version__}",
            meta={"wavelength": wavelength},
        )

        output_dict["jcamp"] = jcamp

    logger.debug("Trying to set cache")
    pattern_cache.set(input_hash, output_dict)

    return output_dict


@func_set_timeout(60)
def calculate_laue_pattern(  # pylint:disable=invalid-name,too-many-arguments, too-many-locals
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
    wavelength: str = "CuKa",
    return_jcamp: bool = False,
) -> dict:
    """Calculate the diffraction positions for a cell

    Args:
        a (float): Cell parameter a
        b (float): Cell parameter b
        c (float): Cell parameter c
        alpha (float): Cell angle alpha
        beta (float): Cell angle beta
        gamma (float): Cell angle gamma
        wavelength (str, optional): Wavelength that is used
            for the calculation of the reflex positions.
            According to the Bragg equation that influences the spacing
            of the reflexes
            Defaults to "CuKa".
        return_jcamp (bool, optional): If true, also returns a JCAMP as string.
            Defaults to False.

    Returns:
        dict: [description]
    """
    request_content = f"{a}{b}{c}{alpha}{beta}{gamma}{wavelength}"
    m = hashlib.md5()
    m.update(request_content.encode("utf-8"))
    input_hash = m.hexdigest()
    response = None
    try:
        response = laue_cache.get(input_hash)
    except KeyError:
        pass
    logger.debug("Response from cache for key {} is {}".format(input_hash, response))
    if response is not None:
        logger.info("Returning from cache")
        return response
    try:
        lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        structure = Structure(lattice=lattice, species=[], coords=[])
        calculator = LatticeXRDCalculator(wavelength=wavelength)
        pattern = calculator.get_pattern(structure)

        output_dict = {
            "x": list(pattern.x),
            "y": list(pattern.y),
            "hkls": pattern.hkls,
            "apiVersion": __version__,
        }

        if return_jcamp:

            jcamp = from_dict(
                {
                    "x": {
                        "data": output_dict["x"],
                        "unit": "°",
                        "type": "INDEPENDENT",
                        "name": "2 theta",
                    },
                    "y": {
                        "data": output_dict["y"],
                        "type": "DEPENDENT",
                        "unit": "",
                        "name": "intensity",
                    },
                },
                data_type="Predicted PXRD pattern",
                origin=f"Pymatgen version {pymatgen.__version__}\
                            converted to JCAMP with pytojcamp version\
                                {pytojcamp.__version__}",
                meta={"wavelength": wavelength},
            )

            output_dict["jcamp"] = jcamp

        logger.debug("Trying to set cache")
        laue_cache.set(input_hash, output_dict)

    except Exception as e:
        raise ValueError("Some error occured {}".format(e)) from e
    return output_dict
