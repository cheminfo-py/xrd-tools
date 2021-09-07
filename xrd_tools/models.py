# -*- coding: utf-8 -*-
"""Pydantic input and response models"""
from typing import Optional, List

from pydantic import BaseModel, validator
from pymatgen.analysis.diffraction.xrd import WAVELENGTHS

from . import __version__

EXTENSIONS = "cif"


class XRDReflexes(BaseModel):
    x: List[float]
    y: List[float]


class XRDDatabaseEntry(BaseModel):
    """Representing an XRD database entry"""

    name: str  # for example, ORIWIG
    source: str  # for example, CSD
    composition: str  # sorted chemical formula
    a: float  # cell constant a in Angstrom
    b: float  # cell constant b in Angstromß
    c: float  # cell constant c in Angstrom
    alpha: float  # cell angle alpha in degree
    beta: float  # cell angle beta in degree
    gamma: float  # cell angle gamma in degree
    reflexes: XRDReflexes
    apiVersion: Optional[str] = __version__

class XRDDatabaseEntries(BaseModel):
    entries: List[XRDDatabaseEntry]


class XRDResponse(BaseModel):
    """Default response for XRD prediction"""

    x: list
    y: list
    hkls: list
    jcamp: Optional[str]
    apiVersion: Optional[str] = __version__


class DataBaseEntry(BaseModel):
    """Default response for lookup of a database entry"""

    x: list
    y: list
    hkls: list
    dbName: str
    link: Optional[str] # if we can make a link, we create a link to the COD, or CSD. For the hypothetical databases we currently cannot do that
    cif: str
    composition: Optional[str] 
    a: Optional[float]  # cell constant a in Angstrom
    b: Optional[float]  # cell constant b in Angstromß
    c: Optional[float]  # cell constant c in Angstrom
    alpha: Optional[float]  # cell angle alpha in degree
    beta: Optional[float]  # cell angle beta in degree
    gamma: Optional[float]  # cell angle gamma in degree
    apiVersion: Optional[str] = __version__


class XRDInput(BaseModel):
    """Input for the prediction of a XRD pattern"""

    fileContent: str
    wavelength: Optional[str] = "CuKa"
    extension: Optional[str] = "cif"
    jcamp: Optional[bool] = False

    @validator("wavelength")
    def wavelength_must_be_known(
        cls, value
    ):  # pylint:disable=no-self-argument,no-self-use
        """We only allow for anode names for which we know the wavelength"""
        if value not in list(WAVELENGTHS.keys()):
            raise ValueError(
                "Wavelength must be in {}".format(", ".join(list(WAVELENGTHS.keys())))
            )
        return value

    @validator("extension")
    def extension_must_be_known(
        cls, value
    ):  # pylint:disable=no-self-argument,no-self-use
        """We only allow for extensions that we can read"""
        if value not in EXTENSIONS:
            raise ValueError("Extension must be in {}".format(", ".join(EXTENSIONS)))
        return value


class LatticeInput(BaseModel):
    """Input for the prediction of XRD reflex positions"""

    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    wavelength: Optional[str] = "CuKa"
    jcamp: Optional[bool] = False

    @validator("wavelength")
    def wavelength_must_be_known(
        cls, value
    ):  # pylint:disable=no-self-argument,no-self-use
        """We only allow for anode names for which we know the wavelength"""
        if value not in list(WAVELENGTHS.keys()):
            raise ValueError(
                "Wavelength must be in {}".format(", ".join(list(WAVELENGTHS.keys())))
            )
        return value


class MatchingInput(BaseModel): 
    """Input for the matching of PXRD patterns to the database"""
    x: List[float] # angle in two-theta
    y: List[float] # intensity, does not need to be scaled 
    k: int = 10 # the number of matches that should be returned
