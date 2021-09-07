# -*- coding: utf-8 -*-
"""Testing basic working of the API"""
import os
import sys 
sys.path.append('../')
from fastapi.testclient import TestClient

from xrd_tools import __version__, app

THIS_DIR = os.path.dirname(os.path.realpath(__file__))


client = TestClient(app)

with open(os.path.join(THIS_DIR, "xalxuf01.cif"), "r") as fh:
    f = fh.read()


def test_read_main():
    """Health check"""
    response = client.get("/version")
    assert response.status_code == 200
    assert response.json() == {"version": __version__}


def test_predict():
    """Test prediction endpoint"""
    response = client.post("/predictxrd", json={"fileContent": f})
    assert response.status_code == 200
    body = response.json()
    assert "x" in body.keys()
    assert "y" in body.keys()
    assert "apiVersion" in body.keys()


def test_lattice_predict():
    """Test prediction endpoint"""
    response = client.post(
        "latticepattern",
        json={"a": 1, "b": 1, "c": 2, "alpha": 90, "beta": 90, "gamma": 90},
    )
    assert response.status_code == 200
    body = response.json()
    assert "x" in body.keys()
    assert len(body["x"]) == 1
    assert "y" in body.keys()
    assert "apiVersion" in body.keys()
