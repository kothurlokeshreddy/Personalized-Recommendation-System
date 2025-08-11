#!/bin/bash
# Upgrade pip & install build essentials
pip install --upgrade pip setuptools wheel
pip install numpy<2.0 cython cmake
