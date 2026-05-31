#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:28:10 2026

@author: alidavanwijngaarden
"""

import numpy as np


TIME_POINTS = np.array([0, 15, 20, 70, 100, 160], dtype=float)

def analyze_crystal(sizes, uncertainties, times=TIME_POINTS):
    sizes = np.asarray(sizes, dtype=float)
    uncertainties = np.asarray(uncertainties, dtype=float)
    times = np.asarray(times, dtype=float)


    # 1. Interpolate sizes if needed
    mask = np.isnan(sizes)
    if np.any(mask):
        sizes[mask] = np.interp(
            times[mask],
            times[~mask],
            sizes[~mask]
        )

    area = np.trapezoid(sizes, times)
    total_time = times[-1] - times[0]
    average = area / total_time

    # 3. Uncertainty propagation (trapezoidal rule)
    weights = np.zeros_like(times)

    weights[0] = (times[1] - times[0]) / 2.0
    for i in range(1, len(times) - 1):
        weights[i] = (times[i + 1] - times[i - 1]) / 2.0
    weights[-1] = (times[-1] - times[-2]) / 2.0

    area_uncertainty = np.sqrt(np.sum((weights * uncertainties) ** 2))
    average_uncertainty = area_uncertainty / total_time

    return area, area_uncertainty, average, average_uncertainty


# Input values (edit as needed)

crystals = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]

uncertainties = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]

print("Type of experiment: Gypsum,  Analysis based on: Various hydrogen contact time")

for i, (sizes, errs) in enumerate(zip(crystals, uncertainties), start=1):

    area, area_u, avg, avg_u = analyze_crystal(sizes, errs)

    print(f"\nCrystal {i}")
    print(f"Area = {area:.1f} ± {area_u:.1f} µm²·h")
    print(f"Average = {avg:.1f} ± {avg_u:.1f} µm²")







