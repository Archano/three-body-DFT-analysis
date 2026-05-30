import csv

import numpy as np

from DFTAnalysis import (
    DFTCharacteristicAnalysis,
    ReturnCSVVals,
    pointDistanceDFTCalculator,
    positionDFTCalculator,
)


PRIMARY_ORBIT_TIME_UNITS = 2 * np.pi


def load_csv_metadata(file_path):
    timestep = None
    mu = None

    with open(file_path, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row.get("timestep"):
                timestep = float(row["timestep"])
            if row.get("mu"):
                mu = float(row["mu"])
            break

    return timestep, mu


def load_position_csv(file_path):
    x_values, y_values = ReturnCSVVals(file_path)
    timestep, _ = load_csv_metadata(file_path)

    if timestep is None:
        raise ValueError("CSV does not contain a timestep value.")

    return x_values, y_values, timestep


def position_spectrum(position_values, timestep):
    position_array = np.array(position_values, dtype=float)
    sample_count = len(position_array)

    if sample_count < 2:
        raise ValueError("At least two position samples are required.")

    if timestep <= 0:
        raise ValueError("Timestep must be positive.")

    magnitudes, cleaned_signal = positionDFTCalculator(position_array)
    frequencies = np.fft.rfftfreq(sample_count, d=timestep)
    frequencies = frequencies * PRIMARY_ORBIT_TIME_UNITS

    dominant_index = None
    dominant_frequency = None
    dominant_magnitude = None

    if len(magnitudes) > 1:
        dominant_index = int(np.argmax(magnitudes[1:]) + 1)
        dominant_frequency = frequencies[dominant_index]
        dominant_magnitude = magnitudes[dominant_index]

    return {
        "frequencies": frequencies,
        "magnitudes": magnitudes,
        "signal": cleaned_signal,
        "signal_time": np.arange(sample_count) * timestep / PRIMARY_ORBIT_TIME_UNITS,
        "dominant_index": dominant_index,
        "dominant_frequency": dominant_frequency,
        "dominant_magnitude": dominant_magnitude,
    }


def point_distance_spectrum(point_x, point_y, x_values, y_values, timestep):
    if len(x_values) != len(y_values):
        raise ValueError("x/y position arrays must have the same length.")

    sample_count = len(x_values)

    if sample_count < 2:
        raise ValueError("At least two position samples are required.")

    if timestep <= 0:
        raise ValueError("Timestep must be positive.")

    magnitudes, distances, cleaned_distances = pointDistanceDFTCalculator(
        point_x,
        point_y,
        x_values,
        y_values,
    )
    frequencies = np.fft.rfftfreq(sample_count, d=timestep)
    frequencies = frequencies * PRIMARY_ORBIT_TIME_UNITS

    dominant_index = None
    dominant_frequency = None
    dominant_magnitude = None

    if len(magnitudes) > 1:
        dominant_index = int(np.argmax(magnitudes[1:]) + 1)
        dominant_frequency = frequencies[dominant_index]
        dominant_magnitude = magnitudes[dominant_index]

    return {
        "frequencies": frequencies,
        "magnitudes": magnitudes,
        "distances": distances,
        "signal": cleaned_distances,
        "signal_time": np.arange(sample_count) * timestep / PRIMARY_ORBIT_TIME_UNITS,
        "cleaned_distances": cleaned_distances,
        "dominant_index": dominant_index,
        "dominant_frequency": dominant_frequency,
        "dominant_magnitude": dominant_magnitude,
    }


def spectrum_characteristics(spectrum):
    return DFTCharacteristicAnalysis(
        spectrum["magnitudes"],
        spectrum["frequencies"],
    )
