import csv

import numpy as np

from DFTAnalysis import ReturnCSVVals, positionDFTCalculator


def load_position_csv(file_path):
    x_values, y_values = ReturnCSVVals(file_path)
    timestep = None

    with open(file_path, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row.get("timestep"):
                timestep = float(row["timestep"])
                break

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

    magnitudes = positionDFTCalculator(position_array)
    frequencies = np.fft.rfftfreq(sample_count, d=timestep)

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
        "dominant_index": dominant_index,
        "dominant_frequency": dominant_frequency,
        "dominant_magnitude": dominant_magnitude,
    }
