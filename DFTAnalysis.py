#Output csv structure is as below
#step_index,system,mu,timestep,time_sim,time_real_days,x,y,vx,vy,ax,ay,r1,r2
import argparse
import csv
import math
import numpy as np 

def ReturnCSVVals(filePath): 
    resultsArray = [] 
    xValues = [] 
    yValues = [] 
    with open(filePath, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            xValues.append(float(row["x"]))
            yValues.append(float(row["y"]))
    
    return xValues, yValues

def positionDFTCalculator(positionValuesArray): 
    positionValuesArray = np.array(positionValuesArray)

    positionMean = np.mean(positionValuesArray)
    cleanedArray = positionValuesArray - positionMean

    fftResults = np.fft.rfft(cleanedArray)
    magnitudes = np.abs(fftResults)

    return magnitudes, cleanedArray 

def pointDistanceDFTCalculator(objectPositionX,objectPositionY,xValues,yValues): 
    iterCount = len(xValues)

    distances = []

    for i in range (iterCount): 
        localParticleXPosition = xValues[i]
        localParticleYPosition = yValues[i]
        xTerm = math.pow(localParticleXPosition-objectPositionX,2)
        yTerm = math.pow(localParticleYPosition-objectPositionY,2)

        localDistance = math.sqrt(xTerm + yTerm)
        distances.append(localDistance)

    distancesArray = np.array(distances)
    positionMean = np.mean(distancesArray)
    cleanedArray = distancesArray - positionMean

    fftResults = np.fft.rfft(cleanedArray)
    magnitudes = np.abs(fftResults)

    return magnitudes, distancesArray, cleanedArray, 


def DFTCharacteristicAnalysis(magnitudesArray, frequenciesArray): 
    workingMagnitudes = np.array(magnitudesArray)
    workingFrequencies = np.array(frequenciesArray)

    if len(workingMagnitudes) == 0:
        raise ValueError("At least one DFT magnitude is required.")

    if len(workingMagnitudes) != len(workingFrequencies):
        raise ValueError("Magnitude and frequency arrays must have the same length.")

    peakCount = min(3, len(workingMagnitudes))

    # Indices of the top 3 largest magnitude peaks
    highestPeakIndices = np.argsort(workingMagnitudes)[-peakCount:][::-1]

    highestFrequencies = workingFrequencies[highestPeakIndices]
    highestMagnitudes = workingMagnitudes[highestPeakIndices]

    # Period = 1 / frequency; removes infinite period possibility
    highestPeriods = np.divide(1.0, highestFrequencies,
        out=np.full_like(highestFrequencies, np.inf),where=highestFrequencies != 0)

    dominantIndex = highestPeakIndices[0]
    dominantFrequency = workingFrequencies[dominantIndex]
    dominantMagnitude = workingMagnitudes[dominantIndex]

    if dominantFrequency == 0:
        dominantPeriod = float("inf")
    else:
        dominantPeriod = 1 / dominantFrequency

    power = workingMagnitudes ** 2
    totalPower = np.sum(power)

    if totalPower == 0:
        spectralConcentration3 = 0
    else:
        top3Power = power[highestPeakIndices]
        spectralConcentration3 = np.sum(top3Power) / totalPower

    return {
        "dominant_index": dominantIndex,
        "dominant_frequency": dominantFrequency,
        "dominant_magnitude": dominantMagnitude,
        "dominant_period": dominantPeriod,

        "top_peak_indices": highestPeakIndices,
        "top_frequencies": highestFrequencies,
        "top_magnitudes": highestMagnitudes,
        "top_periods": highestPeriods,

        "spectral_concentration_3": spectralConcentration3,
        "total_power": totalPower
    }

