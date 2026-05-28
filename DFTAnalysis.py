#Output csv structure is as below
#step_index,system,mu,timestep,time_sim,time_real_days,x,y,vx,vy,ax,ay,r1,r2
import csv
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

    return magnitudes
