# Importing various packages
from math import exp, sqrt
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys
import time

#Trial wave function for quantum dots in two dims
def WaveFunction(r,alpha,beta):
    r1 = r[0,0]**2 + r[0,1]**2
    r2 = r[1,0]**2 + r[1,1]**2
    r12 = sqrt((r[0,0]-r[1,0])**2 + (r[0,1]-r[1,1])**2)
    deno = r12/(1+beta*r12)
    return exp(-0.5*alpha*(r1+r2)+deno)

#Local energy  for quantum dots in two dims, using analytical local energy
def LocalEnergy(r,alpha,beta):
    
    r1 = (r[0,0]**2 + r[0,1]**2)
    r2 = (r[1,0]**2 + r[1,1]**2)
    r12 = sqrt((r[0,0]-r[1,0])**2 + (r[0,1]-r[1,1])**2)
    deno = 1.0/(1+beta*r12)
    deno2 = deno*deno
    return 0.5*(1-alpha*alpha)*(r1 + r2) +2.0*alpha + 1.0/r12+deno2*(alpha*r12-deno2+2*beta*deno-1.0/r12)

# The Monte Carlo sampling with the Metropolis algo
def MonteCarloSampling(numMCCycles, stepSize, alpha, beta):
    # positions
    PositionOld = np.random.normal(size=(2,2))
    PositionNew = np.zeros((2,2), np.double)
    # seed for rng generator
    seed()
    # start variational parameter
    energy = 0.0
    deltaEnergy = 0.0
    #Initial position
    # for i in range(2):
    #     for j in range(2):
    #         PositionOld[i,j] = stepSize * (random() - .5)
    wfold = WaveFunction(PositionOld,alpha,beta)

    #Loop over MC MCcycles
    for MCcycle in range(numMCCycles):
        #Trial position
        for i in range(2):
            for j in range(2):
                PositionNew[i,j] = PositionOld[i,j] + stepSize * (random() - .5)
        wfnew = WaveFunction(PositionNew,alpha,beta)

        #Metropolis test to see whether we accept the move
        if random() < wfnew**2 / wfold**2:
            PositionOld = PositionNew.copy()
            wfold = wfnew
            deltaEnergy = LocalEnergy(PositionOld,alpha,beta)
        energy += deltaEnergy
    #We calculate mean, variance and error ...
    energy /= numMCCycles

    return energy

startTime = time.time()
print(MonteCarloSampling(10**6, 0.5, 1.0, 1.0))
print(time.time() - startTime)

