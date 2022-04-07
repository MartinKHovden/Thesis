module Calogero

include("../MLVMC.jl")
using .MLVMC

#Set up the system:
numParticles = 2
numDimensions = 3
hamiltonian = "calogeroSutherland" # Use "quantumDot" or "calogeroSutherland" or "bosons"
harmonicOscillatorFrequency = 0.0
interactingParticles = true



# #Set up the optimiser from Flux: 

learningrate = 0.01#0.01
optim = ADAM(learningrate)

s = System(numParticles, 
    numDimensions, 
    hamiltonian, 
    omega=harmonicOscillatorFrequency, 
    interacting = interactingParticles)

#Add the wavefunction elements:
# addWaveFunctionElement(s, SlaterMatrix( s ))
addWaveFunctionElement(s, GaussianSimple( 1.0 ))
addWaveFunctionElement(s, NN(s, 10, 10, "sigmoid"))



numOptimizationSteps = 1000
numMCMCSteps = 10000
mcmcStepLength = 0.01#0.001



runVMC!(s, numOptimizationSteps, numMCMCSteps, mcmcStepLength, optim, sampler = "is", writeToFile = true)
@time runMetropolis!(s, 
                        2^22,  
                        mcmcStepLength, 
                        sampler="is", 
                        writeToFile = true, 
                        calculateOnebody = true)



end
