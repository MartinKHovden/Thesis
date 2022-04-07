module timeComparisonWaveFunctionElements

include("../MLVMC.jl")
using .MLVMC

#Set up the system:
numParticles = 20
numDimensions = 2
hamiltonian = "quantumDot" # Use "quantumDot" or "calogeroSutherland" or "bosons"
harmonicOscillatorFrequency = 1.0
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
addWaveFunctionElement(s, SlaterMatrix( s ))
addWaveFunctionElement(s, Gaussian( 1.0 ))
addWaveFunctionElement(s, PadeJastrow(s, beta= 1.0))
# addWaveFunctionElement(s, NN(s, 80, 80, "sigmoid"))



# numOptimizationSteps = 2000
numMCMCSteps = 10000
mcmcStepLength = 0.001#0.001



# runVMC!(s, numOptimizationSteps, numMCMCSteps, mcmcStepLength, optim, sampler = "is", writeToFile = false)
@time runMetropolis!(s, 
                        10000,  
                        mcmcStepLength, 
                        sampler="is", 
                        writeToFile = false, 
                        calculateOnebody = false)

@time runMetropolis!(s, 
                        10000,  
                        mcmcStepLength, 
                        sampler="is", 
                        writeToFile = false, 
                        calculateOnebody = false)

@time runMetropolis!(s, 
                        10000,  
                        mcmcStepLength, 
                        sampler="is", 
                        writeToFile = false, 
                        calculateOnebody = false)

@time runMetropolis!(s, 
                        10000,  
                        mcmcStepLength, 
                        sampler="is", 
                        writeToFile = false, 
                        calculateOnebody = false)


end