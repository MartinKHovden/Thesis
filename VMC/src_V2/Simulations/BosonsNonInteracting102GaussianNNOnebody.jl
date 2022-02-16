module BosonsNonInteracting102SlaterOneBody

include("../MLVMC.jl")
using .MLVMC

#Set up the system:
numParticles = 10
numDimensions = 2
hamiltonian = "bosons" # Use "quantumDot" or "calogeroSutherland" or "bosons"
harmonicOscillatorFrequency = 1.0
interactingParticles = false



# #Set up the optimiser from Flux: 



# addWaveFunctionElement(s, Jastrow(s))
# addWaveFunctionElement(s, PadeJastrow( s; beta = 5000.0 ))
# addWaveFunctionElement(s, RBM(s, 4, 1.0))
# println(s)

learningrate = 0.1
optim = Descent(learningrate)

s = System(numParticles, 
    numDimensions, 
    hamiltonian, 
    omega=harmonicOscillatorFrequency, 
    interacting = interactingParticles)

#Add the wavefunction elements:
# addWaveFunctionElement(s, SlaterMatrix( s ))
addWaveFunctionElement(s, Gaussian( 1.0 ))
addWaveFunctionElement(s, NN(s, 12, 12, "tanh"))



mcmcStepLength = 0.5
numOptimizationSteps = 100
numMCMCSteps = 1000

runVMC!(s, numOptimizationSteps, numMCMCSteps, mcmcStepLength, optim, sampler = "is", writeToFile = false)


@time runMetropolis!(s, 
            2^21,  
            mcmcStepLength, 
            sampler="is", 
            writeToFile = false, 
            calculateOnebody = true)


end