module main

include("MLVMC.jl")
using .MLVMC

#Set up the system:
numParticles = 6
numDimensions = 2
hamiltonian = "quantumDot" # Use "quantumDot" or "calogeroSutherland" or "bosons"
harmonicOscillatorFrequency = 1.0
interactingParticles = true
s = System(numParticles, 
        numDimensions, 
        hamiltonian, 
        omega=harmonicOscillatorFrequency, 
        interacting = interactingParticles)

#Add the wavefunction elements:
# addWaveFunctionElement(s, SlaterMatrix( s ))
addWaveFunctionElement(s, Gaussian( 1.0 ))
addWaveFunctionElement(s, Jastrow(s))
# addWaveFunctionElement(s, NN(s, 20, 10, "tanh"))
# addWaveFunctionElement(s, RBM(s, 2, 1.0))
# @time runMetropolis!(s, 
#                 100000,  
#                 0.0005, 
#                 optimizationIteration = 1000,
#                 sampler="is", 
#                 writeToFile = true, 
#                 calculateOnebody = false)

# #Set up the optimiser from Flux: 
learningrate = 0.01
optim = ADAM(learningrate)

# #Set up and run the VMC-calculation:
numOptimizationSteps = 80
numMCMCSteps = 100000
mcmcStepLength = 0.01
runVMC!(s, numOptimizationSteps, numMCMCSteps, mcmcStepLength, optim, sampler = "is", writeToFile = true)

end

