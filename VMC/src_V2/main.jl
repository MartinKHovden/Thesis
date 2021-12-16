module main

include("MLVMC.jl")
using .MLVMC

#Set up the system:
numParticles = 2
numDimensions = 2
hamiltonian = "quantumDot" # Use "quantumDot" or "calogeroSutherland"
harmonicOscillatorFrequency = 1.0
interactingParticles = false
s = System(numParticles, numDimensions, hamiltonian, omega=harmonicOscillatorFrequency, interacting = interactingParticles)

#Add the wavefunction elements:
addWaveFunctionElement(s, SlaterMatrix( s ))
addWaveFunctionElement(s, Gaussian( 1.0 ))
# addWaveFunctionElement(s, Jastrow(s))
# addWaveFunctionElement(s, NN(s, 4, 2, "sigmoid"))
# addWaveFunctionElement(s, RBM(s, 2, 1.0))
@time runMetropolis!(s, 100000,  sampler="is", 0.5, writeToFile = false, calculateOnebody = false)

# #Set up the optimiser from Flux: 
learningrate = 0.5
optim = Descent(learningrate)

# #Set up and run the VMC-calculation:
numOptimizationSteps = 40000
numMCMCSteps = 100000
mcmcStepLength = 0.5
# runVMC!(s, numOptimizationSteps, numMCMCSteps, mcmcStepLength, optim)

end

