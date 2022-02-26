module calogero
include("MLVMC.jl")
using .MLVMC

#Set up the system:
numParticles = 3
numDimensions = 1
hamiltonian = "calogeroSutherland" # Use "quantumDot" or "calogeroSutherland" or "bosons"
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
# addWaveFunctionElement(s, GaussianSimple( 0.5 ))

# addWaveFunctionElement(s, Jastrow(s))
addWaveFunctionElement(s, NN(s, 12, 6, "sigmoid"))
# addWaveFunctionElement(s, RBM(s, 2, 1.0))
# @time runMetropolis!(s, 
#                 100000,  
#                 0.0005, 
#                 optimizationIteration = 1000,
#                 sampler="is", 
#                 writeToFile = true, 
#                 calculateOnebody = false)

# #Set up the optimiser from Flux: 
learningrate = 0.1
optim = Descent(learningrate)

# #Set up and run the VMC-calculation:
numOptimizationSteps = 40000
numMCMCSteps = 2^10
mcmcStepLength = 0.1
runVMC!(s, numOptimizationSteps, numMCMCSteps, mcmcStepLength, optim, sampler = "is", writeToFile = false)






end