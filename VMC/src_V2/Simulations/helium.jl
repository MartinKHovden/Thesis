module helium

include("../MLVMC.jl")
using .MLVMC

#Set up the system:
numParticles = 2
numDimensions = 3
hamiltonian = "helium" # Use "quantumDot" or "calogeroSutherland" or "bosons"
harmonicOscillatorFrequency = 0.0
interactingParticles = true
s = System(numParticles, 
        numDimensions, 
        hamiltonian, 
        omega=harmonicOscillatorFrequency, 
        interacting = interactingParticles)

#Add the wavefunction elements:
# addWaveFunctionElement(s, SlaterMatrix( s ))
# addWaveFunctionElement(s, Gaussian( 1.0 ))
addWaveFunctionElement(s, GaussianSimple( 1.0 ))

# addWaveFunctionElement(s, PadeJastrow(s, beta = 1.0))
addWaveFunctionElement(s, NN(s, 6, 2, "sigmoid"))
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
optim = ADAM(learningrate)

# #Set up and run the VMC-calculation:
numOptimizationSteps = 1000
numMCMCSteps = 1000
mcmcStepLength = 0.1
runVMC!(s, numOptimizationSteps, numMCMCSteps, mcmcStepLength, optim, sampler = "is", writeToFile = false)


# addWaveFunctionElement(s, NN(s, 15, 15, "sigmoid"))

optim=ADAM(0.1)
#
numOptimizationSteps = 1000
numMCMCSteps = 1000

runVMC!(s, numOptimizationSteps, numMCMCSteps, mcmcStepLength, optim, sampler = "is", writeToFile = false)




end