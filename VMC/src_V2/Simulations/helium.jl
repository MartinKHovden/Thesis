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
# addWaveFunctionElement(s, NN(s, 6, 2, "sigmoid"))
# addWaveFunctionElement(s, RBM(s, 2, 1.0))
# @time runMetropolis!(s, 
#                 1000000,  
#                 0.5, 
#                 optimizationIteration = 1000,
#                 sampler="is", 
#                 writeToFile = false, 
#                 calculateOnebody = false)

# #Set up the optimiser from Flux: 
learningrate = 0.1
optim = Descent(learningrate)

#Set up and run the VMC-calculation:
numOptimizationSteps = 100
numMCMCSteps = 50000
mcmcStepLength = 0.1
runVMC!(s, numOptimizationSteps, numMCMCSteps, mcmcStepLength, optim, sampler = "is", writeToFile = false)

addWaveFunctionElement(s, NN(s, 20, 20, "sigmoid"))

optim=Momentum(0.01)
#
numOptimizationSteps = 100
numMCMCSteps = 3000
mcmcStepLength = 0.01

runVMC!(s, numOptimizationSteps, numMCMCSteps, mcmcStepLength, optim, sampler = "is", writeToFile = false)
@time runMetropolis!(s, 
                1000000,  
                0.1, 
                optimizationIteration = 1000,
                sampler="is", 
                writeToFile = true, 
                calculateOnebody = true)



end