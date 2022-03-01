module QDInteracting62SlaterJastrowOneBody

include("../MLVMC.jl")
using .MLVMC

#Set up the system:
numParticles = 6
numDimensions = 2
hamiltonian = "quantumDot" # Use "quantumDot" or "calogeroSutherland" or "bosons"
harmonicOscillatorFrequency = 1.0
interactingParticles = true



# #Set up the optimiser from Flux: 



# addWaveFunctionElement(s, Jastrow(s))
# addWaveFunctionElement(s, PadeJastrow( s; beta = 5000.0 ))
# addWaveFunctionElement(s, RBM(s, 4, 1.0))
# println(s)

learningrate = 0.01
optim = Descent(learningrate)

s = System(numParticles, 
    numDimensions, 
    hamiltonian, 
    omega=harmonicOscillatorFrequency, 
    interacting = interactingParticles)

#Add the wavefunction elements:
addWaveFunctionElement(s, SlaterMatrix( s ))
addWaveFunctionElement(s, Gaussian( 1.0 ))
addWaveFunctionElement(s, PadeJastrow( s; beta = 0.45702))



mcmcStepLength = 0.01
numOptimizationSteps = 1000
numMCMCSteps = 10000

# runVMC!(s, numOptimizationSteps, numMCMCSteps, mcmcStepLength, optim, sampler = "is", writeToFile = false)


@time runMetropolis!(s, 
            2^22,  
            0.001, 
            sampler="is", 
            writeToFile = false, 
            calculateOnebody = true)


end