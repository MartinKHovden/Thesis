module QDInteracting22SlaterNNDescentOptimization

include("../MLVMC.jl")
using .MLVMC

learningRates = [ 0.1, 0.01, 0.001]

for learningRate in learningRates
    # #Set up the optimiser from Flux: 
    numParticles = 2
    numDimensions = 2
    hamiltonian = "quantumDot" # Use "quantumDot" or "calogeroSutherland" or "bosons"
    harmonicOscillatorFrequency = 1.0
    interactingParticles = true

    # #Set up the optimiser from Flux: 

    lr =learningRate
    optim = Descent(lr)

    s = System(numParticles, 
        numDimensions, 
        hamiltonian, 
        omega=harmonicOscillatorFrequency, 
        interacting = interactingParticles)

    #Add the wavefunction elements:
    addWaveFunctionElement(s, SlaterMatrix( s ))
    addWaveFunctionElement(s, Gaussian( 1.0 ))
    addWaveFunctionElement(s, NN(s, 20, 20, "sigmoid"))

    # addWaveFunctionElement(s, Jastrow(s))
    # addWaveFunctionElement(s, PadeJastrow( s; beta = 1.0 ))
    # addWaveFunctionElement(s, RBM(s, 4, 1.0))
    # println(s)

    numOptimizationSteps = 2000
    numMCMCSteps = 5000
    mcmcStepLength = 0.1
    runVMC!(s, numOptimizationSteps, numMCMCSteps, mcmcStepLength, optim, sampler = "is", writeToFile = true)
end


end