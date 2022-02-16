module BosonsnInteracting22SlaterNNOptim

include("../MLVMC.jl")
using .MLVMC

#Set up the system:
for lr in [0.1, 0.01, 0.001]
    numParticles = 10

    numDimensions = 3
    hamiltonian = "bosons" # Use "quantumDot" or "calogeroSutherland" or "bosons"
    harmonicOscillatorFrequency = 1.0
    interactingParticles = true



    # #Set up the optimiser from Flux: 

    learningrate = lr
    optim = ADAM(learningrate)

    s = System(numParticles, 
        numDimensions, 
        hamiltonian, 
        omega=harmonicOscillatorFrequency, 
        interacting = interactingParticles)

    #Add the wavefunction elements:
    # addWaveFunctionElement(s, SlaterMatrix( s ))
    addWaveFunctionElement(s, Gaussian( 1.0 ))
    addWaveFunctionElement(s, NN(s, 12, 12, "tanh"))

    # addWaveFunctionElement(s, Jastrow(s))
    # addWaveFunctionElement(s, PadeJastrow( s; beta = 5000.0 ))
    # addWaveFunctionElement(s, RBM(s, 4, 1.0))
    # println(s)


    numOptimizationSteps = 500
    numMCMCSteps = 1000
    mcmcStepLength = 0.005
    runVMC!(s, numOptimizationSteps, numMCMCSteps, mcmcStepLength, optim, sampler = "is", writeToFile = true)
end

end