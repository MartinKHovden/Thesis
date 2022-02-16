module QDInteracting22Slater

include("../MLVMC.jl")
using .MLVMC

for lr in [0.1, 0.01]
    numParticles = 2
    numDimensions = 2
    hamiltonian = "quantumDot" # Use "quantumDot" or "calogeroSutherland" or "bosons"
    harmonicOscillatorFrequency = 1.0
    interactingParticles = true



    # #Set up the optimiser from Flux: 

    learningrate = lr
    optim = Descent(learningrate)

    s = System(numParticles, 
        numDimensions, 
        hamiltonian, 
        omega=harmonicOscillatorFrequency, 
        interacting = interactingParticles)

    #Add the wavefunction elements:
    addWaveFunctionElement(s, SlaterMatrix( s ))
    addWaveFunctionElement(s, Gaussian( 1.0 ))
    # addWaveFunctionElement(s, NN(s, 12, 12, "tanh"))

    # addWaveFunctionElement(s, Jastrow(s))
    # addWaveFunctionElement(s, PadeJastrow( s; beta = 5000.0 ))
    # addWaveFunctionElement(s, RBM(s, 4, 1.0))
    # println(s)


    numOptimizationSteps = 100
    numMCMCSteps = 100000
    mcmcStepLength = 0.5
    runVMC!(s, numOptimizationSteps, numMCMCSteps, mcmcStepLength, optim, sampler = "is", writeToFile = true)
end

end