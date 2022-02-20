module QDInteracting22SlaterNNNumberOfLayersAndActivationFunction

include("../MLVMC.jl")
using .MLVMC

for numHiddenLayers in [5, 10, 15, 20, 25]
    for activationFunction in ["tanh", "sigmoid"]
        numParticles = 2
        numDimensions = 2
        hamiltonian = "quantumDot" # Use "quantumDot" or "calogeroSutherland" or "bosons"
        harmonicOscillatorFrequency = 1.0
        interactingParticles = true



        # #Set up the optimiser from Flux: 

        learningrate = 0.01
        optim = ADAM(learningrate)

        s = System(numParticles, 
            numDimensions, 
            hamiltonian, 
            omega=harmonicOscillatorFrequency, 
            interacting = interactingParticles)

        #Add the wavefunction elements:
        addWaveFunctionElement(s, SlaterMatrix( s ))
        addWaveFunctionElement(s, Gaussian( 1.0 ))
        addWaveFunctionElement(s, NN(s, numHiddenLayers, numHiddenLayers, activationFunction))

        # addWaveFunctionElement(s, Jastrow(s))
        # addWaveFunctionElement(s, PadeJastrow( s; beta = 1.0 ))
        # addWaveFunctionElement(s, RBM(s, 4, 1.0))
        # println(s)

        numOptimizationSteps = 1000
        numMCMCSteps = 1000
        mcmcStepLength = 0.01
        runVMC!(s, numOptimizationSteps, numMCMCSteps, mcmcStepLength, optim, sampler = "is", writeToFile = true)
        @time runMetropolis!(s, 
                            2^20,  
                            mcmcStepLength, 
                            sampler="is", 
                            writeToFile = true, 
                            calculateOnebody = true)
    end
end

end