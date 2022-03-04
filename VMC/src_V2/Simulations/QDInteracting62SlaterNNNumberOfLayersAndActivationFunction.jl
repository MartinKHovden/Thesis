module QDInteracting62SlaterNNNumberOfLayersAndActivationFunction

include("../MLVMC.jl")
using .MLVMC

numRuns = 3
layerVariations = [20, 40, 60]

values = zeros(5, length(layerVariations))

for j=1:numRuns
    for i=1:length(layerVariations)
        numParticles = 6
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
        addWaveFunctionElement(s, NN(s, layerVariations[i], layerVariations[i], "sigmoid"))

        # addWaveFunctionElement(s, Jastrow(s))
        # addWaveFunctionElement(s, PadeJastrow( s; beta = 1.0 ))
        # addWaveFunctionElement(s, RBM(s, 4, 1.0))
        # println(s)

        numOptimizationSteps = 1000
        numMCMCSteps = 2000
        mcmcStepLength = 0.0005
        runVMC!(s, numOptimizationSteps, numMCMCSteps, mcmcStepLength, optim, sampler = "is", writeToFile = false)
        mc, _ =  runMetropolis!(s, 
                            2^21,  
                            mcmcStepLength, 
                            sampler="is", 
                            writeToFile = false, 
                            calculateOnebody = false)
        values[j, i] = mc
    end
end

using DelimitedFiles
writedlm("Test2.csv", values)

end