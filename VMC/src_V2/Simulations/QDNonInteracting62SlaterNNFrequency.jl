module QDNonInteracting62SlaterNNFrequency

include("../MLVMC.jl")
using .MLVMC

for frequency in [0.2, 0.4, 0.6, 0.8, 1.0]
    numParticles = 6
    numDimensions = 2
    hamiltonian = "quantumDot" # Use "quantumDot" or "calogeroSutherland" or "bosons"
    harmonicOscillatorFrequency = frequency
    interactingParticles = false
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
    addWaveFunctionElement(s, NN(s, 12, 12, "tanh"))
    
    numOptimizationSteps = 100
    numMCMCSteps = 1000
    mcmcStepLength = 0.05
    runVMC!(s, numOptimizationSteps, numMCMCSteps, mcmcStepLength, optim, sampler = "is", writeToFile = false)

    @time runMetropolis!(s, 
                2^18,  
                0.5, 
                sampler="is", 
                writeToFile = true, 
                calculateOnebody = false)
end

end