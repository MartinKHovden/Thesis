module QDInteracting22SlaterNNOptim

include("../MLVMC.jl")
using .MLVMC

for stepLength in [1.0, 0.1, 0.01, 0.001]
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
    addWaveFunctionElement(s, NN(s, 12, 12, "tanh"))

    # addWaveFunctionElement(s, Jastrow(s))
    # addWaveFunctionElement(s, PadeJastrow( s; beta = 1.0 ))
    # addWaveFunctionElement(s, RBM(s, 4, 1.0))
    # println(s)
s

    numOptimizationSteps = 1000
    numMCMCSteps = 1000
    mcmcStepLength = stepLength
    runVMC!(s, numOptimizationSteps, numMCMCSteps, mcmcStepLength, optim, sampler = "is", writeToFile = true)
    @time runMetropolis!(s, 
                        2^20,  
                        mcmcStepLength, 
                        sampler="is", 
                        writeToFile = true, 
                        calculateOnebody = true)
end

end