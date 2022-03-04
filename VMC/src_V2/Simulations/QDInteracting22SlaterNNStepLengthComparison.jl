module QDInteracting22SlaterNNStepLengthComparison

include("../MLVMC.jl")
using .MLVMC

stepLengths = [10.0]#1.0, 0.1, 0.01, 0.001]

for stepLength in stepLengths
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
    addWaveFunctionElement(s, NN(s, 20, 20, "sigmoid"))

    # addWaveFunctionElement(s, Jastrow(s))
    # addWaveFunctionElement(s, PadeJastrow( s; beta = 1.0 ))
    # addWaveFunctionElement(s, RBM(s, 4, 1.0))
    # println(s)

    numOptimizationSteps = 1000
    numMCMCSteps = 1000
    runVMC!(s, numOptimizationSteps, numMCMCSteps, stepLength, optim, sampler = "is", writeToFile = false)
    mc, _ =  runMetropolis!(s, 
                        2^21,  
                        stepLength, 
                        sampler="is", 
                        writeToFile = true, 
                        calculateOnebody = false)
    println(mc)
end

end