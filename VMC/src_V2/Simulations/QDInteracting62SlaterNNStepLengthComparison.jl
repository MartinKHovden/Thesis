module QDInteracting62SlaterNNStepLengthComparison

include("../MLVMC.jl")
using .MLVMC

stepLengths = [0.0005, 0.00005]

for stepLength in stepLengths
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
    addWaveFunctionElement(s, Gaussian( 0.6 ))
    addWaveFunctionElement(s, NN(s, 40, 40, "sigmoid"))

    # addWaveFunctionElement(s, Jastrow(s))
    # addWaveFunctionElement(s, PadeJastrow( s; beta = 1.0 ))
    # addWaveFunctionElement(s, RBM(s, 4, 1.0))
    # println(s)

    numOptimizationSteps = 500
    numMCMCSteps = 10000
    runVMC!(s, numOptimizationSteps, numMCMCSteps, stepLength, optim, sampler = "is", writeToFile = true)
    mc, _ =  runMetropolis!(s, 
                        2^20,  
                        stepLength, 
                        sampler="is", 
                        writeToFile = false, 
                        calculateOnebody = false)
    println(mc)
end

end