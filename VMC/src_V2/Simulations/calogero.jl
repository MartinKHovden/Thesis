module calogero
include("../MLVMC.jl")
using .MLVMC

#Set up the system:
for stepLength in [0.01, 0.001]
        numParticles = 2
        numDimensions = 1
        hamiltonian = "calogeroSutherland" # Use "quantumDot" or "calogeroSutherland" or "bosons"
        harmonicOscillatorFrequency = 1.0
        interactingParticles = true
        s = System(numParticles, 
                numDimensions, 
                hamiltonian, 
                omega=harmonicOscillatorFrequency, 
                interacting = interactingParticles)

        #Add the wavefunction elements:
        # addWaveFunctionElement(s, SlaterMatrix( s ))
        addWaveFunctionElement(s, Gaussian( 1.0 ))
        # addWaveFunctionElement(s, GaussianSimple( 0.5 ))

        # addWaveFunctionElement(s, Jastrow(s))
        addWaveFunctionElement(s, NN(s, 20, 20, "sigmoid"))
        # addWaveFunctionElement(s, PadeJastrow(s, beta=1.0))
        # addWaveFunctionElement(s, RBM(s, 2, 1.0))
        # @time runMetropolis!(s, 
        #                 100000,  
        #                 0.0005, 
        #                 optimizationIteration = 1000,
        #                 sampler="is", 
        #                 writeToFile = true, 
        #                 calculateOnebody = false)

        # #Set up the optimiser from Flux: 
        learningrate = 0.01
        optim = ADAM(learningrate)

        # #Set up and run the VMC-calculation:
        numOptimizationSteps = 10000
        numMCMCSteps = 2^10
        mcmcStepLength = stepLength
        runVMC!(s, numOptimizationSteps, numMCMCSteps, mcmcStepLength, optim, sampler = "is", writeToFile = true, sortInput=true)
        @time runMetropolis!(s, 
            2^20,  
            stepLength, 
            sampler="is", 
            writeToFile = true, 
            calculateOnebody = true)

end
end