module main

include("MLVMC.jl")
using .MLVMC

#Set up the system:
numParticles = 6
numDimensions = 2
hamiltonian = "quantumDot" # Use "quantumDot" or "calogeroSutherland" or "bosons"
harmonicOscillatorFrequency = 1.0
interactingParticles = true

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

for mcmcStepLength in (0.5, 0.05, 0.005, 0.0005)
        s = System(numParticles, 
        numDimensions, 
        hamiltonian, 
        omega=harmonicOscillatorFrequency, 
        interacting = interactingParticles)

        #Add the wavefunction elements:
        addWaveFunctionElement(s, SlaterMatrix( s ))
        addWaveFunctionElement(s, Gaussian( 0.6 ))
        # addWaveFunctionElement(s, Jastrow(s))
        addWaveFunctionElement(s, NN(s, 20, 10, "sigmoid"))
        # addWaveFunctionElement(s, RBM(s, 4, 1.0))


        numOptimizationSteps = 400
        numMCMCSteps = 10000
        # mcmcStepLength = 0.01
        runVMC!(s, numOptimizationSteps, numMCMCSteps, mcmcStepLength, optim, sampler = "is", writeToFile = false)
        println(last(s.wavefunctionElements))
        @time runMetropolis!(s, 
                        2^18,  
                        mcmcStepLength, 
                        optimizationIteration = 1000,
                        sampler="is", 
                        writeToFile = true, 
                        calculateOnebody = false)
end

end

