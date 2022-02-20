module main

include("MLVMC.jl")
using .MLVMC

#Set up the system:
numParticles = 6
numDimensions = 2
hamiltonian = "quantumDot" # Use "quantumDot" or "calogeroSutherland" or "bosons"
harmonicOscillatorFrequency = 1.0
interactingParticles = false



# #Set up the optimiser from Flux: 

learningrate = 0.1
optim = Descent(learningrate)

s = System(numParticles, 
    numDimensions, 
    hamiltonian, 
    omega=harmonicOscillatorFrequency, 
    interacting = interactingParticles)

#Add the wavefunction elements:
addWaveFunctionElement(s, SlaterMatrix( s ))
addWaveFunctionElement(s, Gaussian( 1.1 ))
# addWaveFunctionElement(s, NN(s, 12, 12, "tanh"))

# addWaveFunctionElement(s, Jastrow(s))
# addWaveFunctionElement(s, PadeJastrow( s; beta = 5000.0 ))
# addWaveFunctionElement(s, RBM(s, 4, 1.0))
# println(s)


numOptimizationSteps = 200
numMCMCSteps = 10000
mcmcStepLength = 0.0005
runVMC!(s, numOptimizationSteps, numMCMCSteps, mcmcStepLength, optim, sampler = "is", writeToFile = false)

# @time runMetropolis!(s, 
#                 100000,  
#                 0.5, 
#                 sampler="is", 
#                 writeToFile = true, 
#                 calculateOnebody = false)


# #Set up and run the VMC-calculation:

# for mcmcStepLength in (0.5, 0.05, 0.005, 0.005)
#         learningrate = 0.1
#         optim = ADAM(learningrate)

#         s = System(numParticles, 
#         numDimensions, 
#         hamiltonian, 
#         omega=harmonicOscillatorFrequency, 
#         interacting = interactingParticles)

#         #Add the wavefunction elements:
#         addWaveFunctionElement(s, SlaterMatrix( s ))
#         addWaveFunctionElement(s, Gaussian( 0.8 ))
#         addWaveFunctionElement(s, Jastrow(s))
#         # addWaveFunctionElement(s, NN(s, 2, 2, "sigmoid"))
#         # addWaveFunctionElement(s, RBM(s, 4, 1.0))


#         numOptimizationSteps = 100
#         numMCMCSteps = 10000
#         # mcmcStepLength = 0.01
#         runVMC!(s, numOptimizationSteps, numMCMCSteps, mcmcStepLength, optim, sampler = "is", writeToFile = false)
#         println(last(s.wavefunctionElements))
#         @time runMetropolis!(s, 
#                         2^18,  
#                         mcmcStepLength, 
#                         sampler="is", 
#                         writeToFile = true, 
#                         calculateOnebody = false)
# end

end

