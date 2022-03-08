module main

include("MLVMC.jl")
using .MLVMC

#Set up the system:
numParticles = 6
numDimensions = 2
hamiltonian = "quantumDot" # Use "quantumDot" or "calogeroSutherland" or "bosons"
harmonicOscillatorFrequency = 0.0#1.0
interactingParticles = true



# #Set up the optimiser from Flux: 

learningrate = 0.1#0.01
optim = ADAM(learningrate)

s = System(numParticles, 
    numDimensions, 
    hamiltonian, 
    omega=harmonicOscillatorFrequency, 
    interacting = interactingParticles)

#Add the wavefunction elements:
addWaveFunctionElement(s, SlaterMatrix( s ))
addWaveFunctionElement(s, Gaussian( 1.0 ))
addWaveFunctionElement(s, NN(s, 40, 40, "sigmoid"))

# println(s)

# addWaveFunctionElement(s, Jastrow(s))
# addWaveFunctionElement(s, PadeJastrow( s; beta = 5000.0 ))
# addWaveFunctionElement(s, RBM(s, 4, 1.0))
# println(s)


numOptimizationSteps = 100
numMCMCSteps = 5000
mcmcStepLength = 0.1#0.001

# @time runMetropolis!(s, 
#                 10,  
#                 0.5, 
#                 sampler="is", 
#                 writeToFile = false, 
#                 calculateOnebody = false)

runVMC!(s, numOptimizationSteps, numMCMCSteps, mcmcStepLength, optim, sampler = "is", writeToFile = true)
@time runMetropolis!(s, 
                        2^22,  
                        mcmcStepLength, 
                        sampler="is", 
                        writeToFile = true, 
                        calculateOnebody = true)



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

