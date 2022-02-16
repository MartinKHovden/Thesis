module QDNonInteracting22SlaterNN

include("../MLVMC.jl")
using .MLVMC

#Set up the system:
numParticles = 2
numDimensions = 2
hamiltonian = "quantumDot" # Use "quantumDot" or "calogeroSutherland" or "bosons"
harmonicOscillatorFrequency = 1.0
interactingParticles = false



# #Set up the optimiser from Flux: 



# addWaveFunctionElement(s, Jastrow(s))
# addWaveFunctionElement(s, PadeJastrow( s; beta = 5000.0 ))
# addWaveFunctionElement(s, RBM(s, 4, 1.0))
# println(s)

learningrate = 0.1
optim = Descent(learningrate)

s = System(numParticles, 
    numDimensions, 
    hamiltonian, 
    omega=harmonicOscillatorFrequency, 
    interacting = interactingParticles)

#Add the wavefunction elements:
addWaveFunctionElement(s, SlaterMatrix( s ))
addWaveFunctionElement(s, Gaussian( 1.0 ))
addWaveFunctionElement(s, NN(s, 6, 4, "tanh"))

numOptimizationSteps = 100
numMCMCSteps = 10000
mcmcStepLength = 0.05
runVMC!(s, numOptimizationSteps, numMCMCSteps, mcmcStepLength, optim, sampler = "bf", writeToFile = false)


for stepLength in [50.0, 5.0, 0.5, 0.05, 0.005, 0.0005, 0.00005]


    @time runMetropolis!(s, 
                2^18,  
                stepLength, 
                sampler="bf", 
                writeToFile = true, 
                calculateOnebody = false)
end


end