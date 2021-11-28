module main

include("MLVMC.jl")
using .MLVMC

#Set up the system:
numParticles = 2
numDimensions = 2
hamiltonian = "quantumDot"
harmonicOscillatorFrequency = 1.0
interactingParticles = false
s = System(numParticles, numDimensions, hamiltonian, omega=harmonicOscillatorFrequency, interacting = interactingParticles)

#Add the wavefunction elements:
addWaveFunctionElement(s, SlaterMatrix(s))
addWaveFunctionElement(s, Gaussian(1.0))
# addWaveFunctionElement(s, Jastrow(s))
addWaveFunctionElement(s, NN(s, 2, 2, "sigmoid"))
# addWaveFunctionElement(s, RBM(s, 2, 1.0))
# @time runMetropolis!(s, 100000, 0.5)

# println(s)

# #Set up the optimiser from Flux: 
learningrate = 0.5
optim = Descent(learningrate)

# #Set up and run the VMC-calculation:
numOptimizationSteps = 100
numMCMCSteps = 100000
mcmcStepLength = 0.5
runVMC!(s, numOptimizationSteps, numMCMCSteps, mcmcStepLength, optim)

end


# include("Wavefunctions/wavefunction.jl")
# include("Wavefunctions/slater.jl")
# include("Wavefunctions/gaussian.jl")
# include("Wavefunctions/jastrow.jl")
# include("Wavefunctions/rbm.jl")
# include("system.jl")
# include("Hamiltonians/harmonicOscillator.jl")
# include("Samplers/metropolis.jl")
# include("VMC/vmc.jl")

# using OrderedCollections
# using Random
# using Flux: Descent, ADAM, Momentum
# using .wavefunction
# using .slater
# using .gaussian
# using .jastrow
# using .rbm
# using .system
# using .harmonicOscillator
# using .metropolis
# using .vmc