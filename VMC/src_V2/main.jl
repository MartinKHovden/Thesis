module main

include("MLVMC.jl")
using .MLVMC

#Set up the system:
num_particles = 2
num_dimensions = 2
hamiltonian = "quantumDot"
harmonic_oscillator_frequency = 1.0
interacting_particles = false
s = System(num_particles, num_dimensions, hamiltonian, omega=harmonic_oscillator_frequency, interacting = interacting_particles)

#Add the wavefunction elements:
addWaveFunctionElement(s, SlaterMatrix( s ))
addWaveFunctionElement(s, Gaussian( 1.0 ))
# addWaveFunctionElement(s, Jastrow(s))
# addWaveFunctionElement(s, NN(s, 2, 2, "sigmoid"))
# addWaveFunctionElement(s, RBM(s, 2, 1.0))
@time runMetropolis!(s, 100000,  sampler="bf", 0.5, writeToFile = true, calculateOnebody = true)

# println(s)

# #Set up the optimiser from Flux: 
# learningrate = 0.5
# optim = Descent(learningrate)

# #Set up and run the VMC-calculation:
# numOptimizationSteps = 100
# numMCMCSteps = 100000
# mcmcStepLength = 0.5
# runVMC!(s, numOptimizationSteps, numMCMCSteps, mcmcStepLength, optim)

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