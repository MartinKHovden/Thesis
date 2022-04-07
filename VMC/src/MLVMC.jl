"""
# MLVMC.jl

MLVMC is a package for doing Variational Monte Carlo (VMC) simulations where
the wave function can consist of neural networks. This is where the the ML 
name comes from. It is written with the goal of making it easy to 
extend by others, as well being easy to understand. In addition, 
measures are taken to get the code to run as fast as possible, without 
sacrifising readability.  
"""
module MLVMC 

export System, addWaveFunctionElement
export SlaterMatrix, Jastrow, PadeJastrow, Gaussian, GaussianSimple, RBM, NN
export Descent, ADAM, Momentum
export runVMC!, runMetropolis!

include("Wavefunctions/wavefunction.jl")
include("Wavefunctions/slater.jl")
include("Wavefunctions/gaussian.jl")
include("Wavefunctions/gaussianSimple.jl")
include("Wavefunctions/jastrow.jl")
include("Wavefunctions/padeJastrow.jl")
include("Wavefunctions/rbm.jl")
include("Wavefunctions/nn.jl")
include("system.jl")
include("Hamiltonians/harmonicOscillator.jl")
include("Samplers/metropolis.jl")
include("VMC/vmc.jl")

using OrderedCollections
using Random
using Flux: Descent, ADAM, Momentum, RADAM
using .wavefunction
using .slater
using .gaussian
using .gaussianSimple
using .jastrow
using .padeJastrow
using .rbm
using .nn
using .system
using .harmonicOscillator
using .metropolis
using .vmc



end