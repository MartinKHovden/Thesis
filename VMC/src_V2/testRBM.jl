module testRBM

include("Wavefunctions/wavefunction.jl")
include("Wavefunctions/slater.jl")
include("Wavefunctions/gaussian.jl")
include("Wavefunctions/jastrow.jl")
include("Wavefunctions/rbm.jl")

include("system.jl")

include("Hamiltonians/harmonicOscillator.jl")

include("Samplers/metropolis.jl")

include("VMC/vmc.jl")


using OrderedCollections
using Random
using Flux: Descent, ADAM


using .wavefunction
using .slater
using .gaussian
using .jastrow
using .rbm

using .system


using .harmonicOscillator

using .metropolis

using .vmc
using .system
using .rbm

s = System(4, 2, "quantumDot", omega=1.0)
addWaveFunctionElement(s, slaterMatrix(s))
# addWaveFunctionElement(s, Gaussian(1.0))
# addWaveFunctionElement(s, Jastrow(s))
addWaveFunctionElement(s, RBM(s, 4, 1.0))
println(s)
initialPosition = [1 2; 3 4; 5 6; 7 8]
newPosition = [1 4; 3 4; 5 6; 7 8]
s.particles = newPosition
s.wavefunctionElements[2].variationalParameter[1][:] = [1; 2; 3; 4; 5; 6; 7 ;8]
s.wavefunctionElements[2].variationalParameter[2][:] = [1; 2; 3; 4]
s.wavefunctionElements[2].variationalParameter[3][:,:] = [1 2 3 4; 1 2 3 4; 1 2 3 4; 1 2 3 4; 1 2 3 4; 1 2 3 4; 1 2 3 4 ;1 2 3 4].*0.01
s.wavefunctionElements[2].h[:] = [1; 1.2; 1.3; 1.4]


println(s.particles)
println(computeRatio(s, s.wavefunctionElements[2], 1, 2, initialPosition))
end