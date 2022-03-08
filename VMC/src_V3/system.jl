module system

# TODO: Fix code so that I dont have to index to find the slater determinant.
#       Possible solution: Only allow slater to be added as first element.  

export System
export initializeSystem
export addWaveFunctionElement

using OrderedCollections
using Random

using ..slater

mutable struct System
    numParticles::Int64
    numDimensions::Int64
    particles::Array{Float64, 2}
    omega::Float64

    wavefunctionElements::OrderedSet
    slaterInWF::Bool                     # Is the slater element part of the wave function? 

    hamiltonian::String
    beta::Float64

    interacting::Bool

    iteration::Int

    function System(numParticles::Int64, 
                numDimensions::Int64, 
                hamiltonian::String; 
                omega::Float64=1.0, 
                interacting = false, 
                beta = 2.0)
        os = OrderedSet()
        particles = initializeParticlesNormalDist(numParticles, numDimensions)
        return new(numParticles, 
                numDimensions, 
                particles, 
                omega, 
                os, 
                false, 
                hamiltonian, 
                beta, 
                interacting,
                0)
    end
end

function initializeParticlesNormalDist(numParticles::Int64, numDimensions::Int64)
    rng = MersenneTwister(1234)
    particles = 0.5*randn(rng, Float64, (numParticles, numDimensions))
    return particles
end

function addWaveFunctionElement(system::System, wavefunctionElement)
    push!(system.wavefunctionElements, wavefunctionElement)
    if typeof(wavefunctionElement) == SlaterMatrix
        system.slaterInWF = true
    end
end

#End module
end