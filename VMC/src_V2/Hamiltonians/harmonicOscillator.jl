module harmonicOscillator 

export computeLocalEnergy

using LinearAlgebra
using ..slater
using ..gaussian

"""
    computeLocalEnergy(system, iteration)

Compute the local energy of the system. 
"""
function computeLocalEnergy(system, iteration)
    particleCoordinates = system.particles
    numParticles = system.numParticles
    numDimensions = system.numDimensions
    numWavefunctionElements = length(system.wavefunctionElements)
    omega = system.omega

    localEnergy = 0
    harmonicTerm = 0

    gradients = zeros(numWavefunctionElements, numParticles*numDimensions)
    laplacian = 0

    for elementNum=1:numWavefunctionElements
        gradients[elementNum,:] = computeGradient(system, system.wavefunctionElements[elementNum])
        laplacian += computeLaplacian(system, system.wavefunctionElements[elementNum])
    end

    localEnergy += sum((sum(gradients, dims=1)).^2)
    localEnergy += laplacian

    interactionTerm = 0
    if system.interacting
        interactionTerm = computeParticleInteraction(system, iteration)
    end

    harmonicTerm = omega*omega*sum(system.particles.^2)

    return -0.5*localEnergy + 0.5*harmonicTerm + interactionTerm
end

function computeParticleInteraction(system, iteration)
    if system.hamiltonian == "quantumDot"
        return computeParticleInteractionQuantumDot(system)
    elseif system.hamiltonian == "calogeroSutherland"
        return computeParticleInteractionCalogeroSutherland(system, iteration)
    end
end

function computeParticleInteractionQuantumDot(system)
    interaction = 0
    numParticles = system.numParticles
    particles = system.particles
    for i=1:numParticles
        for j=i+1:numParticles
            difference = particles[i,:] - particles[j,:]
            distance = sqrt(dot(difference, difference))
            interaction += 1.0/distance 
        end 
    end
    return interaction
end

function computeParticleInteractionCalogeroSutherland(system, iteration)
    interaction = 0
    numParticles = system.numParticles
    particles = system.particles
    beta = system.beta
    for i=1:numParticles
        for j=i+1:numParticles
            difference = particles[i,:] - particles[j,:]
            distance = dot(difference, difference)
            interaction += beta*(beta-1)/(distance) 
        end 
    end
    return min(interaction, (0.001*iteration)^2)
end

end