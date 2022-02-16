module harmonicOscillator 

export computeLocalEnergy

using LinearAlgebra
using ..slater
using ..gaussian

"""
    computeLocalEnergy(args...)

Computes the local energy of the system.

# Arguments
- `system`: The system struct.
- `iteration`: MCMC-iteration. Used for gradually integrating the potential if needed. 

# Returns
- `Float`: The local energy of the system. 
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

    if system.hamiltonian == "helium"
        harmonicTerm = 0
        for i=1:numParticles
            harmonicTerm -= 2.0/sqrt(dot(system.particles[i,:], system.particles[i,:]))
        end
        harmonicTerm*=2.0
    end

    return -0.5*localEnergy + 0.5*harmonicTerm + interactionTerm
end

"""
    computeParticleInteraction(args...)

Computes the interaction part of the local energy. Can compute the interactions 
for two different hamiltonians: quantum dots and Calogero-Sutherland. 

# Arguments
- `system`: The system struct.
- `iteration`: MCMC-iteration. Used for gradually integrating the potential if needed. 

# Returns
- `Float`: The interaction part of the local energy. 
"""
function computeParticleInteraction(system, iteration)
    if system.hamiltonian == "quantumDot"
        return computeParticleInteractionQuantumDot(system)
    elseif system.hamiltonian == "calogeroSutherland"
        return computeParticleInteractionCalogeroSutherland(system, iteration)
    elseif system.hamiltonian == "bosons"
        return computeParticleInteractionBosons(system)
    elseif system.hamiltonian == "helium"
        return computeParticleInteractionHelium(system)
    else 
        println("The hamiltonian for this system is not implemented")
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
    # println(iteration)
    return min(interaction, (0.001*iteration)^2)
end

function computeParticleInteractionBosons(system)
    a = 2.0
    interaction = 0
    numParticles = system.numParticles
    particles = system.particles
    beta = system.beta
    for i=1:numParticles
        for j=i+1:numParticles
            difference = particles[i,:] - particles[j,:]
            distance = sqrt(dot(difference, difference))
            if distance <= a
                interaction += 1000000.0
            else 
                interaction += 0.0
            end
        end 
    end
    return interaction
end

function computeParticleInteractionHelium(system)
    # println("Here")
    interaction = computeParticleInteractionQuantumDot(system)
    return interaction
end

function computeParticleInteractionHydrogenMolecule(system)
    nucleus1 = [-1.0,0]
    nucleus2 = [1.0,0]
end

end