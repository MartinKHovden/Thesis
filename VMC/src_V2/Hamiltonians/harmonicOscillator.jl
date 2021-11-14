module harmonicOscillator 

export computeLocalEnergy

using ..slater
using ..gaussian

function computeLocalEnergy(system)
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
        interactionTerm = computeParticleInteraction(system)
    end

    harmonicTerm = omega*omega*sum(system.particles.^2)

    return -0.5*localEnergy + 0.5*harmonicTerm + interactionTerm
end

function computeParticleInteraction(system)
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

end