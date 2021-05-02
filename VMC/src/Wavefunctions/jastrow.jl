module jastrow 

export jastrowComputeRatio, jastrowComputeGradient, jastrowComputeLaplacian

using LinearAlgebra

function jastrowComputeRatio(system, oldPosition, particleMoved)
    positionDifferenceSum = 0
    newPosition = system.particles
    for i=1:system.numParticles
        if i != particleMoved
            newDifference = newPosition[i, :] - newPosition[particleMoved, :]
            newDistance = sqrt(dot(newDifference, newDifference))

            oldDifference = oldPosition[i, :] - oldPosition[particleMoved, :]
            oldDistance = sqrt(dot(oldDifference, oldDifference))

            positionDifferenceSum += system.beta*(newDistance - oldDistance)
        end 
    end

    ratio = exp(2*positionDifferenceSum)

    return ratio
end 

function jastrowComputeGradient(system, particleNum)
    numParticles = system.numParticles
    numDimensions = system.numDimensions
    particles = system.particles
    beta = system.beta

    gradient = zeros(numDimensions)

    for i=1:numParticles 
        if i != particleNum
            difference = particles[particleNum, :] - particles[i, :]
            distance = sqrt(dot(difference, difference))
            gradient += beta*difference./distance
        end
    end

    return gradient
end 

function jastrowComputeLaplacian(system)
    numParticles = system.numParticles
    numDimensions = system.numDimensions
    particles = system.particles
    beta = system.beta

    laplacian = 0

    for i=1:numParticles
        for j=1:numParticles
            if i!=j
                difference = particles[i, :] - particles[j, :]
                distance = sqrt(dot(difference, difference))
                for k=1:numDimensions
                    laplacian += (beta/distance)*(1 - (difference[k]/distance)^2)
                end
            end
        end 
    end 
    return laplacian
end 

end