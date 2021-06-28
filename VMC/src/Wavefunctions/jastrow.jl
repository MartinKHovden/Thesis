module jastrow 

export jastrowComputeRatio, jastrowComputeGradient, jastrowComputeLaplacian
export jastrowComputeParameterGradient, jastrowUpdateDistanceMatrix
export jastrowWavefunction

using LinearAlgebra

"""
    jastrowComputeRatio(system, oldPosition, particleMoved)

Computes the ratio for the Jastrow part of the wavefunction. 
"""
function jastrowComputeRatio(system, oldPosition, particleMoved)
    positionDifferenceSum = 0
    newPosition = system.particles
    kappa = system.jastrowFactor.kappa
    for j=1:system.numParticles
        if j != particleMoved
            newDifference = newPosition[j, :] - newPosition[particleMoved, :]
            newDistance = sqrt(dot(newDifference, newDifference))

            oldDifference = oldPosition[j, :] - oldPosition[particleMoved, :]
            oldDistance = sqrt(dot(oldDifference, oldDifference))

            positionDifferenceSum += kappa[particleMoved, j]*(newDistance - oldDistance)
        end 
    end
    ratio = exp(2*positionDifferenceSum)
    return ratio
end 

""" 
    jastrowComputeGradient(system, particleNum)

Computes the gradient of the Jastrow part of the wavefunction. 
"""
function jastrowComputeGradient(system, particleNum)
    numParticles = system.numParticles
    numDimensions = system.numDimensions
    particles = system.particles
    kappa = system.jastrowFactor.kappa
    gradient = zeros(numDimensions)

    # display(kappa)

    # for dim=1:numDimensions
    #     for j=1:numParticles
    #         if j!=particleNum
    #             gradient[dim] += kappa[particleNum, j]*(particles[particleNum, dim] - particles[j, dim])/system.jastrowFactor.distanceMatrix[particleNum, j] 
    #         end 
    #     end 
    # end
    for j=1:numParticles 
        if j != particleNum
            difference = particles[particleNum, :] - particles[j, :]
            distance = sqrt(dot(difference, difference))
            gradient += kappa[particleNum, j]*difference./distance
        end
    end

    return gradient
end 

"""
    jastrowComputeLaplacian(system)

Computes the Laplacian of the Jastrow part of the wavefunction. 
"""
function jastrowComputeLaplacian(system, i)
    numParticles = system.numParticles
    numDimensions = system.numDimensions
    particles = system.particles
    kappa = system.jastrowFactor.kappa

    laplacian = 0

    for j=1:numParticles
        if i!=j
            difference = particles[i, :] - particles[j, :]
            distance = sqrt(dot(difference, difference))
            for k=1:numDimensions
                laplacian += (kappa[i,j]/distance)*(1 - (difference[k]/distance)^2)
            end
        end
    end 

    return laplacian
end 

function jastrowComputeParameterGradient(system)
    return system.jastrowFactor.distanceMatrix
end 

function jastrowUpdateDistanceMatrix(system)
    numParticles = system.numParticles
    distanceMatrix = zeros(numParticles, numParticles)
    particles = system.particles
    for i=1:numParticles
        for j=i:numParticles 
            difference = particles[i, :] - particles[j, :]
            distance = sqrt(dot(difference, difference))
            distanceMatrix[i,j] = distance
        end 
    end 
    system.jastrowFactor.distanceMatrix[:,:] = distanceMatrix + distanceMatrix'
end

function jastrowWavefunction(system)
    expArgument = 0
    for i=1:system.numParticles
        for j=i+1:system.numParticles
            expArgument += system.jastrowFactor.kappa[i,j]*system.jastrowFactor.distanceMatrix[i,j]
        end 
    end 
    return exp(expArgument)
end 

end