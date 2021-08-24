module simpleJastrow 

export simpleJastrowComputeGradient, simpleJastrowComputeLaplacian
export simpleJastrowComputeParameterGradient, simpleJastrowComputeRatio

using LinearAlgebra

# function simpleJastrowComputeRatio(system, oldPosition, particleMoved)
#     numParticles = system.numParticles
#     newPosition = system.particles
#     beta = system.jastrowFactor.beta[1]
#     ratio = 1.0
#     for j=1:numParticles
#         if j!=particleMoved
#             newDifference = newPosition[j, :] - newPosition[particleMoved, :]
#             newDistance = sqrt(dot(newDifference, newDifference))

#             oldDifference = oldPosition[j, :] - oldPosition[particleMoved, :]
#             oldDistance = sqrt(dot(oldDifference, oldDifference))

#             ratio *= ((newDistance^(2*beta))/(oldDistance^(2*beta)))
#         end
#     end
#     return ratio
# end

function simpleJastrowComputeRatio(system, oldPosition, particleMoved)
    newWavefunctionValue = computePsi(system, system.particles)
    oldWavefunctionValue = computePsi(system, oldPosition)
    return (newWavefunctionValue^2)/(oldWavefunctionValue^2)
end

function computePsi(system, x)
    numParticles = system.numParticles
    psi = 1.0
    for i=1:numParticles
        for j=i+1:numParticles 
            difference = x[i, :] - x[j,:]
            distance = sqrt(dot(difference, difference))
            psi*=distance
        end 
    end
    return psi
end

function simpleJastrowComputeGradient(system, particleNum)
    numParticles = system.numParticles
    numDimensions = system.numDimensions
    particles = system.particles
    beta = system.jastrowFactor.beta[1]
    gradient = zeros(numDimensions)
    for j=1:numParticles 
        if j != particleNum
            # difference = particles[particleNum, :] - particles[j, :]
            # distance = sqrt(dot(difference, difference))
            gradient += [beta/(particles[particleNum,1] - particles[j, 1])]
        end
    end
    return gradient
end 

function simpleJastrowComputeLaplacian(system, particleNum)
    numParticles = system.numParticles
    numDimensions = system.numDimensions
    particles = system.particles
    laplacian = 0
    beta = system.jastrowFactor.beta[1]
    for j=1:numParticles
        if j!=particleNum
            # difference = particles[particleNum, :] - particles[j, :]
            # distance = sqrt(dot(difference, difference))
            for k=1:numDimensions
                laplacian += - beta/((particles[particleNum,1] - particles[j,1])^2)
            end
        end
    end
    return laplacian
end

function simpleJastrowComputeParameterGradient(system)
    numParticles = system.numParticles 
    particles = system.particles
    parameterGradient = 0
    for i = 1:numParticles
        for j = i+1:numParticles
            difference = particles[i,:] - particles[j,:]
            distance = sqrt(dot(difference, difference))
            parameterGradient += log(distance)
        end
    end
    return parameterGradient
end

#END MODULE
end