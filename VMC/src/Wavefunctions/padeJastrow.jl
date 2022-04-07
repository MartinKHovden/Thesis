module padeJastrow

export PadeJastrow, computeRatio, computeGradient, computeLaplacian, updateElement!, computeParameterGradient, computeDriftForce

using ..wavefunction
using Random
using LinearAlgebra

"""
    Jastrow

Stores the variational parameter and its gradient for the Jastrow part of the 
wave function.

# Fields 
- `variationalParameter`: Stores the variational parameter as an array of arrays. 
- `variationalParameterGradient`: Stores the gradient of the variational parameters. 
- `distanceMatrix`: Stores the distance between each particle in the system. 
"""
mutable struct PadeJastrow 
    variationalParameter::Array
    variationalParameterGradient::Array

    distanceMatrix::Array{Float64, 2}
    a::Array{Float64,2} # Matrix of spin-constants

    localEnergyPsiParameterDerivativeSum 
    psiParameterDerivativeSum


    function PadeJastrow(system; beta = 0.1)
        particles = system.particles
        numParticles = system.numParticles

        distanceMatrix = zeros(numParticles, numParticles)
        for i=1:numParticles
            for j=i:numParticles 
                difference = particles[i, :] - particles[j, :]
                distance = sqrt(dot(difference, difference))
                distanceMatrix[i,j] = distance
            end 
        end 
        distanceMatrix = distanceMatrix + distanceMatrix'

        a = zeros(numParticles, numParticles)
        numDimensions = system.numDimensions

        for i = 1:numParticles 
            for j = 1:numParticles
                if (i/numParticles <= 0.5 && j/numParticles <= 0.5) ||  (i/numParticles > 0.5 && j/numParticles > 0.5)
                    a[i,j] = 1.0/(numDimensions + 1.0)
                else 
                    a[i,j] = 1.0/(numDimensions - 1.0)
                end
            end
        end 

        return new([[beta]], [[0.0]], distanceMatrix, a, [[0.0]], [[0.0]])
    end
end

function padeJastrowComputeWaveFunction(padeJastrow, particles)
    argument = 0
    for i=1:2
        for j =i+1:2
            difference = particles[i,:] - particles[j,:]
            distance = sqrt(dot(difference, difference))
            argument += padeJastrow.a[i, j]*distance/(1 + padeJastrow.variationalParameter[1][1]*distance)
        end 
    end
    return exp(argument)
end 

function wavefunction.computeRatio(system, 
                                wavefunctionElement::PadeJastrow, 
                                particleToUpdate, 
                                coordinateToUpdate, 
                                oldPosition)
    return padeJastrowComputeRatio(system, wavefunctionElement, oldPosition, particleToUpdate)
end

function padeJastrowComputeRatio(system, padeJastrow::PadeJastrow, oldPosition, particleMoved)
    positionDifferenceSum = 0
    newPosition = system.particles
    a = padeJastrow.a
    beta = padeJastrow.variationalParameter[1][1]
    for j=1:system.numParticles
        if j != particleMoved
            newDifference = newPosition[j, :] - newPosition[particleMoved, :]
            newDistance = sqrt(dot(newDifference, newDifference))

            oldDifference =  oldPosition[j, :] - oldPosition[particleMoved, :]
            oldDistance = sqrt(dot(oldDifference, oldDifference))

            newH = newDistance/(1.0 + beta*newDistance)
            oldH = oldDistance/(1.0 + beta*oldDistance)

            positionDifferenceSum += a[particleMoved, j]*(newH - oldH)
        end
    end
    ratio = exp(2*positionDifferenceSum)
    return ratio
end

function wavefunction.computeGradient(system, wavefunctionElement::PadeJastrow)
    numParticles = system.numParticles
    numDimensions = system.numDimensions
    gradient = zeros(numParticles*numDimensions)
    for particle=1:numParticles
        gradient[(particle-1)*numDimensions + 1: (particle-1)*numDimensions + numDimensions] = padeJastrowComputeGradient(system, wavefunctionElement, particle)
    end
    return gradient
end

function padeJastrowComputeGradient(system, padeJastrow::PadeJastrow, particleNum)
    particles = system.particles
    numParticles = system.numParticles
    numDimensions = system.numDimensions
    
    a = padeJastrow.a
    beta = padeJastrow.variationalParameter[1][1]
    gradient = zeros(numDimensions)

    for j=1:numParticles
        if j != particleNum
            difference = particles[particleNum, :] - particles[j, :]
            distance =  sqrt(dot(difference, difference))
            f = a[particleNum, j]/((1.0 + beta*distance)^2)
            g = difference./distance
            gradient += f*g
        end
    end

    return gradient
end 

function wavefunction.computeLaplacian(system, wavefunctionElement::PadeJastrow)
    numParticles = system.numParticles
    laplacian = 0
    for i=1:numParticles
        laplacian += padeJastrowComputeLaplacian(system, wavefunctionElement, i)
    end
    return laplacian
end 

function padeJastrowComputeLaplacian(system, padeJastrow::PadeJastrow, i)
    numParticles = system.numParticles
    numDimensions = system.numDimensions
    particles = system.particles
    a = padeJastrow.a
    beta = padeJastrow.variationalParameter[1][1]
    laplacian = 0

    for j=1:numParticles
        if i!=j
            difference = particles[i, :] - particles[j, :]
            distance = sqrt(dot(difference, difference))
            f = a[i, j]/((1.0 + beta*distance)^2)
            g = difference./distance
            h = distance/(1.0 + beta*distance)
            for k=1:numDimensions
                laplacian += (f/distance)*(1.0 - (1.0 + 2.0*beta*h)*(g[k]^2))
            end
        end
    end
    return laplacian
end

function wavefunction.computeParameterGradient(system, wavefunctionElement::PadeJastrow)
    numParticles = system.numParticles
    grad = 0
    a = wavefunctionElement.a 
    beta = wavefunctionElement.variationalParameter[1][1]
    particles = system.particles
    distance = wavefunctionElement.distanceMatrix
    for i = 1:numParticles
        for j=i+1:numParticles 
            difference = particles[i, :] - particles[j, :]
            distance = sqrt(dot(difference, difference))
            f = a[i, j]/((1.0 + beta*distance)^2)
            grad += f*dot(difference, difference)
        end 
    end
    
    return [[-grad]]
end

function wavefunction.computeDriftForce(system, 
                                    element::PadeJastrow, 
                                    particleToUpdate, 
                                    coordinateToUpdate)
    return 2*padeJastrowComputeGradient(system, element, particleToUpdate)[coordinateToUpdate]
end

function wavefunction.computeDriftForceFull(system, 
    element::PadeJastrow, 
    particleToUpdate)
return 2*padeJastrowComputeGradient(system, element, particleToUpdate)
end

function wavefunction.updateElement!(system, wavefunctionElement::PadeJastrow, particle::Int64)
    padeJastrowUpdateDistanceMatrix!(system, wavefunctionElement)
end

function padeJastrowUpdateDistanceMatrix!(system, padeJastrow::PadeJastrow)
    numParticles = system.numParticles
    distanceMatrix = zeros(numParticles, numParticles)
    particles = system.particles
    for i=1:numParticles
        for j=i+1:numParticles 
            difference = particles[i, :] - particles[j, :]
            distance = sqrt(dot(difference, difference))
            distanceMatrix[i,j] = distance
        end 
    end 
    padeJastrow.distanceMatrix[:,:] = distanceMatrix + distanceMatrix'
end


end

