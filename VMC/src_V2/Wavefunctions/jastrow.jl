module jastrow

export Jastrow, computeRatio, computeGradient, computeLaplacian, updateElement!, computeParameterGradient, computeDriftForce

using ..wavefunction
using Random
using LinearAlgebra

mutable struct Jastrow 
    variationalParameter::Array{Float64, 2}
    variationalParameterGradient::Array{Float64, 2}

    distanceMatrix::Array{Float64, 2}

    function Jastrow(system)
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

        rng = MersenneTwister(123)
        kappa = randn(rng, Float64, (numParticles, numParticles))
        kappa = 0.5*(kappa + kappa')

        return new(kappa, zeros(size(kappa)), distanceMatrix)
    end
end

function wavefunction.computeRatio(system, wavefunctionElement::Jastrow, particleToUpdate, coordinateToUpdate, oldPosition)
    return jastrowComputeRatio(system, wavefunctionElement, oldPosition, particleToUpdate)
end

function jastrowComputeRatio(system, jastrow::Jastrow, oldPosition, particleMoved)
    positionDifferenceSum = 0
    newPosition = system.particles
    kappa = jastrow.variationalParameter
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

function wavefunction.computeGradient(system, wavefunctionElement::Jastrow)
    numParticles = system.numParticles
    numDimensions = system.numDimensions
    gradient = zeros(numParticles*numDimensions)
    for particle=1:numParticles
        gradient[(particle-1)*numDimensions + 1: (particle-1)*numDimensions + numDimensions] = jastrowComputeGradient(system, wavefunctionElement, particle)
    end
    return gradient
end

function jastrowComputeGradient(system, jastrow::Jastrow, particleNum)
    particles = system.particles
    numParticles = system.numParticles
    numDimensions = system.numDimensions
    
    kappa = jastrow.variationalParameter
    gradient = zeros(numDimensions)

    for j=1:numParticles 
        if j != particleNum
            difference = particles[particleNum, :] - particles[j, :]
            distance = sqrt(dot(difference, difference))
            gradient += kappa[particleNum, j]*difference./distance
        end
    end

    return gradient
end 

function wavefunction.computeLaplacian(system, wavefunctionElement::Jastrow)
    numParticles = system.numParticles
    laplacian = 0
    for i=1:numParticles
        laplacian += jastrowComputeLaplacian(system, wavefunctionElement, i)
    end
    return laplacian
end 

function jastrowComputeLaplacian(system, jastrow::Jastrow, i)
    numParticles = system.numParticles
    numDimensions = system.numDimensions
    particles = system.particles
    kappa = jastrow.variationalParameter

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

function wavefunction.computeParameterGradient(system, wavefunctionElement::Jastrow)
    return wavefunctionElement.distanceMatrix
end

function wavefunction.computeDriftForce(system, element::Jastrow)
    return 2*jastrowComputeGradient(system, element, particleNum)[dimension]
end

function wavefunction.updateElement!(system, wavefunctionElement::Jastrow, particle::Int64)
    jastrowUpdateDistanceMatrix!(system, wavefunctionElement)
end

function jastrowUpdateDistanceMatrix!(system, jastrow::Jastrow)
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
    jastrow.distanceMatrix[:,:] = distanceMatrix + distanceMatrix'
end


end

