module gaussianSimple

export GaussianSimple, computeRatio, computeGradient, computeLaplacian, updateElement!, computeParameterGradient, computeDriftForce

using LinearAlgebra
using ..wavefunction

"""
    Gaussian

Stores the variational parameter and its gradient for the Gaussian part of the 
wave function.

# Fields 
- `variationalParameter`: Stores the variational parameter as an array of arrays. 
- `variationalParameterGradient`: Stores the gradient of the variational parameters. 
"""
mutable struct GaussianSimple
    variationalParameter::Array
    variationalParameterGradient::Array

    localEnergyPsiParameterDerivativeSum 
    psiParameterDerivativeSum
    function GaussianSimple(alpha)
        g = new([[alpha]], [[0.0]], [[0.0]], [[0.0]])
        return g
    end
end

"""
    computeRatio(args...)

Computes the ratio between the current and previous squared wave function value.

# Arguments
- `system`: The system struct.
- `wavefunctionElement::Gaussian`: The wave function element to compute the ratio for. 
- `particleToUpdate`: The particle moved.
- `coordinateToUpdate`: The coordinate moved. 
- `oldPosition`: The old position of the particle that was updated. 

# Returns
- `Float`: The ratio between the current and previous squared wave function value. 
"""
function wavefunction.computeRatio(system, wavefunctionElement::GaussianSimple, particleToUpdate, coordinateToUpdate, oldPosition)
    distanceOld = sqrt(dot(oldPosition[particleToUpdate,:],oldPosition[particleToUpdate,:]))
    distanceNew = sqrt(dot(system.particles[particleToUpdate,:],system.particles[particleToUpdate,:]))

    ratio =  exp(-wavefunctionElement.variationalParameter[1][1]*(distanceNew - distanceOld))

    return ratio
end

"""
    computeGradient(args...)

Computes the gradient of the wave function with respect to all particles 
and coordinates.

# Arguments
- `system`: The system struct.
- `wavefunctionElement::Gaussian`: The wave function element to compute the ratio for. 

# Returns
- `Array{Float}`: An array with the gradient of the wave function with respect 
    to each particle. 
"""
function wavefunction.computeGradient(system, wavefunctionElement::GaussianSimple)
    numParticles = system.numParticles
    numDimensions = system.numDimensions
    gradient = zeros(numParticles*numDimensions)
    for particle=1:numParticles
        gradient[(particle-1)*numDimensions + 1: (particle-1)*numDimensions + numDimensions] = gaussianSimpleComputeGradient(system, wavefunctionElement, particle)
    end
    return gradient
end

function gaussianSimpleComputeGradient(system, gaussian::GaussianSimple, particle_num)
    coordinates = system.particles[particle_num,:]
    alpha = gaussian.variationalParameter[1][1]
    distance = sqrt(dot(coordinates, coordinates))
    grad = -alpha*coordinates/distance
    return grad
end

"""
    computeLaplacian(args...)

Computes the laplacian of the wave function with respect to all particles coordinates.

# Arguments
- `system`: The system struct.
- `wavefunctionElement::Gaussian`: The wave function element to compute the ratio for. 

# Returns
- `Array{Float}`: An array with the laplacian of the wave function with respect 
    to each particle. 
"""
function wavefunction.computeLaplacian(system, wavefunctionElement::GaussianSimple)
    return gaussianSimpleComputeLaplacian(system, wavefunctionElement)
end


function gaussianSimpleComputeLaplacian(system, gaussian::GaussianSimple)
    laplacian = 0
    particles = system.particles
    numParticles = system.numParticles
    numDimensions = system.numDimensions 
    alpha = gaussian.variationalParameter[1][1] 
    for i=1:numParticles
        coordinates = system.particles[i,:]
        denominator = (dot(coordinates, coordinates))^(3/2)
        laplacian += sum(coordinates.^2)/denominator
    end
    return -(numDimensions-1)*alpha*laplacian
end

"""
    computeParameterGradient(args...)

Computes the gradient of the wave function with respect to the variational 
parameter. 

# Arguments
- `system`: The system struct.
- `wavefunctionElement::Gaussian`: The wave function element to compute the ratio for. 

# Returns
- `Array{Float}`: An array with the gradient of the wave function with respect 
    to the variational parameter. 
"""
function wavefunction.computeParameterGradient(system, wavefunctionElement::GaussianSimple)
    return  [[gaussianSimpleComputeParameterGradient(system)]]
end

function gaussianSimpleComputeParameterGradient(system)
    particles = system.particles
    parameterGradient = 0
    numParticles = system.numParticles
    numDimensions = system.numDimensions
    for i=1:numParticles
        coordinates = particles[i, :]
        parameterGradient += sqrt(dot(coordinates, coordinates))
    end
    return -parameterGradient
end 

"""
    computeDriftForce(args...)

Computes the drift force of the wave function with respect to a given coordinate
of a given particle. 

# Arguments
- `system`: The system struct.
- `wavefunctionElement::Gaussian`: The wave function element to compute the ratio for. 
- `particleToUpdate`: The particle moved.
- `coordinateToUpdate`: The coordinate moved. 

# Returns
- `Array{Float}`: The drift force of the wave function with respect to a given 
coordinate of a given particle.
"""
function wavefunction.computeDriftForce(system, element::GaussianSimple, particleToUpdate, coordinateToUpdate)
    return 2*gaussianSimpleComputeGradient(system, element, particleToUpdate)[coordinateToUpdate]
end

function wavefunction.computeDriftForceFull(system, element::GaussianSimple, particleToUpdate)
    return 2*slaterGaussianSimpleComputeGradient(system, element, particleToUpdate)
end

function wavefunction.updateElement!(system, wavefunctionElement::GaussianSimple, particle::Int64)
end

end