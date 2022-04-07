module hydroSlater 

export Gaussian, computeRatio, computeGradient, computeLaplacian, updateElement!, computeParameterGradient, computeDriftForce

using ..wavefunction

"""
    Gaussian

Stores the variational parameter and its gradient for the Gaussian part of the 
wave function.

# Fields 
- `variationalParameter`: Stores the variational parameter as an array of arrays. 
- `variationalParameterGradient`: Stores the gradient of the variational parameters. 
"""
mutable struct Gaussian
    variationalParameter::Array
    variationalParameterGradient::Array

    function Gaussian(alpha)
        g = new([[alpha]], [[0.0]])
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
u
# Returns
- `Float`: The ratio between the current and previous squared wave function value. 
"""
function wavefunction.computeRatio(system, wavefunctionElement::Gaussian, particleToUpdate, coordinateToUpdate, oldPosition)
    return exp(system.omega*wavefunctionElement.variationalParameter[1][1]*(oldPosition[particleToUpdate,coordinateToUpdate]^2 - system.particles[particleToUpdate,coordinateToUpdate]^2))
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
function wavefunction.computeGradient(system, wavefunctionElement::Gaussian)
    numParticles = system.numParticles
    numDimensions = system.numDimensions
    gradient = zeros(numParticles*numDimensions)
    for particle=1:numParticles
        gradient[(particle-1)*numDimensions + 1: (particle-1)*numDimensions + numDimensions] = slaterGaussianComputeGradient(system, wavefunctionElement, particle)
    end
    return gradient
end

function slaterGaussianComputeGradient(system, gaussian::Gaussian, particle_num)
    coordinates = system.particles[particle_num,:]
    omega = system.omega 
    alpha = gaussian.variationalParameter[1][1]
    grad = -omega*alpha*coordinates
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
function wavefunction.computeLaplacian(system, wavefunctionElement::Gaussian)
    return slaterGaussianComputeLaplacian(system, wavefunctionElement)
end


function slaterGaussianComputeLaplacian(system, gaussian::Gaussian)
    numParticles = system.numParticles
    numDimensions = system.numDimensions 
    alpha = gaussian.variationalParameter[1][1] 
    omega = system.omega
    return -alpha*omega*numDimensions*numParticles
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
function wavefunction.computeParameterGradient(system, wavefunctionElement::Gaussian)
    return  [[slaterGaussianComputeParameterGradient(system)]]
end

function slaterGaussianComputeParameterGradient(system)
    return -0.5*system.omega*sum(system.particles.^2)
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
function wavefunction.computeDriftForce(system, element::Gaussian, particleToUpdate, coordinateToUpdate)
    return 2*slaterGaussianComputeGradient(system, element, particleToUpdate)[coordinateToUpdate]
end

function wavefunction.updateElement!(system, wavefunctionElement::Gaussian, particle::Int64)
end

end