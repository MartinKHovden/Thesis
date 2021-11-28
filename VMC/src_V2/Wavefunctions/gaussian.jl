"""

# Gaussian
## Constructor
```julia 
    Gaussian(alpha)
```

## Description
The Gaussian part 
"""
module gaussian 

export Gaussian, computeRatio, computeGradient, computeLaplacian, updateElement!, computeParameterGradient, computeDriftForce

# using ..system
using ..wavefunction

mutable struct Gaussian
    variationalParameter::Array{Float64, 1}
    variationalParameterGradient::Array{Float64, 1}

    function Gaussian(alpha)
        g = new([alpha], [0])
        return g
    end
end

function wavefunction.computeRatio(system, wavefunctionElement::Gaussian, particleToUpdate, coordinateToUpdate, oldPosition)
    return exp(system.omega*wavefunctionElement.variationalParameter[1]*(oldPosition[particleToUpdate,coordinateToUpdate]^2 - system.particles[particleToUpdate,coordinateToUpdate]^2))
end

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
    alpha = gaussian.variationalParameter[1]
    grad = -omega*alpha*coordinates
    return grad
end

function wavefunction.computeLaplacian(system, wavefunctionElement::Gaussian)
    return slaterGaussianComputeLaplacian(system, wavefunctionElement)
end


function slaterGaussianComputeLaplacian(system, gaussian::Gaussian)
    numParticles = system.numParticles
    numDimensions = system.numDimensions 
    alpha = gaussian.variationalParameter[1] 
    omega = system.omega
    return -alpha*omega*numDimensions*numParticles
end

function wavefunction.computeParameterGradient(system, wavefunctionElement::Gaussian)
    return  [slaterGaussianComputeParameterGradient(system)]
end

function slaterGaussianComputeParameterGradient(system)
    return -0.5*system.omega*sum(system.particles.^2)
end 

function wavefunction.computeDriftForce(system, element::Gaussian, particleToUpdate, coordinateToUpdate)
    return 2*slaterGaussianComputeGradient(system, element, particleToUpdate)[coordinateToUpdate]
end

function wavefunction.updateElement!(system, wavefunctionElement::Gaussian, particle::Int64)
end

end