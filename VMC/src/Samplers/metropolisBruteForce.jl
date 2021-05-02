module metropolisBruteForce

export metropolisStepBruteForce

# include("../Wavefunctions/slaterDeterminant.jl")

using ..initializeSystem
using ..slaterDeterminant
using ..jastrow
using ..neuralNetwork

""" 
    metropolisBruteForce(stepLength, system)

Function for updating the position doing one step with the Metropolis algorithm. 
"""
function metropolisStepBruteForce(stepLength, system)
    numParticles = system.numParticles 
    numDimensions = system.numDimensions

    # Chooses one coordinate randomly to update.
    coordinateToUpdate::Int64 = rand(1:numDimensions)
    particleToUpdate::Int64 = rand(1:numParticles)

    # Update the coordinate:
    oldPosition = copy(system.particles)
    system.particles[particleToUpdate, coordinateToUpdate] += (rand(Float64) - 0.5)*stepLength
    # system.nqs.x[:] = reshape(system.particles', 1,:)

    # Update the slater matrix:
    slaterMatrixUpdate(system, particleToUpdate)

    U = rand(Float64)

    ratio, R = computeRatio(system, particleToUpdate, coordinateToUpdate, oldPosition)

    if U < ratio
        inverseSlaterMatrixUpdate(system, particleToUpdate, (R))
    else 
        system.particles[particleToUpdate, coordinateToUpdate] = oldPosition[particleToUpdate, coordinateToUpdate]
        slaterMatrixUpdate(system, particleToUpdate)
        # system.nqs.x[:] = reshape(system.particles', 1,:)
    end
end

""" 
    computeRatio(system::slater, particleToUpdate, coordinateToUpdate, oldPosition)

Returns the wavefunction ratio for the system with the Slater wavefunction,
given the particle moved, the dimension, and the old position. 
"""
function computeRatio(system::slater, particleToUpdate, coordinateToUpdate, oldPosition)
    R = slaterMatrixComputeRatio(system, particleToUpdate)
    ratioSlaterDeterminant = R^2
    ratioSlaterGaussian = slaterGaussianComputeRatio(system, oldPosition, particleToUpdate, coordinateToUpdate)
    return ratioSlaterDeterminant*ratioSlaterGaussian, R
end 

""" 
    computeRatio(system::slaterJastrow, particleToUpdate, coordinateToUpdate, oldPosition)

Returns the wavefunction ratio for the system with the Slater-Jastrow wavefunction,
given the particle moved, the dimension, and the old position. 
"""
function computeRatio(system::slaterJastrow, particleToUpdate, coordinateToUpdate, oldPosition)
    R = slaterMatrixComputeRatio(system, particleToUpdate)
    ratioSlaterDeterminant = R^2
    ratioSlaterGaussian = slaterGaussianComputeRatio(system, oldPosition, particleToUpdate, coordinateToUpdate)
    ratioJastrow = jastrowComputeRatio(system, oldPosition, particleToUpdate)
    return ratioSlaterDeterminant*ratioSlaterGaussian, R
end 

""" 
    computeRatio(system::slaterNN, particleToUpdate, coordinateToUpdate, oldPosition)

Returns the wavefunction ratio for the system with the Slater-NeuralNetwork wavefunction,
given the particle moved, the dimension, and the old position. 
"""
function computeRatio(system::slaterNN, particleToUpdate, coordinateToUpdate, oldPosition)
    R = slaterMatrixComputeRatio(system, particleToUpdate)
    ratioSlaterDeterminant = R^2
    ratioSlaterGaussian = slaterGaussianComputeRatio(system, oldPosition, particleToUpdate, coordinateToUpdate)
    ratioNN = nnComputeRatio(system, oldPosition)
    return ratioSlaterDeterminant*ratioSlaterGaussian*ratioNN, R
end 

end #MODULE