module metropolisImportanceSampling

export metropolisStepImportanceSampling

""" 
    metropolisStepImportanceSampling(stepLength, system)

Function for updating the position doing one step with the Importance Sampling Metropolis algorithm. 
"""
function metropolisStepImportanceSampling(stepLength, system)
    numParticles = system.numParticles 
    numDimensions = system.numDimensions

    # Chooses one coordinate randomly to update.
    coordinateToUpdate::Int64 = rand(1:numDimensions)
    particleToUpdate::Int64 = rand(1:numParticles)

    # Update the coordinate:
    oldPosition = copy(system.particles)

    D = 0.5

    currentDriftForce = computeDriftForce(system, particleToUpdate, coordinateToUpdate)
    system.particles[particleToUpdate, coordinateToUpdate] += D*currentDriftForce*stepLength + randn(Float64)*sqrt(stepLength)

    newDriftForce = computeDriftForce(system, particleToUpdate, coordinateToUpdate)
    # Update the slater matrix:
    slaterMatrixUpdate(system, particleToUpdate)

    U = rand(Float64)

    greens_function = computeGreensFunction(oldPosition, system.particles, particleToUpdate, coordinateToUpdate, currentDriftForce, newDriftForce)

    ratio, R = computeRatio(system, particleToUpdate, coordinateToUpdate, oldPosition)

    if U < greens_function*ratio
        inverseSlaterMatrixUpdate(system, particleToUpdate, (R))
    else 
        system.particles[particleToUpdate, coordinateToUpdate] = oldPosition[particleToUpdate, coordinateToUpdate]
        slaterMatrixUpdate(system, particleToUpdate)
    end
end

function computeDriftForce(system::slater, particleNumber::Int8, dimension::Int8)
    return slaterDeterminantComputeDriftForce(system, particleNumber, dimension)
            + slaterGaussianComputeDriftForce(system, particleNumber, dimension)
end 

function computeDriftForce(system::slaterJastrow, particleNumber::Int8, dimension::Int8)
    return slaterDeterminantComputeDriftForce(system, particleNumber, dimension) 
            + slaterGaussianComputeDriftForce(system, particleNumber, dimension)
            + jastrowComputeDriftForce(system, particleNumber, dimension)
end

function computeDriftForce(system::slaterRBM, particleNumber::Int8, dimension::Int8)
    return slaterDeterminantComputeDriftForce(system, particleNumber, dimension)
            + slaterGaussianComputeDriftForce(system, particleNumber, dimension) 
            + rbmComputeDriftForce(system, particleNumber, dimension)
end 

function computeDriftForce(system::slaterNN, particleNumber::Int8, dimension::Int8)
    return slaterDeterminantComputeDriftForce(system, particleNumber, dimension)
            + slaterGaussianComputeDriftForce(system, particleNumber, dimension) 
            + nnComputeDriftForce(system, particleNumber, dimension)
end 

function computeGreensFunction(oldPosition, newPosition, particleToUpdate, dimension, oldDriftForce, newDriftForce)
    greens_function_argument = (oldPosition[particleToUpdate, coordinateToUpdate] 
                                - system.particles[particleToUpdate, coordinateToUpdate] 
                                - D*stepLength*newDriftForce)^2 
                                - (system.particles[particleToUpdate, coordinateToUpdate] 
                                - oldPosition[particleToUpdate, coordinateToUpdate] 
                                - D*stepLength*currentDriftForce)^2

    greens_function_argument /= (4.0*D*stepLength)
    greens_function = exp(-greens_function_argument)
    return greens_function
end
#END MODULE
end