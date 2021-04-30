module metropolisBruteForce

export metropolisStepBruteForce

# include("../Wavefunctions/slaterDeterminant.jl")

using ..initializeSystem
using ..slaterDeterminant
using ..jastrow

function metropolisStepBruteForce(stepLength, system)
    numParticles = system.numParticles 
    numDimensions = system.numDimensions

    # #Chooses one coordinate randomly to update.
    coordinateToUpdate::Int64 = rand(1:numDimensions)
    particleToUpdate::Int64 = rand(1:numParticles)

    # #Update the coordinate:
    oldPosition = copy(system.particles)
    # println("Coors ", system.particles)
    # old_slater_wf = slaterWaveFunction(system)
    # old_rbm_wf = computePsi(system.nqs)
    # println(old_slater_wf)
    system.particles[particleToUpdate, coordinateToUpdate] += (rand(Float64) - 0.5)*stepLength
    # system.nqs.x[:] = reshape(system.particles', 1,:)

    slaterMatrixUpdate(system, particleToUpdate)
    # new_slater_wf = slaterWaveFunction(system)
    # new_rbm_wf = computePsi(system.nqs)
    # println(new_slater_wf)
    # ratioSlaterDeterminant = (abs(new_slater_wf)^2)/(abs(old_slater_wf)^2)
    # println("\n Analytic = ", (abs(new_slater_wf)^2)/(abs(old_slater_wf)^2))

    # ratioRBM = (abs(new_rbm_wf)^2)/(abs(old_rbm_wf)^2)

    # ratioSlaterDeterminant = slaterMatrixComputeRatio(system, particleToUpdate)

    # println("Fast = ", ratioSlaterDeterminant)

    # if ratioSlaterDeterminant > 10
    #     println(" Analytic = ", ratioSlaterDeterminant)
    #     # println(" Num = ", slaterMatrixComputeRatio(system, particleToUpdate))

    # end
    # ratioSlaterGaussian = slaterGaussianComputeRatio(system, oldPosition, particleToUpdate, coordinateToUpdate)

    U = rand(Float64)

    # ratio = (ratioSlaterDeterminant^2)*ratioSlaterGaussian#*ratioRBM
    ratio, R = computeRatio(system, particleToUpdate, coordinateToUpdate, oldPosition)

    if U < ratio
        # println(1)
        inverseSlaterMatrixUpdate(system, particleToUpdate, (R))

        # system.inverseSlaterMatrixSpinUp[:, :] = inv(system.slaterMatrixSpinUp)
        # system.inverseSlaterMatrixSpinDown[:, :] = inv(system.slaterMatrixSpinDown)
    else 
        # println(2)
        system.particles[particleToUpdate, coordinateToUpdate] = oldPosition[particleToUpdate, coordinateToUpdate]
        slaterMatrixUpdate(system, particleToUpdate)
        # system.nqs.x[:] = reshape(system.particles', 1,:)
        # println(computeLocalEnergy(system))
    end
end

function computeRatio(system::slater, particleToUpdate, coordinateToUpdate, oldPosition)
    R = slaterMatrixComputeRatio(system, particleToUpdate)
    ratioSlaterDeterminant = R^2
    ratioSlaterGaussian = slaterGaussianComputeRatio(system, oldPosition, particleToUpdate, coordinateToUpdate)
    return ratioSlaterDeterminant*ratioSlaterGaussian, R
end 

function computeRatio(system::slaterJastrow, particleToUpdate, coordinateToUpdate, oldPosition)
    R = slaterMatrixComputeRatio(system, particleToUpdate)
    ratioSlaterDeterminant = R^2
    ratioSlaterGaussian = slaterGaussianComputeRatio(system, oldPosition, particleToUpdate, coordinateToUpdate)
    ratioJastrow = jastrowComputeRatio(system, oldPosition, particleToUpdate)
    return ratioSlaterDeterminant*ratioSlaterGaussian, R
end 

end #MODULE