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

"""
    runMetropolisBruteForce()

Runs the full metropolis sampling for a set of parameters. 
"""
function runMetropolisBruteForce(system::Slater)
    local_energy_sum::Float64 = 0.0

    #Initializes the arrays and matrices to save the derivatives and the sums.
    local_energy_psi_derivative_a_sum = 0
    psi_derivative_a_sum = 0

    #Vector to store the energies for each step.
    local_energies::Array{Float64, 1} = zeros(Float64, Int(num_mc_iterations))

    start = time()

    for i = 1:num_mc_iterations
        # Does one step with the brute force method.
        metropolisStepBruteForce(step_length, system)

        # Computes the contribution to Monte carlo estimate of the local energy given the new system configuration.
        local_energy = computeLocalEnergy(system)
        local_energies[i] = local_energy

        # Computes the contribution to the gradients given the new system configuration.
        psi_derivative_a = computePsiParameterDerivative(system)

        # Calculates the estimates of the energy and derivatives. Uses only those after the burn-in period.
        if i > burn_in*num_mc_iterations
            local_energy_sum += local_energy
            local_energy_psi_derivative_a_sum += local_energy*psi_derivative_a
            psi_derivative_a_sum += psi_derivative_a
        end
    end

    runtime = time() - start

    # Updates the final estimates of local energy and gradients.
    samples = num_mc_iterations - burn_in*num_mc_iterations

    mc_local_energy = local_energy_sum/samples
    mc_local_energy_psi_derivative_a = local_energy_psi_derivative_a_sum/samples
    mc_psi_derivative_a = psi_derivative_a_sum/samples
    local_energy_derivative_a = 2*(mc_local_energy_psi_derivative_a - mc_local_energy*mc_psi_derivative_a)

    return mc_local_energy, local_energy_derivative_a
end

"""
    optimizationStep(system::Slater, grad_a::Float64, learning_rate::Float64)

Updates the variational parameter of the system accorind to the gradient 
descent method. 
"""
function optimizationStep(system::Slater, grad_a::Float64, learning_rate::Float64)
    system.alpha = system.alpha - learning_rate*grad_a
end

"""
    runVMC()

Runs the full vmc calculation by calling the runMetropolisBruteForce step multiple 
times and updating the variational parameters accordingly. 
"""
function runVMC(system::Slater, numVMCIterations::Int64, numMonteCarloIterations)
    local_energies::Array{Float64, 2} = zeros(Float64, (numVMCIterations, 1))
    # Loops for running multiple gradient descent steps.
    for k = 1:numVMCIterations
        local_energy, _grad_a = runMetorpolisBruteForce(system, numMonteCarloIterations, mc_burn_in, mc_step_length)
        optimizationStep(system, _grad_a, learning_rate)
        local_energies[k] = local_energy
        println("Iteration = ", k, "    E = ", local_energy)
    end

    return local_energies
end 

end #MODULE