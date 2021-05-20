module metropolisBruteForce

export metropolisStepBruteForce, runVMC

# include("../Wavefunctions/slaterDeterminant.jl")

using ..initializeSystem
using ..slaterDeterminant
using ..jastrow
using ..neuralNetwork
using ..harmonicOscillator
using Flux:params

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
function runMetropolisBruteForce(system::slater, num_mc_iterations, step_length; burn_in = 0.01)
    local_energy_sum::Float64 = 0.0

    local_energy_psi_parameter_derivative_sum = 0
    psi_parameter_derivative_sum = 0

    local_energies::Array{Float64, 1} = zeros(Float64, Int(num_mc_iterations))

    start = time()

    for i = 1:num_mc_iterations
        metropolisStepBruteForce(step_length, system)

        local_energy = computeLocalEnergy(system)
        local_energies[i] = local_energy

        psi_parameter_derivative = slaterGaussianComputeParameterGradient(system)

        if i > burn_in*num_mc_iterations
            local_energy_sum += local_energy
            local_energy_psi_parameter_derivative_sum += local_energy*psi_parameter_derivative
            psi_parameter_derivative_sum += psi_parameter_derivative
        end
    end

    runtime = time() - start

    samples = num_mc_iterations - burn_in*num_mc_iterations

    mc_local_energy = local_energy_sum/samples
    mc_local_energy_psi_derivative_a = local_energy_psi_parameter_derivative_sum/samples
    mc_psi_derivative_a = psi_parameter_derivative_sum/samples
    local_energy_derivative_a = 2*(mc_local_energy_psi_derivative_a - mc_local_energy*mc_psi_derivative_a)

    println(mc_local_energy)

    return mc_local_energy, local_energy_derivative_a
end

"""
    optimizationStep(system::Slater, grad_a::Float64, learning_rate::Float64)

Updates the variational parameter of the system accorind to the gradient 
descent method. 
"""
function optimizationStep(system::slater, grad_a::Float64, learning_rate::Float64)
    system.alpha = system.alpha - learning_rate*grad_a
end

"""
    runVMC()

Runs the full vmc calculation by calling the runMetropolisBruteForce step multiple 
times and updating the variational parameters accordingly. 
"""
function runVMC(system::slater, numVMCIterations::Int64, numMonteCarloIterations, mc_step_length, learning_rate)
    local_energies::Array{Float64, 2} = zeros(Float64, (numVMCIterations, 1))
    for k = 1:numVMCIterations
        local_energy, _grad_a = runMetropolisBruteForce(system, numMonteCarloIterations, mc_step_length)
        optimizationStep(system, _grad_a, learning_rate)
        local_energies[k] = local_energy
        println("Iteration = ", k, "    E = ", local_energy, "   alpha =  ", system.alpha)
    end
    return local_energies
end 

""" 
    runVMc()

Function for running full VMC calculation for a Slater-Jastrow wavefunction.
"""
function runVMC(system::slaterJastrow, numVMCIterations::Int64, numMonteCarloIterations, mc_step_length, learning_rate)
    return 0
end

################################################################################

###   ##    ###   ##    
####  ##    ####  ##
## ## ##    ## ## ##
##  ####    ##  ####
##   ###    ##   ###
##    ##    ##    ##

################################################################################

function runMetropolisBruteForce(system::slaterNN, num_mc_iterations, step_length; burn_in = 0.01)
    numGradients = length(params(system.nn.model))
    
    local_energy_sum::Float64 = 0.0

    local_energy_psi_weights_derivative_sum = params(system.nn.model)[:].*0

    psi_weights_derivative_sum = params(system.nn.model)[:].*0

    local_energies::Array{Float64, 1} = zeros(Float64, Int(num_mc_iterations))

    start = time()

    for i = 1:num_mc_iterations
        metropolisStepBruteForce(step_length, system)

        local_energy = computeLocalEnergy(system)
        local_energies[i] = local_energy

        psi_param_derivative = nnComputeParameterGradient(system)
        psi_weights_derivative = params(system.nn.model).*0

        slater_psi = slaterGaussianWaveFunction(system)*slaterWaveFunction(system)

        for grad=1:numGradients
            psi_weights_derivative[grad] = slater_psi*psi_param_derivative[params(system.nn.model)[grad]]
        end 

        if i > burn_in*num_mc_iterations
            local_energy_sum += local_energy
            local_energy_psi_weights_derivative_sum += local_energy*psi_weights_derivative
            psi_weights_derivative_sum += psi_weights_derivative
        end
    end

    runtime = time() - start

    samples = num_mc_iterations - burn_in*num_mc_iterations

    mc_local_energy = local_energy_sum/samples
    mc_local_energy_psi_derivative_weights = local_energy_psi_weights_derivative_sum/samples

    mc_psi_derivative_weights = psi_weights_derivative_sum/samples

    local_energy_derivative_weights = 2*(mc_local_energy_psi_derivative_weights - mc_local_energy*mc_psi_derivative_weights)

    println(mc_local_energy)

    return mc_local_energy, local_energy_derivative_weights
end

function runVMC(system::slaterNN, numVMCIterations::Int64, numMonteCarloIterations, mc_step_length, learning_rate)
    local_energies::Array{Float64, 2} = zeros(Float64, (numVMCIterations, 1))
    for k = 1:numVMCIterations
        local_energy, _grad_w = runMetropolisBruteForce(system, numMonteCarloIterations, mc_step_length)
        optimizationStep(system, _grad_w, learning_rate)
        local_energies[k] = local_energy
        println("Iteration = ", k, "    E = ", local_energy, "   alpha =  ", system.alpha)
    end
    return local_energies
end 

function optimizationStep(system::slaterNN, grad_W, learning_rate::Float64)
    ps = params(system.nn.model)
    for (p,grad) in zip(ps, grad_W) 
        p -= learning_rate*grad
    end
end

################################################################################

#######       ########     ###      ###
##    ##      ##     ##    ####    ####
##     ##     ##     ##    ## ##  ## ##
##    ##      ##     ##    ##  ####  ##
#######       ########     ##   ##   ##
##   ##       ##     ##    ##        ## 
##    ##      ##     ##    ##        ##
##     ##     ########     ##        ##

################################################################################


end #MODULE