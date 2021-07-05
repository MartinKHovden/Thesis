module metropolisBruteForce

export metropolisStepBruteForce, runVMC

# include("../Wavefunctions/slaterDeterminant.jl")

using ..initializeSystem
using ..slaterDeterminant
using ..jastrow
using ..neuralNetwork
using ..harmonicOscillator
using ..boltzmannMachine
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

    # Update the slater matrix:
    slaterMatrixUpdate(system, particleToUpdate)

    U = rand(Float64)

    ratio, R = computeRatio(system, particleToUpdate, coordinateToUpdate, oldPosition)

    if U < ratio
        inverseSlaterMatrixUpdate(system, particleToUpdate, (R))
    else 
        system.particles[particleToUpdate, coordinateToUpdate] = oldPosition[particleToUpdate, coordinateToUpdate]
        slaterMatrixUpdate(system, particleToUpdate)
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
    return ratioSlaterDeterminant*ratioSlaterGaussian*ratioJastrow, R
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
    computeRatio(system::slaterRBM, particleToUpdate, coordinateToUpdate, oldPosition)

Returns the wavefunction ratio for the system with the Slater-RBM wavefunction,
given the particle moved, the dimension, and the old position. 
"""
function computeRatio(system::slaterRBM, particleToUpdate, coordinateToUpdate, oldPosition)
    R = slaterMatrixComputeRatio(system, particleToUpdate)
    ratioSlaterDeterminant = R^2
    ratioSlaterGaussian = 1 # slaterGaussianComputeRatio(system, oldPosition, particleToUpdate, coordinateToUpdate)
    ratioRBM = rbmComputeRatio(system, oldPosition)
    # println(ratioSlaterDeterminant,"   ", ratioSlaterGaussian,"    ",  ratioRBM)
    return ratioSlaterDeterminant*ratioSlaterGaussian*ratioRBM, R
end

################################################################################

########  #                #        #########  #########  ########
#         #               # #           #      #          #      #
#         #              #   #          #      #          #      #
########  #             #######         #      #########  #######
       #  #            #       #        #      #          #    #
       #  #           #         #       #      #          #     #
########  #########  #           #      #      #########  #      #

################################################################################
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

        local_energy = computeLocalEnergy(system, system.interacting)
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

###    ##    ###    ##    
####   ##    ####   ##
## ##  ##    ## ##  ##
##  ## ##    ##  ## ##
##   ####    ##   ####
##    ###    ##    ###

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

        local_energy = computeLocalEnergy(system, system.interacting)
        local_energies[i] = local_energy

        psi_param_derivative = nnComputeParameterGradient(system)
        psi_weights_derivative = params(system.nn.model).*0

        # slater_psi = slaterGaussianWaveFunction(system)*slaterWaveFunction(system)

        x = reshape(system.particles', 1,:)'

        slater_psi = 1.0#/system.nn.model(x)[1]

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

"""
    runMetorpolisBruteForce(nqs::NQS, num_mc_iterations::Int64, burn_in::Float64, step_length::Float64, write_to_file::Bool)

Uses the Metropolis Brute force algorithm to produce num_mc_iterations samples from the distribution and writes the samples to file if wanted.
Only the samples after the burn-in are used to calculate the local energy and gradient estimate that is returned.
Calculates the estimate for the local energy as well as the gradients.
"""
function runMetropolisBruteForce(system::slaterRBM, num_mc_iterations::Int64, step_length::Float64; burn_in = 0.01)

    nqs = system.nqs

    x = reshape(system.particles', 1,:)'

    local_energy_sum::Float64 = 0.0

    #Initializes the arrays and matrices to save the derivatives and the sums.
    local_energy_psi_derivative_a_sum::Array{Float64, 2} = zeros(Float64, size(nqs.a))
    local_energy_psi_derivative_b_sum::Array{Float64, 2} = zeros(Float64, size(nqs.b))
    local_energy_psi_derivative_w_sum::Array{Float64, 2} = zeros(Float64, size(nqs.w))

    psi_derivative_a_sum::Array{Float64, 2} = zeros(Float64, size(nqs.a))
    psi_derivative_b_sum::Array{Float64, 2} = zeros(Float64, size(nqs.b))
    psi_derivative_w_sum::Array{Float64, 2} = zeros(Float64, size(nqs.w))

    psi_derivative_a::Array{Float64, 2} = zeros(Float64, size(nqs.a))
    psi_derivative_b::Array{Float64, 2} = zeros(Float64, size(nqs.b))
    psi_derivative_w::Array{Float64, 2} = zeros(Float64, size(nqs.w))

    # precalc::Array{Float64, 2} = nqs.b + transpose((1.0/nqs.sigma_squared)*(transpose(x)* nqs.w))

    #Vector to store the energies for each step.
    local_energies::Array{Float64, 1} = zeros(Float64, Int(num_mc_iterations))

    start = time()

    for i = 1:num_mc_iterations

        # Does one step with the brute force method.
        metropolisStepBruteForce(step_length, system)

        x = reshape(system.particles', 1,:)'

        precalc = nqs.b + transpose((1.0/nqs.sigma_squared)*(transpose(x)* nqs.w))

        # Computes the contribution to Monte carlo estimate of the local energy given the new system configuration.
        local_energy = computeLocalEnergy(system, system.nqs.interacting)

        local_energies[i] = local_energy

        # Computes the contribution to the gradients given the new system configuration.
        rbmComputeParameterGradient!(system, psi_derivative_a, psi_derivative_b, psi_derivative_w, precalc)

        # Calculates the estimates of the energy and derivatives. Uses only those after the burn-in period.
        if i > burn_in*num_mc_iterations
            local_energy_sum += local_energy

            local_energy_psi_derivative_a_sum += local_energy*psi_derivative_a
            local_energy_psi_derivative_b_sum += local_energy*psi_derivative_b
            local_energy_psi_derivative_w_sum += local_energy*psi_derivative_w

            psi_derivative_a_sum += psi_derivative_a
            psi_derivative_b_sum += psi_derivative_b
            psi_derivative_w_sum += psi_derivative_w
        end
    end

    # Updates the final estimates of local energy and gradients.
    samples = num_mc_iterations - burn_in*num_mc_iterations

    mc_local_energy = local_energy_sum/samples

    mc_local_energy_psi_derivative_a = local_energy_psi_derivative_a_sum/samples
    mc_local_energy_psi_derivative_b = local_energy_psi_derivative_b_sum/samples
    mc_local_energy_psi_derivative_w = local_energy_psi_derivative_w_sum/samples

    mc_psi_derivative_a = psi_derivative_a_sum/samples
    mc_psi_derivative_b = psi_derivative_b_sum/samples
    mc_psi_derivative_w = psi_derivative_w_sum/samples

    local_energy_derivative_a = 2*(mc_local_energy_psi_derivative_a - mc_local_energy*mc_psi_derivative_a)
    local_energy_derivative_b = 2*(mc_local_energy_psi_derivative_b - mc_local_energy*mc_psi_derivative_b)
    local_energy_derivative_w = 2*(mc_local_energy_psi_derivative_w - mc_local_energy*mc_psi_derivative_w)

    return mc_local_energy, local_energy_derivative_a, local_energy_derivative_b, local_energy_derivative_w
end

function optimizationStep(system::slaterRBM, grad_a::Array{Float64, 2}, grad_b::Array{Float64, 2}, grad_w::Array{Float64, 2}, learning_rate::Float64)
    nqs = system.nqs
    nqs.a[:] = nqs.a - learning_rate*grad_a
    nqs.b[:] = nqs.b - learning_rate*grad_b
    nqs.w[:,:] = nqs.w - learning_rate*grad_w
end

function runVMC(system::slaterRBM, numVMCIterations::Int64, numMonteCarloIterations, mc_step_length, learning_rate)
    local_energies::Array{Float64, 2} = zeros(Float64, (numVMCIterations, 1))
    for k = 1:numVMCIterations
        local_energy,_grad_a,  _grad_b, _grad_w = runMetropolisBruteForce(system, numMonteCarloIterations, mc_step_length)
        optimizationStep(system, _grad_a, _grad_b, _grad_w, learning_rate)
        local_energies[k] = local_energy
        println("Iteration = ", k, "    E = ", local_energy)
    end
    return local_energies
end 

################################################################################

        ##
        ##
        ##
        ##
        ##
        ##
#      ##
#     ##
 #####

################################################################################

function runMetropolisBruteForce(system::slaterJastrow, num_mc_iterations, step_length; burn_in = 0.01)
    local_energy_sum::Float64 = 0.0

    local_energy_psi_parameter_derivative_sum = zeros(size(system.jastrowFactor.kappa))

    # display(local_energy_psi_parameter_derivative_sum)

    psi_parameter_derivative_sum = zeros(size(system.jastrowFactor.kappa))

    local_energies::Array{Float64, 1} = zeros(Float64, Int(num_mc_iterations))

    start = time()

    for i = 1:num_mc_iterations
        metropolisStepBruteForce(step_length, system)

        jastrowUpdateDistanceMatrix(system)

        local_energy = computeLocalEnergy(system, system.interacting)
        local_energies[i] = local_energy

        psi_parameter_derivative = jastrowComputeParameterGradient(system)

        if i > burn_in*num_mc_iterations
            local_energy_sum += local_energy
            local_energy_psi_parameter_derivative_sum += local_energy*psi_parameter_derivative
            psi_parameter_derivative_sum += psi_parameter_derivative
        end
    end

    runtime = time() - start

    samples = num_mc_iterations - burn_in*num_mc_iterations

    mc_local_energy = local_energy_sum/samples
    mc_local_energy_psi_derivative_parameters = local_energy_psi_parameter_derivative_sum/samples
    mc_psi_derivative_parameters = psi_parameter_derivative_sum/samples

    local_energy_derivative_parameters = 2*(mc_local_energy_psi_derivative_parameters - mc_local_energy*mc_psi_derivative_parameters)

    println(mc_local_energy)

    println(mc_local_energy, local_energy_derivative_parameters)

    return mc_local_energy, local_energy_derivative_parameters
end

function optimizationStep(system::slaterJastrow, gradKappa, learning_rate::Float64)
    system.jastrowFactor.kappa[:,:] = system.jastrowFactor.kappa[:,:] - learning_rate*gradKappa
end

function runVMC(system::slaterJastrow, numVMCIterations::Int64, numMonteCarloIterations, mc_step_length, learning_rate)
    local_energies::Array{Float64, 2} = zeros(Float64, (numVMCIterations, 1))
    for k = 1:numVMCIterations
        display(system.jastrowFactor.kappa)
        local_energy, _grad_kappa = runMetropolisBruteForce(system, numMonteCarloIterations, mc_step_length)
        optimizationStep(system, _grad_kappa, learning_rate)
        local_energies[k] = local_energy
        println("Iteration = ", k, "    E = ", local_energy)
    end
    return local_energies
end 

end #MODULE