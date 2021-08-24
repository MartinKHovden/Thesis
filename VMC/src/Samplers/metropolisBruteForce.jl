module metropolisBruteForce

export metropolisStepBruteForce, runVMC, runMetropolisBruteForce, runMetropolis

# include("../Wavefunctions/slaterDeterminant.jl")

using ..initializeSystem
using ..slaterDeterminant
using ..jastrow
using ..simpleJastrow
using ..neuralNetwork
using ..neuralNetworkAnalytical
using ..harmonicOscillator
using ..boltzmannMachine
using Flux:params, ADAM, Descent, update!, Params

function saveDataToFile(data, filename::String)
    open(filename, "w") do file
        for d in data
            println(file, d)
        end
    end
end

function generateFileNameMonteCarlo(system, sampler)
    return 0
end

function generateFileNameVMC(system, optimizer)
    return 0
end

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

function metropolisStepBruteForce(stepLength, system::gaussianJastrow)
    numParticles = system.numParticles 
    numDimensions = system.numDimensions

    # Chooses one coordinate randomly to update.
    coordinateToUpdate::Int64 = rand(1:numDimensions)
    particleToUpdate::Int64 = rand(1:numParticles)

    # Update the coordinate:
    oldPosition = copy(system.particles)
    system.particles[particleToUpdate, coordinateToUpdate] += (rand(Float64) - 0.5)*stepLength

    # Update the slater matrix:

    U = rand(Float64)

    ratio = computeRatio(system, particleToUpdate, coordinateToUpdate, oldPosition)

    if U < ratio
    else 
        system.particles[particleToUpdate, coordinateToUpdate] = oldPosition[particleToUpdate, coordinateToUpdate]
    end
end

function metropolisStepBruteForce(stepLength, system::gaussianNNAnalytical)
    numParticles = system.numParticles 
    numDimensions = system.numDimensions

    # Chooses one coordinate randomly to update.
    coordinateToUpdate::Int64 = rand(1:numDimensions)
    particleToUpdate::Int64 = rand(1:numParticles)

    # Update the coordinate:
    oldPosition = copy(system.particles)
    system.particles[particleToUpdate, coordinateToUpdate] += (rand(Float64) - 0.5)*stepLength

    # Update the slater matrix:

    U = rand(Float64)

    ratio = computeRatio(system, particleToUpdate, coordinateToUpdate, oldPosition)

    if U < ratio
    else 
        system.particles[particleToUpdate, coordinateToUpdate] = oldPosition[particleToUpdate, coordinateToUpdate]
    end
end
""" 
    metropolisStepImportanceSampling(stepLength, system)

Function for updating the position doing one step with the Importance Sampling Metropolis algorithm. 
"""
function metropolisStepImportanceSampling(stepLength, system)
    numParticles = system.numParticles 
    numDimensions = system.numDimensions

    coordinateToUpdate::Int64 = rand(1:numDimensions)
    particleToUpdate::Int64 = rand(1:numParticles)

    oldPosition = copy(system.particles)

    D = 0.5

    currentDriftForce = computeDriftForce(system, particleToUpdate, coordinateToUpdate)

    # println(currentDriftForce)

    system.particles[particleToUpdate, coordinateToUpdate] += D*currentDriftForce*stepLength + randn(Float64)*sqrt(stepLength)

    newDriftForce = computeDriftForce(system, particleToUpdate, coordinateToUpdate)

    # Update the slater matrix:
    slaterMatrixUpdate(system, particleToUpdate)

    U = rand(Float64)

    greens_function = computeGreensFunction(oldPosition, system.particles, particleToUpdate,
                                            coordinateToUpdate, currentDriftForce, 
                                            newDriftForce, D, stepLength)

    ratio, R = computeRatio(system, particleToUpdate, coordinateToUpdate, oldPosition)

    if U < greens_function*ratio
        inverseSlaterMatrixUpdate(system, particleToUpdate, (R))
    else 
        system.particles[particleToUpdate, coordinateToUpdate] = oldPosition[particleToUpdate, coordinateToUpdate]
        slaterMatrixUpdate(system, particleToUpdate)
    end
end

"""
    computeDriftForce(system::slater, particleNumber::Int64, dimension::Int64)

Computes the driftforce used in importance sampling when chaniging the 
position for the particle. 
"""
function computeDriftForce(system::slater, particleNumber::Int64, dimension::Int64)
    return slaterDeterminantComputeDriftForce(system, particleNumber, dimension) + 
            + slaterGaussianComputeDriftForce(system, particleNumber, dimension)
end

"""
    computeDriftForce(system::slater, particleNumber::Int64, dimension::Int64)

Computes the driftforce used in importance sampling when chaniging the 
position for the particle. 
"""
function computeDriftForce(system::slaterJastrow, particleNumber::Int64, dimension::Int64)
    return slaterDeterminantComputeDriftForce(system, particleNumber, dimension) +  
            + slaterGaussianComputeDriftForce(system, particleNumber, dimension) +
            + jastrowComputeDriftForce(system, particleNumber, dimension)
end

"""
    computeDriftForce(system::slater, particleNumber::Int64, dimension::Int64)

Computes the driftforce used in importance sampling when chaniging the 
position for the particle. 
"""
function computeDriftForce(system::slaterRBM, particleNumber::Int64, dimension::Int64)
    return slaterDeterminantComputeDriftForce(system, particleNumber, dimension) + 
            + rbmComputeDriftForce(system, particleNumber, dimension)
end

"""
    computeDriftForce(system::slater, particleNumber::Int64, dimension::Int64)

Computes the driftforce used in importance sampling when chaniging the 
position for the particle. 
"""
function computeDriftForce(system::slaterNN, particleNumber::Int64, dimension::Int64)
    return slaterDeterminantComputeDriftForce(system, particleNumber, dimension) +
            + slaterGaussianComputeDriftForce(system, particleNumber, dimension) +
            + nnComputeDriftForce(system, particleNumber, dimension)
end 

function computeDriftForce(system::slaterNNAnalytical, particleNumber::Int64, dimension::Int64)
    return slaterDeterminantComputeDriftForce(system, particleNumber, dimension) +
            + slaterGaussianComputeDriftForce(system, particleNumber, dimension) +
            + nnAnalyticalComputeDriftForce(system, particleNumber, dimension)
end

""" 
    computeGreensFunction(oldPosition, newPosition, 
                                        particleToUpdate,        
                                        coordinateToUpdate, 
                                        oldDriftForce, 
                                        newDriftForce, 
                                        D,
                                        stepLength)

Function for computing the greens function used in importance sampling. 
"""
function computeGreensFunction(oldPosition, newPosition, 
                                            particleToUpdate,        
                                            coordinateToUpdate, 
                                            oldDriftForce, 
                                            newDriftForce, 
                                            D,
                                            stepLength)

    greens_function_argument = (oldPosition[particleToUpdate, coordinateToUpdate] +
                                - newPosition[particleToUpdate, coordinateToUpdate] +
                                - D*stepLength*newDriftForce)^2 +
                                - (newPosition[particleToUpdate, coordinateToUpdate] + 
                                - oldPosition[particleToUpdate, coordinateToUpdate] +
                                - D*stepLength*oldDriftForce)^2

    greens_function_argument /= (4.0*D*stepLength)
    greens_function = exp(-greens_function_argument)
    return greens_function
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

function computeRatio(system::gaussianJastrow, particleToUpdate, coordinateToUpdate, oldPosition)
    ratioSlaterGaussian = slaterGaussianComputeRatio(system, oldPosition, particleToUpdate, coordinateToUpdate)
    ratioJastrow = simpleJastrowComputeRatio(system, oldPosition, particleToUpdate)
    # println(ratioSlaterGaussian, ratioJastrow)
    return ratioSlaterGaussian*ratioJastrow
end

function computeRatio(system::gaussianNNAnalytical, particleToUpdate, coordinateToUpdate, oldPosition)
    ratioSlaterGaussian = slaterGaussianComputeRatio(system, oldPosition, particleToUpdate, coordinateToUpdate)
    ratioNNAnalytical = nnAnalyticalComputeRatio!(system, oldPosition)
    # println(ratioSlaterGaussian, ratioNNAnalytical)
    return ratioSlaterGaussian*ratioNNAnalytical
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
    computeRatio(system::slaterNNAnalytical, particleToUpdate, coordinateToUpdate, oldPosition)

Returns the wavefunction ratio for the system with the Slater-NeuralNetwork wavefunction,
given the particle moved, the dimension, and the old position. 
"""
function computeRatio(system::slaterNNAnalytical, particleToUpdate, coordinateToUpdate, oldPosition)
    R = slaterMatrixComputeRatio(system, particleToUpdate)
    ratioSlaterDeterminant = R^2
    ratioSlaterGaussian = slaterGaussianComputeRatio(system, oldPosition, particleToUpdate, coordinateToUpdate)
    ratioNNAnalytical = nnAnalyticalComputeRatio!(system, oldPosition)
    return ratioSlaterDeterminant*ratioSlaterGaussian*ratioNNAnalytical, R
end

""" 
    computeRatio(system::slaterRBM, particleToUpdate, coordinateToUpdate, oldPosition)

Returns the wavefunction ratio for the system with the Slater-RBM wavefunction,
given the particle moved, the dimension, and the old position. 
"""
function computeRatio(system::slaterRBM, particleToUpdate, coordinateToUpdate, oldPosition)
    R = slaterMatrixComputeRatio(system, particleToUpdate)
    ratioSlaterDeterminant = R^2
    ratioRBM = rbmComputeRatio(system, oldPosition)
    return ratioSlaterDeterminant*ratioRBM, R 
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
function runMetropolisBruteForce(system::slater, num_mc_iterations, step_length; burn_in = 0.01, writeToFile=false)
    local_energy_sum::Float64 = 0.0

    local_energy_psi_parameter_derivative_sum = 0
    psi_parameter_derivative_sum = 0

    local_energies::Array{Float64, 1} = zeros(Float64, Int(num_mc_iterations))

    start = time()

    for i = 1:num_mc_iterations
        metropolisStepBruteForce(step_length, system)
        # metropolisStepImportanceSampling(step_length, system)

        local_energy = computeLocalEnergy(system, system.interacting)
        local_energies[i] = local_energy

        psi_parameter_derivative = slaterGaussianComputeParameterGradient(system)

        if i > burn_in*num_mc_iterations
            local_energy_sum += local_energy
            local_energy_psi_parameter_derivative_sum += local_energy*psi_parameter_derivative
            psi_parameter_derivative_sum += psi_parameter_derivative
        end
    end

    if writeToFile
        filename = "../Data/slater_bf_step_length_" * string(step_length) * ".txt" 
        open(filename, "w") do file
            for e in local_energies
                println(file, e)
            end
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

function optimizationStep(system::slater, optimizer, grad)
    update!(optimizer, system.alpha, grad)
end

"""
    runVMC(system::slater, numVMCIterations::Int64, numMonteCarloIterations, mc_step_length, learning_rate)

Runs the full vmc calculation by calling the runMetropolisBruteForce step multiple 
times and updating the variational parameters accordingly. 
"""
function runVMC(system::slater, numVMCIterations::Int64, numMonteCarloIterations, 
                                                        mc_step_length, 
                                                        learning_rate)

    local_energies::Array{Float64, 2} = zeros(Float64, (numVMCIterations, 1))
    for k = 1:numVMCIterations
        local_energy, grad = runMetropolisBruteForce(system, numMonteCarloIterations, mc_step_length)
        optimizationStep(system, grad, learning_rate)
        local_energies[k] = local_energy
        println("Iteration = ", k, "    E = ", local_energy, "   alpha =  ", system.alpha)
    end
    return local_energies
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
        # metropolisStepImportanceSampling(step_length, system)

        local_energy = computeLocalEnergy(system, system.interacting)
        local_energies[i] = local_energy

        psi_param_derivative = nnComputeParameterGradient(system)
        psi_weights_derivative = params(system.nn.model).*0

        x = reshape(system.particles', 1,:)'

        for grad=1:numGradients
            psi_weights_derivative[grad] = psi_param_derivative[params(system.nn.model)[grad]]
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

    return mc_local_energy, local_energy_derivative_weights
end

function runVMC(system::slaterNN, numVMCIterations::Int64, numMonteCarloIterations, mc_step_length, learning_rate; writeToFile=false)
    local_energies::Array{Float64, 2} = zeros(Float64, (numVMCIterations, 1))
    for k = 1:numVMCIterations
        local_energy, _grad_w = runMetropolisBruteForce(system, numMonteCarloIterations, mc_step_length)
        optimizationStep(system, _grad_w, learning_rate)
        local_energies[k] = local_energy
        println("Iteration = ", k, "    E = ", local_energy, "   Error =  ", abs(local_energy - 2.0))
    end

    if writeToFile
        filename = "vmc_nn_bf_step_length_" * string(mc_step_length) * ".txt"  
        saveDataToFile(local_energies, filename)
    end

    return local_energies
end 

function optimizationStep(system::slaterNN, grad_W, learning_rate::Float64)
    ps = params(system.nn.model)
    for (p,grad) in zip(ps, grad_W) 
        p -= learning_rate*grad
    end
end

function optimizationStep(system::slaterNN, optimizer, grad)
    return 0
end

################################################################################
################################################################################
################################################################################

function runMetropolisBruteForce(system::slaterNNAnalytical, num_mc_iterations, step_length; burn_in = 0.01)
    local_energy_sum::Float64 = 0.0

    local_energy_psi_w1_derivative_sum = zeros(size(system.nn.w1))
    local_energy_psi_w2_derivative_sum = zeros(size(system.nn.w2))
    local_energy_psi_w3_derivative_sum = zeros(size(system.nn.w3))
    local_energy_psi_b1_derivative_sum = zeros(size(system.nn.b1))
    local_energy_psi_b2_derivative_sum = zeros(size(system.nn.b2))
    local_energy_psi_b3_derivative_sum = zeros(size(system.nn.b3))

    psi_w1_derivative_sum = zeros(size(system.nn.w1))
    psi_w2_derivative_sum = zeros(size(system.nn.w2))
    psi_w3_derivative_sum = zeros(size(system.nn.w3))
    psi_b1_derivative_sum = zeros(size(system.nn.b1))
    psi_b2_derivative_sum = zeros(size(system.nn.b2))
    psi_b3_derivative_sum = zeros(size(system.nn.b3))

    local_energies::Array{Float64, 1} = zeros(Float64, Int(num_mc_iterations))

    start = time()

    for i = 1:num_mc_iterations
        metropolisStepBruteForce(step_length, system)
        # metropolisStepImportanceSampling(step_length, system)

        nnAnalyticalComputePsi!(system, system.particles)

        local_energy = computeLocalEnergy(system, system.interacting)
        local_energies[i] = local_energy

        psi_param_derivative = nnAnalyticalComputeParameterGradient!(system)

        if i > burn_in*num_mc_iterations
            local_energy_sum += local_energy

            local_energy_psi_w1_derivative_sum += local_energy*psi_param_derivative[1]
            local_energy_psi_w2_derivative_sum += local_energy*psi_param_derivative[2]
            local_energy_psi_w3_derivative_sum += local_energy*psi_param_derivative[3]
            local_energy_psi_b1_derivative_sum += local_energy*psi_param_derivative[4]
            local_energy_psi_b2_derivative_sum += local_energy*psi_param_derivative[5]
            local_energy_psi_b3_derivative_sum += local_energy*psi_param_derivative[6]

            psi_w1_derivative_sum += psi_param_derivative[1]
            psi_w2_derivative_sum += psi_param_derivative[2]
            psi_w3_derivative_sum += psi_param_derivative[3]
            psi_b1_derivative_sum += psi_param_derivative[4]
            psi_b2_derivative_sum += psi_param_derivative[5]
            psi_b3_derivative_sum += psi_param_derivative[6]
        end
    end

    runtime = time() - start

    samples = num_mc_iterations - burn_in*num_mc_iterations

    mc_local_energy = local_energy_sum/samples

    mc_local_energy_psi_derivative_w1 = local_energy_psi_w1_derivative_sum/samples
    mc_local_energy_psi_derivative_w2 = local_energy_psi_w2_derivative_sum/samples
    mc_local_energy_psi_derivative_w3 = local_energy_psi_w3_derivative_sum/samples
    mc_local_energy_psi_derivative_b1 = local_energy_psi_b1_derivative_sum/samples
    mc_local_energy_psi_derivative_b2 = local_energy_psi_b2_derivative_sum/samples
    mc_local_energy_psi_derivative_b3 = local_energy_psi_b3_derivative_sum/samples

    mc_psi_derivative_w1 = psi_w1_derivative_sum/samples
    mc_psi_derivative_w2 = psi_w2_derivative_sum/samples
    mc_psi_derivative_w3 = psi_w3_derivative_sum/samples
    mc_psi_derivative_b1 = psi_b1_derivative_sum/samples
    mc_psi_derivative_b2 = psi_b2_derivative_sum/samples
    mc_psi_derivative_b3 = psi_b3_derivative_sum/samples


    local_energy_derivative_w1 = 2*(mc_local_energy_psi_derivative_w1 - mc_local_energy*mc_psi_derivative_w1)
    local_energy_derivative_w2 = 2*(mc_local_energy_psi_derivative_w2 - mc_local_energy*mc_psi_derivative_w2)
    local_energy_derivative_w3 = 2*(mc_local_energy_psi_derivative_w3 - mc_local_energy*mc_psi_derivative_w3)
    local_energy_derivative_b1 = 2*(mc_local_energy_psi_derivative_b1 - mc_local_energy*mc_psi_derivative_b1)
    local_energy_derivative_b2 = 2*(mc_local_energy_psi_derivative_b2 - mc_local_energy*mc_psi_derivative_b2)
    local_energy_derivative_b3 = 2*(mc_local_energy_psi_derivative_b3 - mc_local_energy*mc_psi_derivative_b3)

    return mc_local_energy, [Array(local_energy_derivative_w1), Array(local_energy_derivative_w2), Array(local_energy_derivative_w3), Array(local_energy_derivative_b1), Array(local_energy_derivative_b2), Array(local_energy_derivative_b3)]
end

# function runVMC(system::slaterNNAnalytical, numVMCIterations::Int64, numMonteCarloIterations, mc_step_length, learning_rate; writeToFile=false)
#     local_energies::Array{Float64, 2} = zeros(Float64, (numVMCIterations, 1))
#     for k = 1:numVMCIterations
#         local_energy, grads = runMetropolisBruteForce(system, numMonteCarloIterations, mc_step_length)
#         # println(grads)
#         optimizationStep(system, grads, learning_rate)
#         local_energies[k] = local_energy
#         println("Iteration = ", k, "    E = ", local_energy, "   Error =  ", abs(local_energy - 2.0))
#     end

#     return local_energies
# end 

function runVMC(system, numVMCIterations::Int64, numMonteCarloIterations, mc_step_length, optimizer; writeToFile=false)
    local_energies::Array{Float64, 2} = zeros(Float64, (numVMCIterations, 1))
    for k = 1:numVMCIterations
        local_energy, grads = runMetropolisBruteForce(system, numMonteCarloIterations, mc_step_length)
        # println(grads)
        optimizationStep(system, optimizer, grads)
        local_energies[k] = local_energy
        println("Iteration = ", k, "    E = ", local_energy, "   Error =  ", abs(local_energy - 2.0))
    end

    return local_energies
end

# function optimizationStep(system::slaterNNAnalytical, grads, learning_rate::Float64)
#     model = system.nn
#     model.w1[:,:] -= grads[1]*learning_rate
#     model.w2[:,:] -= grads[2]*learning_rate
#     model.w3[:,:] -= grads[3]*learning_rate

#     model.b1[:] -= grads[4]*learning_rate
#     model.b2[:] -= grads[5]*learning_rate
#     model.b3[:] -= grads[6]*learning_rate
# end

function optimizationStep(system::slaterNNAnalytical, optimizer, grads)
    model = system.nn

    update!(optimizer, model.w1, grads[1])
    update!(optimizer, model.w2, grads[2])
    update!(optimizer, model.w3, grads[3])

    update!(optimizer, model.b1, grads[4])
    update!(optimizer, model.b2, grads[5])
    update!(optimizer, model.b3, grads[6])
end 

function runMetropolisBruteForce(system::gaussianNNAnalytical, num_mc_iterations, step_length, iteration; burn_in = 0.01)
    local_energy_sum::Float64 = 0.0

    local_energy_psi_w1_derivative_sum = zeros(size(system.nn.w1))
    local_energy_psi_w2_derivative_sum = zeros(size(system.nn.w2))
    local_energy_psi_w3_derivative_sum = zeros(size(system.nn.w3))
    local_energy_psi_b1_derivative_sum = zeros(size(system.nn.b1))
    local_energy_psi_b2_derivative_sum = zeros(size(system.nn.b2))
    local_energy_psi_b3_derivative_sum = zeros(size(system.nn.b3))

    psi_w1_derivative_sum = zeros(size(system.nn.w1))
    psi_w2_derivative_sum = zeros(size(system.nn.w2))
    psi_w3_derivative_sum = zeros(size(system.nn.w3))
    psi_b1_derivative_sum = zeros(size(system.nn.b1))
    psi_b2_derivative_sum = zeros(size(system.nn.b2))
    psi_b3_derivative_sum = zeros(size(system.nn.b3))

    local_energies::Array{Float64, 1} = zeros(Float64, Int(num_mc_iterations))

    start = time()

    for i = 1:num_mc_iterations
        metropolisStepBruteForce(step_length, system)
        # metropolisStepImportanceSampling(step_length, system)

        nnAnalyticalComputePsi!(system, system.particles)

        local_energy = computeLocalEnergy(system, iteration, system.interacting)
        local_energies[i] = local_energy

        psi_param_derivative = nnAnalyticalComputeParameterGradient!(system)

        if i > burn_in*num_mc_iterations
            local_energy_sum += local_energy

            local_energy_psi_w1_derivative_sum += local_energy*psi_param_derivative[1]
            local_energy_psi_w2_derivative_sum += local_energy*psi_param_derivative[2]
            local_energy_psi_w3_derivative_sum += local_energy*psi_param_derivative[3]
            local_energy_psi_b1_derivative_sum += local_energy*psi_param_derivative[4]
            local_energy_psi_b2_derivative_sum += local_energy*psi_param_derivative[5]
            local_energy_psi_b3_derivative_sum += local_energy*psi_param_derivative[6]

            psi_w1_derivative_sum += psi_param_derivative[1]
            psi_w2_derivative_sum += psi_param_derivative[2]
            psi_w3_derivative_sum += psi_param_derivative[3]
            psi_b1_derivative_sum += psi_param_derivative[4]
            psi_b2_derivative_sum += psi_param_derivative[5]
            psi_b3_derivative_sum += psi_param_derivative[6]
        end
    end

    runtime = time() - start

    samples = num_mc_iterations - burn_in*num_mc_iterations

    mc_local_energy = local_energy_sum/samples

    mc_local_energy_psi_derivative_w1 = local_energy_psi_w1_derivative_sum/samples
    mc_local_energy_psi_derivative_w2 = local_energy_psi_w2_derivative_sum/samples
    mc_local_energy_psi_derivative_w3 = local_energy_psi_w3_derivative_sum/samples
    mc_local_energy_psi_derivative_b1 = local_energy_psi_b1_derivative_sum/samples
    mc_local_energy_psi_derivative_b2 = local_energy_psi_b2_derivative_sum/samples
    mc_local_energy_psi_derivative_b3 = local_energy_psi_b3_derivative_sum/samples

    mc_psi_derivative_w1 = psi_w1_derivative_sum/samples
    mc_psi_derivative_w2 = psi_w2_derivative_sum/samples
    mc_psi_derivative_w3 = psi_w3_derivative_sum/samples
    mc_psi_derivative_b1 = psi_b1_derivative_sum/samples
    mc_psi_derivative_b2 = psi_b2_derivative_sum/samples
    mc_psi_derivative_b3 = psi_b3_derivative_sum/samples


    local_energy_derivative_w1 = 2*(mc_local_energy_psi_derivative_w1 - mc_local_energy*mc_psi_derivative_w1)
    local_energy_derivative_w2 = 2*(mc_local_energy_psi_derivative_w2 - mc_local_energy*mc_psi_derivative_w2)
    local_energy_derivative_w3 = 2*(mc_local_energy_psi_derivative_w3 - mc_local_energy*mc_psi_derivative_w3)
    local_energy_derivative_b1 = 2*(mc_local_energy_psi_derivative_b1 - mc_local_energy*mc_psi_derivative_b1)
    local_energy_derivative_b2 = 2*(mc_local_energy_psi_derivative_b2 - mc_local_energy*mc_psi_derivative_b2)
    local_energy_derivative_b3 = 2*(mc_local_energy_psi_derivative_b3 - mc_local_energy*mc_psi_derivative_b3)

    return mc_local_energy, [Array(local_energy_derivative_w1), Array(local_energy_derivative_w2), Array(local_energy_derivative_w3), Array(local_energy_derivative_b1), Array(local_energy_derivative_b2), Array(local_energy_derivative_b3)]
end


function optimizationStep(system::gaussianNNAnalytical, optimizer, grads)
    model = system.nn

    # println(grads)

    update!(optimizer, model.w1, grads[1])
    update!(optimizer, model.w2, grads[2])
    update!(optimizer, model.w3, grads[3])

    update!(optimizer, model.b1, grads[4])
    update!(optimizer, model.b2, grads[5])
    update!(optimizer, model.b3, grads[6])
end 

function runVMC(system::gaussianNNAnalytical, numVMCIterations::Int64, numMonteCarloIterations, mc_step_length, optimizer)
    println("test simple Jas gaussian")
    local_energies::Array{Float64, 2} = zeros(Float64, (numVMCIterations, 1))
    for k = 1:numVMCIterations
        local_energy, grads = runMetropolisBruteForce(system, numMonteCarloIterations, mc_step_length, k)
        optimizationStep(system, optimizer, grads)
        local_energies[k] = local_energy
        println("Iteration = ", k, "    E = ", local_energy, "E relative = ", (local_energy - 2.0)/2.0)
    end
    return local_energies
end


################################################################################
################################################################################
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
    runMetorpolisBruteForce(system:slaterRBM, num_mc_iterations::Int64, burn_in::Float64, step_length::Float64, write_to_file::Bool)

Uses the Metropolis Brute force algorithm to produce num_mc_iterations samples from the distribution and writes the samples to file if wanted.
Only the samples after the burn-in are used to calculate the local energy and gradient estimate that is returned.
Calculates the estimate for the local energy as well as the gradients.
"""
function runMetropolis(system::slaterRBM, num_mc_iterations::Int64, step_length::Float64; sampler = "bf", burn_in = 0.01, writeToFile = false)

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

    if sampler == "bf"
        step = metropolisStepBruteForce 
    elseif  sampler == "is"
        step = metropolisStepImportanceSampling
    else
        println("This sampler is not implemented")
    end

    for i = 1:num_mc_iterations

        # Does one step with the brute force method.
        # metropolisStepBruteForce(step_length, system)
        # metropolisStepImportanceSampling(step_length, system)
        step(step_length, system)

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

    if writeToFile
        if system.nqs.interacting == true
            folder = "Interacting"
        elseif system.nqs.interacting == false
            folder = "Non-interacting"
        end
        filename = "../Data/"*folder*"/metropolis/rbm_"*sampler*"_step_length_" * string(step_length) *"_num_hidden_"* string(length(system.nqs.h)) * ".txt"  
        saveDataToFile(local_energies, filename)
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

    return mc_local_energy,  [local_energy_derivative_a, local_energy_derivative_b, local_energy_derivative_w]

end

function optimizationStep(system::slaterRBM, grad, learning_rate::Float64)
    nqs = system.nqs
    nqs.a[:] = nqs.a - learning_rate*grad[1][:]
    nqs.b[:] = nqs.b - learning_rate*grad[2][:]
    nqs.w[:,:] = nqs.w - learning_rate*grad[3][:,:]
end

function optimizationStep(system::slaterRBM, optimizer, grad)
    nqs = system.nqs 
    update!(optimizer, nqs.a, grad[1])
    update!(optimizer, nqs.b, grad[2])
    update!(optimizer, nqs.w, grad[3])
end
# function runVMC(system::slaterRBM, numVMCIterations::Int64, numMonteCarloIterations, mc_step_length, learning_rate; sampler = "bf", optimization = "gd", writeToFile = false)
#     local_energies::Array{Float64, 2} = zeros(Float64, (numVMCIterations, 1))
#     for k = 1:numVMCIterations
#         local_energy, grad = runMetropolis(system, numMonteCarloIterations, mc_step_length, sampler = sampler)
#         optimizationStep(system, grad, learning_rate)
#         local_energies[k] = local_energy
#         println("Iteration = ", k, "    E = ", local_energy, "  Error = ", abs(local_energy - 2.0))
#     end

#     if writeToFile
#         if system.nqs.interacting == true
#             folder = "Interacting"
#         elseif system.nqs.interacting == false
#             folder = "Non-interacting"
#         end
#         filename = "../Data/"*folder*"/VMC/vmc_rbm_bf_step_length_" * string(mc_step_length) *"_num_hidden_"* string(length(system.nqs.h)) * ".txt"  
#         open(filename, "w") do file
#             for e in local_energies
#                 println(file, e)
#             end
#         end
#     end

#     return local_energies
# end 

function runVMC(system::slaterRBM, numVMCIterations::Int64, numMonteCarloIterations, mc_step_length, optimizer; sampler = "bf", writeToFile = false)
    local_energies::Array{Float64, 2} = zeros(Float64, (numVMCIterations, 1))
    opt = optimizer
    for k = 1:numVMCIterations
        local_energy, grad = runMetropolis(system, numMonteCarloIterations, mc_step_length, sampler = sampler)
        optimizationStep(system, opt, grad)
        local_energies[k] = local_energy
        println("Iteration = ", k, "    E = ", local_energy, "  Error = ", abs(local_energy - 2.0))
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

function runMetropolisBruteForce(system::slaterJastrow, num_mc_iterations, step_length; burn_in = 0.01, writeToFile= false)
    local_energy_sum::Float64 = 0.0

    local_energy_psi_parameter_derivative_sum = zeros(size(system.jastrowFactor.kappa))

    psi_parameter_derivative_sum = zeros(size(system.jastrowFactor.kappa))

    local_energies::Array{Float64, 1} = zeros(Float64, Int(num_mc_iterations))

    start = time()

    for i = 1:num_mc_iterations
        metropolisStepBruteForce(step_length, system)
        # metropolisStepImportanceSampling(step_length, system)

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

    if writeToFile
        open("output.txt", "w") do file
            for e in local_energies
                println(file, e)
            end
        end
    end

    runtime = time() - start

    samples = num_mc_iterations - burn_in*num_mc_iterations

    mc_local_energy = local_energy_sum/samples
    mc_local_energy_psi_derivative_parameters = local_energy_psi_parameter_derivative_sum/samples
    mc_psi_derivative_parameters = psi_parameter_derivative_sum/samples

    local_energy_derivative_parameters = 2*(mc_local_energy_psi_derivative_parameters - mc_local_energy*mc_psi_derivative_parameters)

    return mc_local_energy, local_energy_derivative_parameters
end

function optimizationStep(system::slaterJastrow, gradKappa, learning_rate::Float64)
    system.jastrowFactor.kappa[:,:] = system.jastrowFactor.kappa[:,:] - learning_rate*gradKappa
end

function optimizationStep(system::slaterJastrow, optimizer, grad)
    update!(optimizer, system.jastrowFactor.kappa, grad)
end

"""
    runVMC(system::slaterJastrow, numVMCIterations::Int64, numMonteCarloIterations, mc_step_length, learning_rate)

Function for running the full simulation for a Slater-Jastrow wavefunction. 
"""
function runVMC(system::slaterJastrow, numVMCIterations::Int64, numMonteCarloIterations, mc_step_length, optimizer)
    local_energies::Array{Float64, 2} = zeros(Float64, (numVMCIterations, 1))
    for k = 1:numVMCIterations
        local_energy, grads = runMetropolisBruteForce(system, numMonteCarloIterations, mc_step_length)
        optimizationStep(system, optimizer, grads)
        local_energies[k] = local_energy
        println("Iteration = ", k, "    E = ", local_energy)
    end
    return local_energies
end


function runMetropolisBruteForce(system::gaussianJastrow, num_mc_iterations, step_length; burn_in = 0.01, writeToFile= false)
    local_energy_sum::Float64 = 0.0

    local_energy_psi_parameter_derivative_sum = 0

    psi_parameter_derivative_sum = 0

    local_energies::Array{Float64, 1} = zeros(Float64, Int(num_mc_iterations))

    start = time()

    for i = 1:num_mc_iterations
        metropolisStepBruteForce(step_length, system)
        # metropolisStepImportanceSampling(step_length, system)

        jastrowUpdateDistanceMatrix(system)

        local_energy = computeLocalEnergy(system, system.interacting)
        # println(local_energy)
        local_energies[i] = local_energy

        psi_parameter_derivative = simpleJastrowComputeParameterGradient(system)

        if i > burn_in*num_mc_iterations
            local_energy_sum += local_energy
            local_energy_psi_parameter_derivative_sum += local_energy*psi_parameter_derivative
            psi_parameter_derivative_sum += psi_parameter_derivative
        end
    end

    if writeToFile
        open("output.txt", "w") do file
            for e in local_energies
                println(file, e)
            end
        end
    end

    runtime = time() - start

    samples = num_mc_iterations - burn_in*num_mc_iterations

    mc_local_energy = local_energy_sum/samples
    mc_local_energy_psi_derivative_parameters = local_energy_psi_parameter_derivative_sum/samples
    mc_psi_derivative_parameters = psi_parameter_derivative_sum/samples

    local_energy_derivative_parameters = 2*(mc_local_energy_psi_derivative_parameters - mc_local_energy*mc_psi_derivative_parameters)

    println(mc_local_energy)

    return mc_local_energy, [local_energy_derivative_parameters]
end

function optimizationStep(system::gaussianJastrow, optimizer, grad)
    update!(optimizer, system.jastrowFactor.beta, grad)
end




function runVMC(system, numVMCIterations::Int64, numMonteCarloIterations, mc_step_length, optimizer)
    println("test simple Jas")
    local_energies::Array{Float64, 2} = zeros(Float64, (numVMCIterations, 1))
    for k = 1:numVMCIterations
        local_energy, grads = runMetropolisBruteForce(system, numMonteCarloIterations, mc_step_length)
        optimizationStep(system, optimizer, grads)
        local_energies[k] = local_energy
        println("Iteration = ", k, "    E = ", local_energy)
    end
    return local_energies
end


end #MODULE