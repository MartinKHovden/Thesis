module metropolis 

export runMetropolis!

using ..harmonicOscillator
using ..slater
using ..gaussian

function runMetropolis!(system, num_mc_iterations::Int64, step_length::Float64; sampler = "bf", burn_in = 0.01)
    local_energy_sum::Float64 = 0.0

    local_energies::Array{Float64, 1} = zeros(Float64, Int(num_mc_iterations))
    local_energy_psi_parameter_derivative_sum = 0 .*last(system.wavefunctionElements).variationalParameterGradient
    psi_parameter_derivative_sum = 0 .*last(system.wavefunctionElements).variationalParameterGradient

    start = time()

    if sampler == "bf"
        stepFunction = metropolisStepBruteForce!
    elseif sampler == "is"
        stepFunction = metropolisStepImportanceSampling!
    else 
        println("Sampler not implemented")
        exit(100)
    end

    optimizerElement = last(system.wavefunctionElements)

    for i = 1:num_mc_iterations
        stepFunction(system, step_length)

        local_energy = computeLocalEnergy(system)
        local_energies[i] = local_energy


        optimizerElement.variationalParameterGradient = computeParameterGradient(system, optimizerElement)

        if i > burn_in*num_mc_iterations
            local_energy_sum += local_energy
            local_energy_psi_parameter_derivative_sum += local_energy*optimizerElement.variationalParameterGradient
            psi_parameter_derivative_sum += optimizerElement.variationalParameterGradient
        end
    end

    runtime = time() - start

    samples = num_mc_iterations - burn_in*num_mc_iterations

    mc_local_energy = local_energy_sum/samples
    mc_local_energy_psi_derivative_a = local_energy_psi_parameter_derivative_sum/samples
    mc_psi_derivative_a = psi_parameter_derivative_sum/samples
    local_energy_derivative_a = 2*(mc_local_energy_psi_derivative_a - mc_local_energy*mc_psi_derivative_a)

    return mc_local_energy, local_energy_derivative_a
end 

function metropolisStepBruteForce!(system, stepLength)
    numParticles = system.numParticles 
    numDimensions = system.numDimensions

    # Chooses one coordinate randomly to update.
    coordinateToUpdate::Int64 = rand(1:numDimensions)
    particleToUpdate::Int64 = rand(1:numParticles)

    # Update the coordinate:
    oldPosition = copy(system.particles)
    system.particles[particleToUpdate, coordinateToUpdate] += (rand(Float64) - 0.5)*stepLength

    # Update the slater matrix:
    ratio = 1.0

    for element in system.wavefunctionElements
        updateElement!(system, element, particleToUpdate)
        ratio *= computeRatio(system, element, particleToUpdate, coordinateToUpdate, oldPosition)
    end

    U = rand(Float64)

    if U < ratio
        # println(system)
        if system.slaterInWF
            inverseSlaterMatrixUpdate(system, system.wavefunctionElements[1], particleToUpdate, system.wavefunctionElements[1].R)
        end
    else 
        # println("Here")
        system.particles[particleToUpdate, coordinateToUpdate] = oldPosition[particleToUpdate, coordinateToUpdate]
        # slaterMatrixUpdate(system, particleToUpdate)
        for element in system.wavefunctionElements
            updateElement!(system, element, particleToUpdate)
        end
    end
end

function metropolisStepImportanceSampling!(system, stepLength)
    numParticles = system.numParticles
    numDimensions = system.numDimensions

    # Chooses one coordinate randomly to update.
    coordinateToUpdate::Int64 = rand(1:numDimensions)
    particleToUpdate::Int64 = rand(1:numParticles)

    # Update the coordinate:
    oldPosition = copy(system.particles)

    D = 0.5

    currentDriftForce = 0.0
    for element in system.wavefunctionElements
        currentDriftForce += computeDriftForce(system, element, particleToUpdate, coordinateToUpdate)
    end

    system.particles[particleToUpdate, coordinateToUpdate] += D*currentDriftForce*stepLength + randn(Float64)*sqrt(stepLength)

    newDriftForce = 0.0
    for element in system.wavefunctionElements
        newDriftForce += computeDriftForce(system, element, particleToUpdate, coordinateToUpdate)
    end

    greensFunction = computeGreensFunction(oldPosition, system.particles, particleToUpdate,
                                            coordinateToUpdate, currentDriftForce, 
                                            newDriftForce, D, stepLength)

    # Update the slater matrix:
    ratio = 1.0

    for element in system.wavefunctionElements
        updateElement!(system, element, particleToUpdate)
        ratio *= computeRatio(system, element, particleToUpdate, coordinateToUpdate, oldPosition)
    end

    U = rand(Float64)

    if U < greensFunction*ratio
        # println(system)
        if system.slaterInWF
            inverseSlaterMatrixUpdate(system, system.wavefunctionElements[1], particleToUpdate, system.wavefunctionElements[1].R)
        end
    else 
        # println("Here")
        system.particles[particleToUpdate, coordinateToUpdate] = oldPosition[particleToUpdate, coordinateToUpdate]
        # slaterMatrixUpdate(system, particleToUpdate)
        for element in system.wavefunctionElements
            updateElement!(system, element, particleToUpdate)
        end
    end
end

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

end