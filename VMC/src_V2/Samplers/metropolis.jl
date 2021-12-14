module metropolis 

export runMetropolis!

using ..harmonicOscillator
using ..slater
using ..gaussian
using ..jastrow 
using ..rbm 
using ..nn

function runMetropolis!(
    system, 
    num_mc_iterations::Int64, 
    step_length::Float64; 
    sampler = "bf", 
    burn_in = 0.01, 
    write_to_file = false, 
    calculate_onebody = false
)
    
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

    if calculate_onebody
        numBins = 1000
        maxLength = 10
        dr = maxLength/numBins
        onebody = zeros(1000)
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

        if calculate_onebody
            for particle=1:system.numParticles
                r = sqrt(sum(system.particles[particle,:].^2))
                onebody[floor(Int, r ÷ dr) + 1] += 1
            end
        end
    end
    if calculate_onebody
        saveDataToFile(onebody, "onebodytest.txt")
    end 

    runtime = time() - start

    if write_to_file
        filename = makeFilename(system,step_length, num_mc_iterations, sampler)
        saveDataToFile(local_energies, filename)
    end

    samples = num_mc_iterations - burn_in*num_mc_iterations

    mc_local_energy = local_energy_sum/samples
    mc_local_energy_psi_derivative_a = local_energy_psi_parameter_derivative_sum/samples
    mc_psi_derivative_a = psi_parameter_derivative_sum/samples
    local_energy_derivative_a = 2*(mc_local_energy_psi_derivative_a - mc_local_energy*mc_psi_derivative_a)

    println("Ground state energy: ", mc_local_energy)

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

function saveDataToFile(data, filename::String)
    open(filename, "w") do file
        for d in data
            println(file, d)
        end
    end
end

function makeFilename(system, steplength, numsteps, sampler)
    wavefunctionCombination = "wf_"
    wavefunctionElementsInfo = "_elementinfo_"
    for element in system.wavefunctionElements
        elementinfo = wavefunctionName(element) 
        wavefunctionCombination  = wavefunctionCombination * elementinfo[2] * "_"
        wavefunctionElementsInfo = wavefunctionElementsInfo * elementinfo[1] * "_"
    end

    if system.interacting == true
        folder = "Interacting"
    elseif system.interacting == false
        folder = "Non_Interacting"
    end
    println(wavefunctionCombination)
    filename = "Data/MC/" * folder *"/" * wavefunctionCombination * "sysInfo_" * sampler * "_stepLength_" * string(steplength)* "_numMCSteps_"* string(numsteps) * "_numDims_" * string(system.numDimensions) * "_numParticles_" * string(system.numParticles) * wavefunctionElementsInfo *".txt"
    return filename
end

function wavefunctionName(element::SlaterMatrix)
    return ["slater_none", "slater"]
end

function wavefunctionName(element::Gaussian)
    return ["gaussian_none", "gaussian"]
end

function wavefunctionName(element::Jastrow)
    return ["jastrow_none", "jastrow"]
end

function wavefunctionName(element::RBM)
    return [("rbm_numhidden_" * string(size(element.h)[1])), "rbm"]
end

function wavefunctionName(element::NN)
    return [("nn_numhidden1_" * string(size(element.a[1])) * "_numhidden2_ " * string(size(element.a[2]))), "nn"]
end

end