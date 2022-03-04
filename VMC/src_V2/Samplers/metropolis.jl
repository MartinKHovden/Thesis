module metropolis 

export runMetropolis!

using ..harmonicOscillator
using ..slater
using ..gaussian
using ..jastrow 
using ..padeJastrow
using ..rbm 
using ..nn

"""
    runMetropolis!(args...)

Function for running the Metropolis algorithm.

# Arguments
- `system`: The system struct.
- `numMCIterations::Int64`: 
- `stepLength::Float64`: 
- `sampler`:
- `burnIn`:
- `writeToFile`:
- `calculateOnebody`:

# Returns
- `Float`: The local energy of the system. 
"""
function runMetropolis!(
    system, 
    numMcIterations::Int64, 
    stepLength::Float64;
    optimizationIteration::Int64 = 0,
    sampler = "bf", 
    burnIn = 0.001, 
    writeToFile = false, 
    calculateOnebody = false
)
    
    localEnergySum::Float64 = 0.0

    localEnergies::Array{Float64, 1} = zeros(Float64, Int(numMcIterations))
    localEnergyPsiParameterDerivativeSum = 0 .*last(system.wavefunctionElements).variationalParameterGradient
    psiParameterDerivativeSum = 0 .*last(system.wavefunctionElements).variationalParameterGradient

    start = time()

    if sampler == "bf"
        stepFunction = metropolisStepBruteForce!
    elseif sampler == "is"
        stepFunction = metropolisStepImportanceSampling!
    else
        println("Sampler not implemented")
        exit(100)
    end

    if calculateOnebody
        numBins = 4000
        maxLength = 10
        dr = maxLength/numBins
        onebody = zeros(numBins)
    end

    optimizerElement = last(system.wavefunctionElements)
    for i = 1:numMcIterations
        stepFunction(system, stepLength)

        localEnergy = computeLocalEnergy(system, i)
        # println(localEnergy)
        localEnergies[i] = localEnergy

        optimizerElement.variationalParameterGradient = computeParameterGradient(system, optimizerElement)
        if i > ceil(burnIn*numMcIterations)
            localEnergySum += localEnergy
            localEnergyPsiParameterDerivativeSum += localEnergy*optimizerElement.variationalParameterGradient
            psiParameterDerivativeSum += optimizerElement.variationalParameterGradient

            if calculateOnebody
                for particle=1:system.numParticles
                    r = sqrt(sum(system.particles[particle,:].^2))
                    onebody[floor(Int, r ÷ dr) + 1] += 1
                end
            end 
        end

        
    end

    if calculateOnebody
        saveDataToFile(onebody, "Data/"*system.hamiltonian* "/OneBody/onebodytest.txt")
    end 

    runtime = time() - start

    if writeToFile
        filename = makeFilename(system,stepLength, numMcIterations, sampler)
        saveDataToFile(localEnergies, filename)
    end

    samples = numMcIterations - ceil(burnIn*numMcIterations) 

    mcLocalEnergy = localEnergySum/samples
    mcLocalEnergyPsiParameterDerivative = localEnergyPsiParameterDerivativeSum/samples
    mcPsiParameterDerivative = psiParameterDerivativeSum/samples
    localEnergyParameterDerivative = 2*(mcLocalEnergyPsiParameterDerivative - mcLocalEnergy*mcPsiParameterDerivative)

    # println("Ground state energy: ", mcLocalEnergy)

    return mcLocalEnergy, localEnergyParameterDerivative
end 

"""
    metropolisStepBruteForce!(args...)

Computes the ratio between the current and previous squared wave function value.

# Arguments
- `system`: The system struct.
- `stepLength`:

# Returns
- `Float`: The local energy of the system. 
"""
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
        ratio *= computeRatio(system, 
                            element, 
                            particleToUpdate, 
                            coordinateToUpdate, 
                            oldPosition)
    end

    U = rand(Float64)

    if U < ratio
        if system.slaterInWF
            inverseSlaterMatrixUpdate(system, 
                                    system.wavefunctionElements[1], 
                                    particleToUpdate, 
                                    system.wavefunctionElements[1].R)
        end
    else 
        system.particles[particleToUpdate, coordinateToUpdate] = oldPosition[particleToUpdate, coordinateToUpdate]
        for element in system.wavefunctionElements
            updateElement!(system, element, particleToUpdate)
        end
    end
end

"""
    metropolisStepImportanceSampling!(args...)

Computes the ratio between the current and previous squared wave function value.

# Arguments
- `system`: The system struct.
- `stepLength`:

# Returns
- `Float`: The local energy of the system. 
"""
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
        currentDriftForce += computeDriftForce(system, 
                                            element, 
                                            particleToUpdate, 
                                            coordinateToUpdate)
    end

    system.particles[particleToUpdate, coordinateToUpdate] += D*currentDriftForce*stepLength + randn(Float64)*sqrt(stepLength)

    newDriftForce = 0.0
    for element in system.wavefunctionElements
        newDriftForce += computeDriftForce(system, element, particleToUpdate, coordinateToUpdate)
    end

    greensFunction = computeGreensFunction(oldPosition,
                                        system.particles,
                                        particleToUpdate,
                                        coordinateToUpdate,
                                        currentDriftForce,
                                        newDriftForce,
                                        D,
                                        stepLength)

    # Update the slater matrix:
    ratio = 1.0

    for element in system.wavefunctionElements
        updateElement!(system, element, particleToUpdate)
        ratio *= computeRatio(system, 
                            element,
                            particleToUpdate,
                            coordinateToUpdate,
                            oldPosition)
    end

    U = rand(Float64)



    if U < greensFunction*ratio
        # println(1, " ", greensFunction, ratio, greensFunction*ratio)
        if system.slaterInWF
            inverseSlaterMatrixUpdate(system, system.wavefunctionElements[1], particleToUpdate, system.wavefunctionElements[1].R)
        end
    else 
        # println(2, " ", greensFunction, ratio, greensFunction*ratio)
        system.particles[particleToUpdate, coordinateToUpdate] = oldPosition[particleToUpdate, coordinateToUpdate]
        for element in system.wavefunctionElements
            updateElement!(system, element, particleToUpdate)
        end
    end
end

"""
    computeGreensFunction(args...)

Computes the ratio between the current and previous squared wave function value.

# Arguments
- `system`: The system struct.
- `stepLength`:

# Returns
- `Float`: The local energy of the system. 
"""
function computeGreensFunction(oldPosition, 
                            newPosition, 
                            particleToUpdate,        
                            coordinateToUpdate, 
                            oldDriftForce, 
                            newDriftForce, 
                            D,
                            stepLength)

    greensFunctionArgument = (oldPosition[particleToUpdate, coordinateToUpdate] +
                                - newPosition[particleToUpdate, coordinateToUpdate] +
                                - D*stepLength*newDriftForce)^2 +
                                - (newPosition[particleToUpdate, coordinateToUpdate] + 
                                - oldPosition[particleToUpdate, coordinateToUpdate] +
                                - D*stepLength*oldDriftForce)^2

    greensFunctionArgument /= (4.0*D*stepLength)
    greensFunction = exp(-greensFunctionArgument)
    return greensFunction
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
    filename = "Data/"* system.hamiltonian * "/MC/" * folder *"/" * wavefunctionCombination * "sysInfo_" * sampler * "_stepLength_" * string(steplength)* "_numMCSteps_"* string(numsteps) * "_numDims_" * string(system.numDimensions) * "_numParticles_" * string(system.numParticles) * wavefunctionElementsInfo *".txt"
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

function wavefunctionName(element::PadeJastrow)
    return ["padeJastrow_none", "padeJastrow"]
end

function wavefunctionName(element::RBM)
    return [("rbm_numhidden_" * string(size(element.h)[1])), "rbm"]
end

function wavefunctionName(element::NN)
    return [("nn_numhidden1_" * string(size(element.a[1])[1]) * "_numhidden2_" * string(size(element.a[2])[1])) * "_activationFunction_" * string(element.activationFunction), "nn"]
end

end