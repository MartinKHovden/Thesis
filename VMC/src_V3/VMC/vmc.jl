module vmc 

export runVMC!

using ..metropolis
using Flux:update!, ADAM, Descent
using ..harmonicOscillator
using ..slater
using ..gaussian
using ..gaussianSimple
using ..jastrow 
using ..padeJastrow
using ..rbm 
using ..nn

function runVMC!(system,
                numOptimizationIterations,
                numMCMCIterations,
                mcmcStepLength,
                optimizer;
                sampler = "bf", 
                writeToFile = false,
                sortInput = false)     
    localEnergies::Array{Float64, 2} = zeros(Float64, (numOptimizationIterations, 1))
    variationalParameters::Array{Float64, 2} = zeros(Float64, (numOptimizationIterations, 1))
    weightNorms = []
    for k = 1:numOptimizationIterations
        localEnergy = runMetropolis!(system, numMCMCIterations, mcmcStepLength, optimizationIteration = k, sampler = sampler, sortInput = sortInput)
        for wavefunctionElement in system.wavefunctionElements
            if typeof(wavefunctionElement) != SlaterMatrix
                numGrads = size((wavefunctionElement).variationalParameter)[1]
                for i = 1:numGrads
                    update!(optimizer, wavefunctionElement.variationalParameter[i], wavefunctionElement.variationalParameterGradient[i])
                end
            end
        end
        localEnergies[k] = localEnergy
        variationalParameters[k] = system.wavefunctionElements[2].variationalParameter[1][1]
        println(sum(system.wavefunctionElements[3].variationalParameter[1].^2))
        push!(weightNorms, system.wavefunctionElements[3].variationalParameter[1][1])
        # push!(weightNorms, [sum(system.wavefunctionElements[3].variationalParameter[1].^2), sum(system.wavefunctionElements[3].variationalParameter[2].^2), sum(system.wavefunctionElements[3].variationalParameter[3].^2)])
        println("Iteration = ", k, "    E = ", localEnergy, "variationalParameter = ", system.wavefunctionElements[2].variationalParameter[1][1])
        # println("variationalParameter = ", system.wavefunctionElements[3].variationalParameter[1][1])
    end

    if writeToFile
        filenameEnergy = makeFilenameOptimizationEnergy(system, 
                                mcmcStepLength, 
                                numMCMCIterations, 
                                numOptimizationIterations, 
                                sampler, 
                                optimizer)
        saveDataToFile(localEnergies, filenameEnergy)

        filenameAlpha = makeFilenameOptimizationAlpha(system, 
                                mcmcStepLength, 
                                numMCMCIterations, 
                                numOptimizationIterations, 
                                sampler, 
                                optimizer)
        saveDataToFile(variationalParameters, filenameAlpha)

        filenameWeightNorms = makeFilenameOptimizationWeightNorms(system, 
                                mcmcStepLength, 
                                numMCMCIterations, 
                                numOptimizationIterations, 
                                sampler, 
                                optimizer)
        saveDataToFile(weightNorms, filenameWeightNorms)
    end
    return localEnergies
end

function saveDataToFile(data, filename::String)
    open(filename, "w") do file
        for d in data
            println(file, d)
        end
    end
end

function makeFilenameOptimizationEnergy(system, steplength, numMCsteps, numoptimsteps, sampler, optimizer)
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
    filename = "Data/" * system.hamiltonian * "/VMC/" * folder * "/energy_" * wavefunctionCombination * "sysInfo_" * sampler  *"_omega_" * string(system.omega) * "_sl_" * string(steplength)* "_mcSteps_"* string(numMCsteps) * "_optim_" * optimizerName(optimizer) * "_lr_" * string(optimizer.eta) * "_optSteps_" * string(numoptimsteps)  * "_numD_" * string(system.numDimensions) * "_numP_" * string(system.numParticles) * wavefunctionElementsInfo *".txt"
    return filename
end

function makeFilenameOptimizationAlpha(system, steplength, numMCsteps, numoptimsteps, sampler, optimizer)
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
    filename = "Data/" * system.hamiltonian * "/VMC/" * folder * "/alpha_" * wavefunctionCombination * "sysInfo_" * sampler  *"_omega_" * string(system.omega) * "_sl_" * string(steplength)* "_mcSteps_"* string(numMCsteps) * "_optim_" * optimizerName(optimizer) * "_lr_" * string(optimizer.eta) * "_optSteps_" * string(numoptimsteps)  * "_numD_" * string(system.numDimensions) * "_numP_" * string(system.numParticles) * wavefunctionElementsInfo *".txt"
    return filename
end

function makeFilenameOptimizationWeightNorms(system, steplength, numMCsteps, numoptimsteps, sampler, optimizer)
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
    filename = "Data/" * system.hamiltonian * "/VMC/" * folder * "/weightNorms_" * wavefunctionCombination * "sysInfo_" * sampler  *"_omega_" * string(system.omega) * "_sl_" * string(steplength)* "_mcSteps_"* string(numMCsteps) * "_optim_" * optimizerName(optimizer) * "_lr_" * string(optimizer.eta) * "_optSteps_" * string(numoptimsteps)  * "_numD_" * string(system.numDimensions) * "_numP_" * string(system.numParticles) * wavefunctionElementsInfo *".txt"
    return filename
end

function wavefunctionName(element::SlaterMatrix)
    return ["slater_none", "slater"]
end

function wavefunctionName(element::Gaussian)
    return ["gaussian_none", "gaussian"]
end

function wavefunctionName(element::GaussianSimple)
    return ["gaussianSimple_none", "gaussianSimple"]
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
    return [("nn_nh1_" * string(size(element.a[1])[1]) * "_nh2_" * string(size(element.a[2])[1])) * "_af_" * string(element.activationFunction), "nn"]
end

function optimizerName(optimizer::ADAM)
    return "adam"
end

function optimizerName(optimizer::Descent)
    return "gd"
end
#End module
end