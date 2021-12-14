module vmc 

export runVMC!

using ..metropolis
using Flux:update!, ADAM, Descent
using ..harmonicOscillator
using ..slater
using ..gaussian
using ..jastrow 
using ..rbm 
using ..nn

function runVMC!(system, numOptimizationIterations, numMCMCIterations, mcmcStepLength, optimizer, sampler = "bf", write_to_file = false)
    local_energies::Array{Float64, 2} = zeros(Float64, (numMCMCIterations, 1))
    for k = 1:numOptimizationIterations
        local_energy, grads = runMetropolis!(system, numMCMCIterations, mcmcStepLength, sampler = sampler)
        update!(optimizer, last(system.wavefunctionElements).variationalParameter, grads)
        local_energies[k] = local_energy
        println("Iteration = ", k, "    E = ", local_energy, "variationalParameter = ", last(system.wavefunctionElements).variationalParameter)
    end

    if write_to_file
        filename = makeFilename(system, mcmcStepLength, numMCMCIterations, numOptimizationIterations, sampler, optimizer)
        saveDataToFile(local_energies, filename)
    end
    return local_energies
end

function saveDataToFile(data, filename::String)
    open(filename, "w") do file
        for d in data
            println(file, d)
        end
    end
end

function makeFilename(system, steplength, numMCsteps, numoptimsteps, sampler, optimizier)
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
    filename = "Data/VMC/" * folder * "/" * wavefunctionCombination * "sysInfo_" * sampler  * "_stepLength_" * string(steplength)* "_numMCSteps_"* string(numMCsteps) * "_optimizer_" * optimizerName(optimizer) * "_learningRate_" * string(optimizer.eta) * "_numOptimSteps_" * string(numoptimsteps)  * "_numDims_" * string(system.numDimensions) * "_numParticles_" * string(system.numParticles) * wavefunctionElementsInfo *".txt"
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
    return [("nn_numhidden1_" * string(size(element.a[1]))* "_numhidden2_ "* string(size(element.a[2]))), "nn"]
end

function optimizerName(optimizer::ADAM)
    return "adam"
end

function optimizerName(optimizer::Descent)
    return "gd"
end
#End module
end