module vmc 

export runVMC!

using ..metropolis
using Flux:update!, ADAM, Descent
using ..harmonicOscillator
using ..slater
using ..gaussian
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
        # println(optimizer)
        localEnergies[k] = localEnergy
        variationalParameters[k] = system.wavefunctionElements[2].variationalParameter[1][1]
        println("Iteration = ", k, "    E = ", localEnergy)#, "variationalParameter = ")
        display(system.wavefunctionElements[2].variationalParameter)
        # display(system.wavefunctionElements[3].variationalParameter )
    end

    if writeToFile
        open("testVMC.txt","w") do file 
            for d in localEnergies
                println(file,d)
            end 
        end 
        open("testVP.txt", "w") do file 
            for d in variationalParameters
                println(file,d)
            end 
        end
    end
        # filename = makeFilename(system, 
        #                         mcmcStepLength, 
        #                         numMCMCIterations, 
        #                         numOptimizationIterations, 
        #                         sampler, 
        #                         optimizer)
        # saveDataToFile(localEnergies, filename)
    return localEnergies
end

function saveDataToFile(data, filename::String)
    open(filename, "w") do file
        for d in data
            println(file, d)
        end
    end
end

# function checkIfOverlap(system)
#     for i=1:system.numParticles
#         for j=i+1:system.numParticles
#             if system.particles

function makeFilename(system, steplength, numMCsteps, numoptimsteps, sampler, optimizer)
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
    filename = "Data/" * system.hamiltonian * "/VMC/" * folder * "/" * wavefunctionCombination * "sysInfo_" * sampler  * "_stepLength_" * string(steplength)* "_numMCSteps_"* string(numMCsteps) * "_optimizer_" * optimizerName(optimizer) * "_learningRate_" * string(optimizer.eta) * "_numOptimSteps_" * string(numoptimsteps)  * "_numDims_" * string(system.numDimensions) * "_numParticles_" * string(system.numParticles) * wavefunctionElementsInfo *".txt"
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

function optimizerName(optimizer::ADAM)
    return "adam"
end

function optimizerName(optimizer::Descent)
    return "gd"
end
#End module
end