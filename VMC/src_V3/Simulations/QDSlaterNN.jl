module QDSlaterNN

include("../MLVMC.jl")
using .MLVMC
using ArgParse

#Set up the system:
s = ArgParseSettings()
@add_arg_table s begin
    "--numParticles"
        help = "Number of particles in the system"
        arg_type = Int
    "--numDimensions" 
        help = "Number of dimensions in the system"
        arg_type = Int
    "--omega"
        help = "an option without argument, i.e. a flag"
        arg_type = Float64
    "--learningRate"
        help = "learning rate of optim"
        arg_type = Float64
    "--numHiddenLayers"
        help = "num hidden layers in network"
        arg_type = Int
    "--stepLength"
        arg_type = Float64
end

parsed_args = parse_args(ARGS, s)
println(parsed_args)

hamiltonian = "quantumDot" # Use "quantumDot" or "calogeroSutherland" or "bosons"
interactingParticles = true

# numParticles = 6
# numDimensions = 2
# harmonicOscillatorFrequency = 0.5
# learningrate = 0.01
# numHiddenLayers = 2
# mcmcStepLength = 0.001

numParticles = parsed_args["numParticles"]
numDimensions = parsed_args["numDimensions"]
harmonicOscillatorFrequency = parsed_args["omega"]
learningrate = parsed_args["learningRate"]
numHiddenLayers = parsed_args["numHiddenLayers"]
mcmcStepLength = parsed_args["stepLength"]

numOptimizationSteps =20 # 10000
numMCMCSteps = 30 # 100000

# #Set up the optimiser from Flux: 
# learningrate = 0.01#0.01
optim = ADAM(learningrate)

s = System(numParticles, 
    numDimensions, 
    hamiltonian, 
    omega=harmonicOscillatorFrequency, 
    interacting = interactingParticles)

#Add the wavefunction elements:
addWaveFunctionElement(s, SlaterMatrix( s ))
addWaveFunctionElement(s, Gaussian( 1.0 ))
addWaveFunctionElement(s, NN(s, numHiddenLayers, numHiddenLayers, "sigmoid"))

runVMC!(s, numOptimizationSteps, numMCMCSteps, mcmcStepLength, optim, sampler = "is", writeToFile = true)

numMCSamplesFinal = 2^2

runMetropolis!(s, 
                            numMCSamplesFinal,  
                            mcmcStepLength, 
                            sampler="is", 
                            writeToFile = true, 
                            calculateOnebody = true)

numReruns = 5
energies = zeros(numReruns)
for i =1:numReruns
    @time energy =  runMetropolis!(s, 
                            numMCSamplesFinal,  
                            mcmcStepLength, 
                            sampler="is", 
                            writeToFile = false, 
                            calculateOnebody = false)
    energies[i] = energy
end

println(energies)



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

function wavefunctionName(element::GaussianSimple)
    return ["gaussianSimple_none", "gaussianSimple"]
end

function wavefunctionName(element::RBM)
    return [("rbm_numhidden_" * string(size(element.h)[1])), "rbm"]
end

function wavefunctionName(element::NN)
    return [("nn_nh1_" * string(size(element.a[1])[1]) * "_nh2_" * string(size(element.a[2])[1])) * "_af_" * string(element.activationFunction), "nn"]
end


function makeFilenameRerunEnergies(system, steplength, numMCsteps, sampler)
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
    filename = "Data/"* system.hamiltonian * "/RerunEnergies/" * folder *"/" * wavefunctionCombination * "sysInfo_" * sampler  *"_omega_" * string(system.omega) * "_sl_" * string(steplength)* "_mcSteps_"* string(numMCsteps) * "_numD_" * string(system.numDimensions) * "_numP_" * string(system.numParticles) * wavefunctionElementsInfo *".txt"
    return filename
end
function saveDataToFile(data)
    open(makeFilenameRerunEnergies(s, mcmcStepLength, numMCMCSteps, "is"), "w") do file
        for d in data
            println(file, d)
        end
    end
end

saveDataToFile(energies)





end

