module QDSlaterJastrow

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
# numHiddenLayers = parsed_args["numHiddenLayers"]
mcmcStepLength = parsed_args["stepLength"]

numOptimizationSteps =1000 # 10000
numMCMCSteps = 5000 # 100000

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
# addWaveFunctionElement(s, NN(s, numHiddenLayers, numHiddenLayers, "sigmoid"))
addWaveFunctionElement(s, PadeJastrow(s, beta=1.0))

runVMC!(s, numOptimizationSteps, numMCMCSteps, mcmcStepLength, optim, sampler = "is", writeToFile = true)

numMCSamplesFinal = 2^24

runMetropolis!(s, 
                            numMCSamplesFinal,  
                            mcmcStepLength, 
                            sampler="is", 
                            writeToFile = true, 
                            calculateOnebody = true,
                            learningRate = learningrate)

end

