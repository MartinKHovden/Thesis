module main

include("initializeSystem.jl")

include("Wavefunctions/slaterDeterminant.jl")
include("Wavefunctions/jastrow.jl")
include("Wavefunctions/boltzmannMachine.jl")
include("Wavefunctions/neuralNetwork.jl")
include("Wavefunctions/neuralNetworkAnalytical.jl")

include("Hamiltonians/harmonicOscillator.jl")

include("Samplers/metropolisBruteForce.jl")

using .initializeSystem

using .slaterDeterminant
using .jastrow
using .neuralNetwork
using .neuralNetworkAnalytical

using .harmonicOscillator
using .metropolisBruteForce

using Flux

# system = initializeSystemSlater(2, 2, alpha=1.1, omega=1.0, interacting=false)
# @time runVMC(system, 70, 10^5, 0.5, 0.5)
# runMetropolisBruteForce(system, 10^7, 0.1, writeToFile=true)

# alpha = getVariationalParameters(system)["alpha"]

# system = initializeSystemSlaterJastrow(2, 2, alpha=1.0, interacting=false)
# @time runVMC(system, 40, 100000, 0.5, 0.5)
# runMetropolisBruteForce(system, 1000000, 0.5; writeToFile = true)

# system = initializeSystemSlaterNN(2, 2, alpha = 1.0, interacting=false, numHiddenNeurons=10)
# @time runVMC(system, 40, 10^4, 5.0, 0.01) #(5.0, 0.01)
# @time runVMC(system, 100, 10^4, 0.01, 0.01) 
# @time runMetropolisBruteForce(system, 3, 0.1)

#@time runVMC(system, 100, 10^4, 0.5, 0.5)

# system = initializeSystemSlaterRBM(6, 2, 3, alpha=1.0, interacting=false)
# opt = ADAM(0.05)
# @time runVMC(system, 400, 10^5, 5.0, opt, writeToFile=true, sampler="bf") #(5.0,0.5)
# runMetropolis(system, 2^20, 0.005; writeToFile = true, sampler="is")
# runVMC(system, 1, 10^6, 0.5, 0.5)

# opt = ADAM(0.05)
# system = initializeSystemSlaterNNAnalytical(6, 2, alpha = 1.0, numNodesLayer1 = 10, numNodesLayer2 = 10)
# runVMC(system, 1000, 10^5, 5.0, opt)
# @time runMetropolisBruteForce(system, 3, 0.1)

end # MODULE