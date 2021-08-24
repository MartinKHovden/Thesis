module main

include("initializeSystem.jl")

include("Wavefunctions/slaterDeterminant.jl")
include("Wavefunctions/jastrow.jl")
include("Wavefunctions/simpleJastrow.jl")
include("Wavefunctions/boltzmannMachine.jl")
include("Wavefunctions/neuralNetwork.jl")
include("Wavefunctions/neuralNetworkAnalytical.jl")

include("Hamiltonians/harmonicOscillator.jl")

include("Samplers/metropolisBruteForce.jl")

using .initializeSystem

using .slaterDeterminant
using .jastrow
using .simpleJastrow
using .neuralNetwork
using .neuralNetworkAnalytical

using .harmonicOscillator
using .metropolisBruteForce

using Flux

# system = initializeSystemSlater(2, 2, alpha=1.0, omega=1.0, interacting=true)
# @time runVMC(system, 70, 10^5, 0.5, 0.5)
# runMetropolisBruteForce(system, 10^7, 0.1, writeToFile=true)

# alpha = getVariationalParameters(system)["alpha"]

# system = initializeSystemSlaterJastrow(2,2, alpha=1.0, interacting=true)
# opt = Descent(0.05)
# @time runVMC(system, 400, 100000, 5.0, opt)
# runMetropolisBruteForce(system, 1000000, 5.0; writeToFile = true)

# system = initializeSystemSlaterNN(2, 2, alpha = 1.0, interacting=false, numHiddenNeurons=10)
# @time runVMC(system, 40, 10^4, 5.0, 0.01) #(5.0, 0.01)
# @time runVMC(system, 100, 10^4, 0.01, 0.01) 
# @time runMetropolisBruteForce(system, 3, 0.1)

#@time runVMC(system, 100, 10^4, 0.5, 0.5)

# system = initializeSystemSlaterRBM(2, 2, 3, alpha=1.0, interacting=true)
# opt = ADAM(0.05)
# @time runVMC(system, 100, 10^5, 0.05, opt, writeToFile=true, sampler="is") #(5.0,0.5)
# runMetropolis(system, 2^20, 0.005; writeToFile = true, sampler="is")
# runVMC(system, 1, 10^6, 0.5, 0.5)

# opt = ADAM(0.05)
# system = initializeSystemSlaterNNAnalytical(2, 2, alpha = 1.0, interacting = true, numNodesLayer1 = 4, numNodesLayer2 = 4)
# runVMC(system, 1000, 10^4, 5.0, opt)# (0.05, 5.0)
# @time runMetropolisBruteForce(system, 3, 0.1)

opt = ADAM(0.05)
system = initializeSystemSlaterJastrowNNAnalytical(2, 2, alpha = 1.0, interacting = true, numNodesLayer1 = 4, numNodesLayer2 = 4)
runVMC(system, 1000, 10^4, 5.0, opt)# (0.05, 5.0)
@time runMetropolisBruteForce(system, 3, 0.1)

# opt = ADAM(0.005)
# system = initializeSystemGaussianJastrow(3,1, alpha = 1.0, beta=2.4)
# println(system.jastrowFactor)
# runVMC(system, 100, 10^5, 0.5, opt)
# runMetropolisBruteForce(system, 1000000, 0.05)

# opt = ADAM(0.05)
# system = initializeSystemGaussianNNAnalytical(5,1, alpha = 1.0, beta=2.0)
# runMetropolisBruteForce(system, 10^4, 0.05)
# runVMC(system, 10000, 10000, 0.05, opt)

end # MODULE