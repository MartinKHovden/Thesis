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
using .boltzmannMachine
using .neuralNetwork
using .neuralNetworkAnalytical

using .harmonicOscillator
using .metropolisBruteForce

using Flux

# system = initializeSystemSlater(6, 2, alpha=1.0, omega=1.0, interacting=false)
# @time runVMC(system, 50, 10^5, 5.0, 0.05)
# runMetropolisBruteForce(system, 10^7, 0.1, writeToFile=true)

# alpha = getVariationalParameters(system)["alpha"]


# system = initializeSystemSlaterJastrow(2,2, alpha=alpha, interacting=true)
# opt = Descent(0.05)
# @time runVMC(system, 100, 100000, 0.05, opt, sampler="is", writeToFile = true)
# println(system.jastrowFactor.kappa)
# runMetropolisBruteForce(system, 2^21, 5.0; sampler="is", writeToFile = true)

# system = initializeSystemSlaterNN(2, 2, alpha = 0.6, interacting=true, numHiddenNeurons=10)
# @time runVMC(system, 40, 10^4, 5.0, 0.01) #(5.0, 0.01)
# @time runVMC(system, 100, 10^4, 0.01, 0.01) 
# @time runMetropolisBruteForce(system, 3, 0.1)

#@time runVMC(system, 100, 10^4, 0.5, 0.5)

# system = initializeSystemSlaterRBM(2, 2, 4, alpha=0.75, interacting=true)
# opt = ADAM(0.05)
# @time runVMC(system, 100, 10^5, 5.0, opt, writeToFile=true, sampler="is") #(5.0,0.5) (0.05)
# runMetropolis(system, 2^21, 5.0; writeToFile = true, sampler="is")

opt = ADAM(0.05)
system = initializeSystemSlaterNNAnalytical(6, 2, alpha = 0.6, interacting = true, numNodesLayer1 = 20, numNodesLayer2 = 10)
runVMC(system, 50, 10^4, 0.0001, opt, writeToFile = true, sampler = "is")#, writeToFile=false, sampler="is")# (0.05, 5.0)
@time runMetropolisBruteForce(system, 2^20, 0.0001, sampler = "is", writeToFile = true)


# kappa = system.jastrowFactor.kappa
# println(kappa)

# opt = ADAM(0.05)
# system = initializeSystemSlaterJastrowNNAnalytical(2, 2, alpha = 1.0, interacting = true, numNodesLayer1 = 10, numNodesLayer2 = 10)

# system.jastrowFactor.kappa[:,:] = [0.0 0.0907814;
                                    # 0.0907814 0.0]
# # println(system)
# runVMC(system, 1000, 10^5, 5.0, opt, sampler="is", writeToFile=true)# (0.05, 5.0)
# @time runMetropolisBruteForce(system, 1000000, 1.0, sampler="is", writeToFile = false)

# opt = Descent(0.005)
# system = initializeSystemGaussianJastrow(5,1, alpha = 1.0, beta=2.4)
# println(system.jastrowFactor)
# runVMC(system, 100, 10^5, 0.5, opt)
# runMetropolisBruteForce(system, 1000000, 0.05)

# opt = ADAM(0.005)
# system = initializeSystemGaussianNNAnalytical(5,1, alpha = 1.0, beta=2.0)
# runMetropolisBruteForce(system, 10^5, 0.005)
# runVMC(system, 10000, 100000, 0.05, opt)

# opt = ADAM(0.005)
# system = initializeSystemGaussianJastrowNNAnalytical(5,1, alpha=1.0, beta=2.0)
# runMetropolisBruteForce(system, 10^4, 0.05, 1)
# runVMC(system, 100, 10000, 0.5, opt)

# opt = Descent(0.0005)
# runVMC(system, 100, 100000, 0.5, opt)

end # MODULE