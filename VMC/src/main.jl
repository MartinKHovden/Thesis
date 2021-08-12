module main

include("initializeSystem.jl")

include("Wavefunctions/slaterDeterminant.jl")
include("Wavefunctions/jastrow.jl")
include("Wavefunctions/boltzmannMachine.jl")
include("Wavefunctions/neuralNetwork.jl")

include("Hamiltonians/harmonicOscillator.jl")

include("Samplers/metropolisBruteForce.jl")


using .initializeSystem
using .slaterDeterminant
using .jastrow
using .neuralNetwork
using .harmonicOscillator
using .metropolisBruteForce

# system = initializeSystemSlater(2, 2, alpha=1.1, omega=1.0, interacting=false)
# @time runVMC(system, 100, 10^5, 0.1, 0.1)
# runMetropolisBruteForce(system, 10^7, 5.0, writeToFile=true)

# alpha = getVariationalParameters(system)["alpha"]

# system = initializeSystemSlaterJastrow(2, 2, alpha=1.0, interacting=false)
# @time runVMC(system, 40, 100000, 0.5, 0.5)
# runMetropolisBruteForce(system, 1000000, 0.5; writeToFile = true)

# system = initializeSystemSlaterNN(2, 2, alpha = 1.0, interacting=false, numHiddenNeurons=2)
# @time runVMC(system, 40, 10^4, 5.0, 0.01) #(5.0, 0.01)
# @time runVMC(system, 100, 10^4, 0.01, 0.01) 

#@time runVMC(system, 100, 10^4, 0.5, 0.5)

system = initializeSystemSlaterRBM(2, 2, 3, alpha=1.0, interacting=false)
@time runVMC(system, 40, 10^5, 0.05, 0.5, writeToFile=true, sampler="is")
runMetropolis(system, 2^20, 0.05; writeToFile = true, sampler="is")
# runVMC(system, 1, 10^6, 0.5, 0.5)

end # MODULE