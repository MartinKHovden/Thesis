module testNN

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

using Flux:sigmoid, Dense,Chain, params

numParticles = 4
numDimensions = 2
numHiddenNeurons = 5

systemNN = initializeSystemSlaterNN(numParticles, numDimensions, numHiddenNeurons=numHiddenNeurons)
systemNNAnalytical = initializeSystemSlaterNNAnalytical(numParticles, numDimensions, numNodesLayer1 = numHiddenNeurons, numNodesLayer2 = numHiddenNeurons)

systemNNAnalytical.nn.w1[:,:] = copy(systemNN.nn.model[1].W)
systemNNAnalytical.nn.b1[:] = copy(systemNN.nn.model[1].b)

systemNNAnalytical.nn.w2[:,:] = copy(systemNN.nn.model[2].W)
systemNNAnalytical.nn.b2[:] = copy(systemNN.nn.model[2].b)

systemNNAnalytical.nn.w3[:,:] = copy(systemNN.nn.model[3].W)
systemNNAnalytical.nn.b3[:] = copy(systemNN.nn.model[3].b)

systemNNAnalytical.particles[:,:] = systemNN.particles
println(systemNN.particles)
println(systemNNAnalytical.particles)

oldPosition = randn(numParticles, numDimensions)

println("Ratio NN =           ", nnComputeRatio(systemNN, oldPosition))
println("Rtio NN Analytical = ", nnAnalyticalComputeRatio!(systemNNAnalytical, oldPosition))
println("")
println("Gradient NN = ", nnComputeGradient(systemNN))
println("Gradient NN Analytical = ", nnAnalyticalComputeGradient!(systemNNAnalytical))
println("")
println("Laplacian NN = ", nnComputeLaplacian(systemNN))
println("Laplacian NN Analytical = ", nnAnalyticalComputeLaplacian!(systemNNAnalytical))
println("")

nnAnalyticalComputePsi!(systemNNAnalytical, systemNN.particles)
println("Params grad NN = ", nnComputeParameterGradient(systemNN)[params(systemNN.nn.model)[5]])
println("Params grad NN Analytical = ", nnAnalyticalComputeParameterGradient!(systemNNAnalytical)[3])
println("")
# println("Drift froce NN = ", nnComputeDriftForce(systemNN, 2, 2))
# println("Drift force NN Analytical = ", nnAnalyticalComputeDriftForce!(systemNNAnalytical, 2, 2))
end