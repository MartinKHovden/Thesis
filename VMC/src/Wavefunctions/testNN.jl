module testNN 

# include("nn.jl")
include("neuralNetwork.jl")
include("analyticalNN.jl")

using .neuralNetwork
using .analyticalNN
using Flux:Chain, Dense, Params, gradient, params, sigmoid

numParticles = 4
numDimensions = 2

numNodesLayer1 = 10
numNodesLayer2 = 10

numIterations = 100000

particlesOld = randn(numParticles, numDimensions)
particlesNew = particlesOld + 0.05*randn(numParticles, numDimensions)

struct NN
    model
end

struct systemNN 
    particles::Array{Float64, 2}
    numParticles::Int64 
    numDimensions::Int64

    interacting::Bool
    nn::NN
end

nn = NN(Chain(Dense(numParticles*numDimensions, numNodesLayer1, sigmoid), Dense(numNodesLayer1,numNodesLayer2, sigmoid), Dense(numNodesLayer2, 1, sigmoid)))

function nnComputeRatio(system, oldPosition)
    oldWavefunctionValue = computePsi(system, oldPosition)
    newWavefunctionValue = computePsi(system, system.particles)
    ratio = exp(newWavefunctionValue*2 - oldWavefunctionValue*2)
    return ratio
end

system = systemNN(particlesNew, numParticles, numDimensions, false, nn)

# println(nnComputeRatio(system, particlesOld))
# println(nnComputeGradient(system))
# println(nnComputeLaplacian(system))
# println(nnComputeParameterGradient(system)[system.nn.model[1].W])

@time for i=1:numIterations
    (nnComputeRatio(system, particlesOld))
end

@time for i=1:numIterations
    nnComputeGradient(system)
end

@time for i=1:numIterations
    nnComputeLaplacian(system)
end

@time for i=1:numIterations
    nnComputeParameterGradient(system)
end

# println(system.nn.model[1].W)

struct systemNNAnalytical
    particles::Array{Float64, 2}
    numParticles::Int64 
    numDimensions::Int64

    interacting::Bool
    nn::NNAnalytical
end

nnAnalytical = initializeNNAnalytical(numParticles, numDimensions, numNodesLayer1, numNodesLayer2, sigmoid)

nnAnalytical.w1[:] = system.nn.model[1].W
nnAnalytical.b1[:] = system.nn.model[1].b

nnAnalytical.w2[:] = system.nn.model[2].W
nnAnalytical.b2[:] = system.nn.model[2].b

nnAnalytical.w3[:] = system.nn.model[3].W
nnAnalytical.b3[:] = system.nn.model[3].b

systemAnalytical = systemNNAnalytical(particlesNew, numParticles, numDimensions, false, nnAnalytical)

model = systemAnalytical.nn

particlesOld1 = vec(reshape(particlesOld', 1,:))
particlesNew1 = vec(reshape(particlesNew', 1,:))

println(" ")

# println(exp(2*wavefunctionValue(model, particlesNew1) - 2*wavefunctionValue(model, particlesOld1)))
# println(wavefunctionValue(model, particlesNew1))
# println(nnAnalyticalComputeGradient!(model))
# println(nnAnalyticalComputeLaplacian!(model))
# println(nnAnalyticalComputeParameterGradient!(model, particlesNew1))

println("Analytical:")

@time for i=1:numIterations
    particlesOld1 = vec(reshape(particlesOld', 1,:))
    particlesNew1 = vec(reshape(particlesNew', 1,:))

    (exp(2*wavefunctionValue(model, particlesNew1) - 2*wavefunctionValue(model, particlesOld1)))
end

@time for i=1:numIterations
    nnAnalyticalComputeGradient!(model)
end

@time for i=1:numIterations
    nnAnalyticalComputeLaplacian!(model)
end

@time for i=1:numIterations
    particlesNew1 = vec(reshape(particlesNew', 1,:))
    nnAnalyticalComputeParameterGradient!(model,particlesNew1)
end
#END MODULE
end