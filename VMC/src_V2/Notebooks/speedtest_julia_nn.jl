module speedtest_julia_nn

include("../Wavefunctions/wavefunction.jl")

include("../Wavefunctions/nn.jl")
include("../Wavefunctions/slater.jl")
include("../system.jl")


using Flux
using Zygote:hessian
using LinearAlgebra
using .wavefunction
using .slater
using .nn
using .system



function nnComputeGradient(model, x)
    loss(m, x) = sum(m(x))
    return gradient(loss, model, x)[2]
end

function nnComputeLaplacian(model, x)
    loss(x) = sum(model(x))
    return diag(hessian(loss, x))
end

numParticles = 5
numDimensions = 2

firstHidden = 20
secondHidden = 20
modelInput = randn(numParticles*numDimensions)

numIterations = 10000

nnmodel = (Chain(Dense(numParticles*numDimensions, firstHidden, sigmoid), Dense(firstHidden, secondHidden, sigmoid), Dense(secondHidden, 1, sigmoid)))

@time for i=1:numIterations
    nnComputeGradient(nnmodel, modelInput)
end

@time for i=1:numIterations
    nnComputeLaplacian(nnmodel, modelInput)
end

s = System(numParticles, numDimensions, "quantumDot")
addWaveFunctionElement(s, NN(s, firstHidden, secondHidden, "sigmoid"))

@time for i=1:numIterations
    computePsi!(s.wavefunctionElements[1], modelInput)
    computeGradient(s, s.wavefunctionElements[1])
end

@time for i=1:numIterations
    computeLaplacian(s, s.wavefunctionElements[1])
end
end