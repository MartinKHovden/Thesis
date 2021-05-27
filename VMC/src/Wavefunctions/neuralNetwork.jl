module neuralNetwork

using Flux:Chain, Dense, Params, gradient, params
using Zygote:hessian
using LinearAlgebra

export computePsi, nnComputeRatio, nnComputeGradient, nnComputeLaplacian, nnComputeParameterGradient

function computePsi(system, position)
    nn = system.nn
    x = reshape(position', 1,:)'
    return nn.model(x)[1]
end

function nnComputeRatio(system, oldPosition)
    oldWavefunctionValue = computePsi(system, oldPosition)
    newWavefunctionValue = computePsi(system, system.particles)
    ratio = (newWavefunctionValue^2)/(oldWavefunctionValue^2)
    return ratio
end

function nnComputeGradient(system)
    numDimensions = system.numDimensions
    nn = system.nn
    x = reshape(system.particles', 1,:)'
    loss(x) = sum(nn.model(x))
    grads = gradient(Params([x])) do 
        loss(x)
    end 
    return grads[x]
end

function nnComputeLaplacian(system)
    x = reshape(system.particles', 1,:)'
    loss(x) = sum(system.nn.model(x))
    return diag(hessian(loss, x))
end

function nnComputeParameterGradient(system)
    nn = system.nn
    ps = params(system.nn.model)
    x = reshape(system.particles', 1,:)'
    loss(x) = sum(nn.model(x))
    grads = gradient(ps) do 
        loss(x)
    end 
    return grads
end 

function nnTestComputeGradient()
    numDims = 2
    numParticles = 1
    numHidden = 2
    x = randn(numDims*numParticles)
    nn = NN(Chain(Dense(numParticles*numDims, numHidden), Dense(numHidden, 1)))
    loss(x) = sum(nn.model(x))
    println("testComputeGradient ", nnComputeGradient(nn, loss, x)[x])
end

function nnTestComputeLaplacian()
    numDims = 2
    numParticles = 3
    numHidden = 2
    x = randn(numDims*numParticles)
    nn = NN(Chain(Dense(numParticles*numDims, numHidden, sigmoid), Dense(numHidden, 1)))
    loss(x) = sum(nn.model(x))
    println("Laplcian = ", nnComputeLaplacian(loss, x))
end

function testComputeParameterGradient()
    numDims = 3
    numParticles = 2
    numHidden = 100
    x = randn(numDims*numParticles)
    nn = NN(Chain(Dense(numParticles*numDims, numHidden), Dense(numHidden, 1)))
    loss(x) = sum(nn.model(x))
    println(nnComputeParameterGradient(nn, loss, x))
    return 0

end

function testComputePsi()
    numDims = 3
    numParticles = 2
    numHidden = 10
    x = randn(numDims*numParticles)
    nn = NN(Chain(Dense(numParticles*numDims, numHidden), Dense(numHidden, 1)))
    println(computePsi(nn, x))
end

# @time testComputePsi()
# @time testComputeParameterGradient()
# @time nnTestComputeGradient()
# @time nnTestComputeGradient()
# @time nnTestComputeGradient()

# @time nnTestComputeLaplacian()
# @time nnTestComputeLaplacian()

end
