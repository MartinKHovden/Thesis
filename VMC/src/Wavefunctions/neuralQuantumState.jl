using Flux
using Zygote

struct NNQS
    model
end

function computePsi(nnqs, x)
    # nnqs = system.NNQS
    return nnqs.model(x)
end 

function testComputePsi()
    numDims = 3
    numParticles = 2
    numHidden = 10
    x = randn(numDims*numParticles)
    nn = NNQS(Chain(Dense(numParticles*numDims, numHidden), Dense(numHidden, 1)))
    println(computePsi(nn, x))
end

function nnComputeParameterGradient(nnqs, loss, x)
    println("Params", params(nnqs.model))
    grads = gradient(params(nnqs.model)) do 
        loss(x)
    end 
    return grads
end 

function testComputeParameterGradient()
    numDims = 3
    numParticles = 2
    numHidden = 100
    x = randn(numDims*numParticles)
    nn = NNQS(Chain(Dense(numParticles*numDims, numHidden), Dense(numHidden, 1)))
    loss(x) = sum(nn.model(x))
    println(nnComputeParameterGradient(nn, loss, x))
    return 0

end

function nnComputeGradient(nnqs, loss, x)
    # println(x)
    grads = gradient(Params([x])) do 
        loss(x)
    end 
    # println(grads.grads)
    return grads
end

function nnTestComputeGradient()
    numDims = 2
    numParticles = 1
    numHidden = 2
    x = randn(numDims*numParticles)
    nn = NNQS(Chain(Dense(numParticles*numDims, numHidden), Dense(numHidden, 1)))
    loss(x) = sum(nn.model(x))
    println("testComputeGradient ", nnComputeGradient(nn, loss, x)[x])
end

function nnComputeLaplacian(loss, x)
    return Zygote.hessian(loss, x)
end

function nnTestComputeLaplacian()
    numDims = 2
    numParticles = 3
    numHidden = 2
    x = randn(numDims*numParticles)
    nn = NNQS(Chain(Dense(numParticles*numDims, numHidden, sigmoid), Dense(numHidden, 1)))
    loss(x) = sum(nn.model(x))
    println("Laplcian = ", nnComputeLaplacian(loss, x))
end

function nnComputeParameterGradient()
    return 0
end 

# @time testComputePsi()
# @time testComputeParameterGradient()
@time nnTestComputeGradient()
@time nnTestComputeGradient()
@time nnTestComputeGradient()

@time nnTestComputeLaplacian()
@time nnTestComputeLaplacian()

