using Flux
using Zygote

struct NNQS
    model
end

# # W = rand(2,2)
# # b = rand(2)

# # nn = NNQS(W, b)

# x = rand(10)

# # y( x) = sum(W*x .+ b)

# # g = gradient(()->y(x), params([W,b]))

# # println("Grads: ", g[W], g[b])

# model = Chain(Dense(10, 5), Dense(5, 1))

# println("Output: ", model(x))
# loss(x) = sum(model(x))

# grads = gradient(params(model)) do 
#     loss(x)
# end

# for p in params(model)
#     println(grads[p])
# end 

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

function nnComputeLaplacian()
    # println("Hessian = ", Zygote.hessian(loss, x))
    return 0
end

function nnComputeParameterGradient()
    return 0
end 

# @time testComputePsi()
# @time testComputeParameterGradient()
@time nnTestComputeGradient()
@time nnTestComputeGradient()
@time nnTestComputeGradient()

# @time nnTestComputeLaplacian()
# @time nnTestComputeLaplacian()

