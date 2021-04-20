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

function nnComputeGradient(nnqs, loss, x)
    grads = gradient(params(nnqs.model)) do 
        loss(x)
    end 
    return grads
end 

function testComputeGradient()
    numDims = 3
    numParticles = 2
    numHidden = 10
    x = randn(numDims*numParticles)
    nn = NNQS(Chain(Dense(numParticles*numDims, numHidden), Dense(numHidden, 1)))
    loss(x) = sum(nn.model(x))
    println(nnComputeGradient(nn, loss, x))

end

function nnComputeLaplacian()
    return 0
end

function nnComputeParameterGradient()
    return 0
end 

testComputePsi()
testComputeGradient()