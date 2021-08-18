module nn

using Random
using StaticArrays
using LinearAlgebra
using Flux:Chain, Dense, Params, gradient, params, sigmoid, relu
using Zygote:hessian

struct nNet
    weights::Array{SMatrix}
    biases::Array{SArray} 
    cacheZ::Array{SArray}
    cacheWADx::Array{SMatrix}
    cacheTanhZ::Array{SArray}
end

function initializeModel(numLayerNodes)
    numLayers = length(numLayerNodes)
    model = nNet([], [], [], [], [])
    for i=2:numLayers
        push!(model.weights, @SMatrix randn(numLayerNodes[i], numLayerNodes[i-1]))
        push!(model.biases, @SArray randn(numLayerNodes[i]))
        push!(model.cacheZ, @SArray zeros(numLayerNodes[i]))
    end
    return model
end

inputSize = 8
firstHidden = 10
secondHidden = 10
numIterations = 100000

model = initializeModel([inputSize, firstHidden, secondHidden, 1])
# println(model.weights)
# println(model.biases)

#Fast
function nnComputeWaveFunction(model::nNet, input::Array)
    empty!(model.cacheZ)
    z = input
    for i=1:length(model.weights)
        a = model.weights[i]*z + model.biases[i]
        z = tanh.(a)
        push!(model.cacheZ, z)
    end 
    return z
end 

function tanhDx(x)
    return -tanh.(x) .+ 1
end

function tanhDxDx(x)
    return -2*tanh.(x).*( - tanh.(x) .+1)
end

input = randn(inputSize)
# @time nnComputeWaveFunction(model, input)
# @time nnComputeWaveFunction(model, input)
# @time nnComputeWaveFunction(model, input)

function nnComputeGradient(model)
    numLayers = length(model.weights)
    empty!(model.cacheWADx)
    empty!(model.cacheTanhZ)
    dadx = I
    for i=1:numLayers 
        wdadx = model.weights[i]*dadx
        tanhZ =  tanhDx(model.cacheZ[i])
        dadx = tanhZ.*(wdadx)
        push!(model.cacheTanhZ, tanhZ)
        push!(model.cacheWADx, wdadx)
    end 
    return dadx
end




function nnComputeLaplacian1(model)
    numLayers = length(model.weights)
    dadxdx = tanhDx(model.cacheZ[1]).*(model.weights[1]*zeros(inputSize,inputSize)) + tanhDxDx(model.cacheZ[1]).*((model.weights[1]*I).^2)
    for i=2:numLayers 
        dadxdx = model.cacheTanhZ[i].*(model.weights[i]*dadxdx) + tanhDxDx(model.cacheZ[i]).*((model.cacheWADx[i].^2))
    end 
    return dadxdx
end 


println("Analytical gradient: ")
@time for i=1:numIterations
    nnComputeGradient(model)
end

println("Analytical laplacian: ")
@time for i=1:numIterations
    nnComputeLaplacian1(model)
end
# println("STOP")

# function forwardStep(W, x, b)
#     z = W*x .+ b
# end

# # function applyActivation(x, )


# function forward(input, modelParams)
#     output = input
#     for layer::Int64=1:(length(modelParams)/2)
#         @time output = forwardStep(modelParams[string("W_layer_", layer)], output, modelParams[string("b_layer_", layer)] )
#     end
#     # println(output)
#     return output
# end

# model = initializeModel([10, 5, 1])

input = randn(inputSize)

# @time forward(input, model.parameters )

# function forward(input, modelParamsW, modelParamsB, numLayers)
#     output = input
#     for layer::Int64=1:numLayers
#         output = forwardStep( modelParamsW[layer], output, modelParamsB[layer])
#     end
#     return output
# end

# mPW = @SArray [randn(5, 10), randn(1,5)]
# mPB = @SArray [randn(5), randn(1)]

# @time forward(input, mPW, mPB, 2)

n = (Chain(Dense(inputSize, firstHidden, sigmoid), Dense(firstHidden, secondHidden, sigmoid), Dense(secondHidden, 1, sigmoid)))

# @time n(input)
# @time n(input)
# @time n(input)


# function nnComputeGradient(model, x)
#     loss(x) = sum(model(x))
#     grads = gradient(Params([x])) do 
#         loss(x)
#     end 
#     return grads[x]
# end

function nnComputeGradient(model, x)
    loss(m, x) = sum(m(x))
    return gradient(loss, model, x)[2]
end

function nnComputeLaplacian(model, x)
    loss(x) = sum(model(x))
    return diag(hessian(loss, x))
end

# function nnComputeLaplacian(model, x)
println("Autograd gradient")

# @time println("Grad2 = ", nnComputeGradient2(n, input))
@time for i=1:100000
    nnComputeGradient(n, input)
end 

println("Autograd laplacian")
@time for i=1:numIterations
    nnComputeLaplacian(n, input)
end


# @time println(nnComputeLaplacian(n, input))


end