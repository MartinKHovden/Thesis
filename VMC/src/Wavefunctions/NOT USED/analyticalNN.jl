module analyticalNN

export NNAnalytical, nnAnalyticalComputeGradient!, nnAnalyticalComputeLaplacian!
export initializeNNAnalytical, wavefunctionValue, nnAnalyticalComputeParameterGradient!

using StaticArrays
using LinearAlgebra

struct NNAnalytical 
    w1#::MMatrix 
    w2#::MMatrix
    w3#::MMatrix

    b1#::MVector
    b2#::MVector 
    b3#::MVector

    a1#::MVector 
    a2#::MVector 
    a3#::MVector 

    z1#::MVector 
    z2#::MVector 
    z3#::MVector

    a1_grad#::MMatrix
    a2_grad#::MMatrix 
    a3_grad#::MMatrix

    a1_double_grad#::MMatrix 
    a2_double_grad#::MMatrix 
    a3_double_grad#::MMatrix

    a1_double_grad_temp#::MMatrix
    a2_double_grad_temp#::MMatrix
    a3_double_grad_temp#::MMatrix

    w1_grad#::MMatrix 
    w2_grad#::MMatrix 
    w3_grad#::MMatrix 

    b1_grad#::MMatrix 
    b2_grad#::MMatrix
    b3_grad#::MMatrix

    delta1#::MVector 
    delta2#::MVector
    delta3#::MVector

    activationFunction
end

function initializeNNAnalytical(numParticles, numDimensions, numNodesLayer1, numNodesLayer2, activationFunction)
    w1 = @MMatrix randn(numNodesLayer1, numParticles*numDimensions)
    w2 = @MMatrix randn(numNodesLayer2, numNodesLayer1)
    w3 = @MMatrix randn(1, numNodesLayer2)

    b1 = @MVector randn(numNodesLayer1)
    b2 = @MVector randn(numNodesLayer2)
    b3 = @MVector randn(1)

    a1 = @MVector zeros(numNodesLayer1)
    a2 = @MVector zeros(numNodesLayer2)
    a3 = @MVector zeros(1)

    z1 = @MVector zeros(numNodesLayer1)
    z2 = @MVector zeros(numNodesLayer2)
    z3 = @MVector zeros(1)

    a1_grad = @MMatrix zeros(numNodesLayer1, numParticles*numDimensions)
    a2_grad = @MMatrix zeros(numNodesLayer2, numParticles*numDimensions)
    a3_grad = @MMatrix zeros(1, numParticles*numDimensions)

    a1_double_grad = @MMatrix zeros(numNodesLayer1, numParticles*numDimensions)
    a2_double_grad = @MMatrix zeros(numNodesLayer2, numParticles*numDimensions)
    a3_double_grad = @MMatrix zeros(1, numParticles*numDimensions)

    a1_double_grad_temp = @MMatrix zeros(numNodesLayer1, numParticles*numDimensions)
    a2_double_grad_temp = @MMatrix zeros(numNodesLayer2, numParticles*numDimensions)
    a3_double_grad_temp = @MMatrix zeros(1, numParticles*numDimensions)

    w1_grad = @MMatrix randn(numNodesLayer1, numParticles*numDimensions) 
    w2_grad = @MMatrix randn(numNodesLayer2, numNodesLayer1) 
    w3_grad = @MMatrix randn(1, numNodesLayer2) 

    b1_grad = @MVector randn(numNodesLayer1)
    b2_grad = @MVector randn(numNodesLayer2)
    b3_grad = @MVector randn(1)

    delta1 = @MVector zeros(numNodesLayer1)
    delta2 = @MVector zeros(numNodesLayer2)
    delta3 = @MVector zeros(1)

    return NNAnalytical(w1, w2, w3, b1, b2, b3, a1, a2, a3, z1, z2, z3, 
                            a1_grad, a2_grad, a3_grad, 
                            a1_double_grad, a2_double_grad, a3_double_grad, 
                            a1_double_grad_temp, a2_double_grad_temp, a3_double_grad_temp,
                            w1_grad, w2_grad, w3_grad,
                            b1_grad, b2_grad, b3_grad,
                            delta1, delta2, delta3,
                            activationFunction)
end


function wavefunctionValue(model::NNAnalytical, x)
    mul!(model.z1, model.w1, x)
    model.z1[:] .+= model.b1
    map!(model.activationFunction, model.a1, model.z1)

    mul!(model.z2, model.w2, model.a1)
    model.z2[:] .+= model.b2
    map!(model.activationFunction, model.a2, model.z2)

    mul!(model.z3, model.w3, model.a2)
    model.z3[:] .+= model.b3
    map!(model.activationFunction, model.a3, model.z3)

    return model.a3[1]
end

# function computePsi()

# function nnAnalyticalComputeRatio(system, oldPosition)

function nnAnalyticalComputeGradient!(model::NNAnalytical)
    gradient = I

    mul!(model.a1_grad, model.w1, gradient)
    broadcast!(*, model.a1_grad, sigmoid_derivative.(model.z1), model.a1_grad)

    mul!(model.a2_grad, model.w2, model.a1_grad)
    broadcast!(*, model.a2_grad,sigmoid_derivative.(model.z2), model.a2_grad)

    mul!(model.a3_grad, model.w3, model.a2_grad)
    broadcast!(*, model.a3_grad, sigmoid_derivative.(model.z3),model.a3_grad)

    return model.a3_grad[:]
end

function nnAnalyticalComputeLaplacian!(model::NNAnalytical)
    gradient = I

    mul!(model.a1_double_grad, model.w1, gradient)
    map!(x->x^2, model.a1_double_grad, model.a1_double_grad)
    broadcast!(*, model.a1_double_grad, sigmoid_double_derivative.(model.z1), model.a1_double_grad)

    mul!(model.a2_double_grad_temp, model.w2, model.a1_double_grad)
    broadcast!(*, model.a2_double_grad_temp, sigmoid_derivative.(model.z2), model.a2_double_grad_temp)

    mul!(model.a2_double_grad, model.w2, model.a1_grad)
    map!(x->x^2, model.a2_double_grad, model.a2_double_grad)
    broadcast!(*, model.a2_double_grad, sigmoid_double_derivative.(model.z2), model.a2_double_grad)

    broadcast!(+, model.a2_double_grad, model.a2_double_grad, model.a2_double_grad_temp)

    mul!(model.a3_double_grad_temp, model.w3, model.a2_double_grad)
    broadcast!(*, model.a3_double_grad_temp, sigmoid_derivative.(model.z3), model.a3_double_grad_temp)

    mul!(model.a3_double_grad, model.w3, model.a2_grad)
    map!(x->x^2, model.a3_double_grad, model.a3_double_grad)
    broadcast!(*, model.a3_double_grad, sigmoid_double_derivative.(model.z3), model.a3_double_grad)

    broadcast!(+, model.a3_double_grad, model.a3_double_grad, model.a3_double_grad_temp)

    return model.a3_double_grad
end

function nnAnalyticalComputeParameterGradient!(model, particles)
    # model.delta3[:] = sigmoid_derivative.(model.z3)
    map!(sigmoid_derivative, model.delta3, model.z3)
    model.w3_grad[:] = model.delta3'.*model.a2
    # broadcast!(*, model.w3_grad, model.delta3', model.a2)
    model.b3_grad[:] = model.delta3

    # model.delta2[:] = (model.w3'*model.delta3).*sigmoid_derivative.(model.z2)
    mul!(model.delta2, model.w3', model.delta3)
    broadcast!(*, model.delta2, sigmoid_derivative.(model.z2), model.delta2)
    model.w2_grad[:] = model.delta2'.*model.a1
    model.b2_grad[:] = model.delta2

    # model.delta1[:] = (model.w2'*model.delta2).*sigmoid_derivative.(model.z1)
    mul!(model.delta1, model.w2', model.delta2)
    broadcast!(*, model.delta1, sigmoid_derivative.(model.z1), model.delta1)
    model.w1_grad[:] = model.delta1'.*particles
    model.b1_grad[:] = model.delta1

    return model.w1_grad, model.w2_grad, model.w3_grad, model.b1_grad, model.b2_grad, model.b3_grad
end

function sigmoid(z)
    return 1/(1 + exp(-z))
end

function sigmoid_derivative(z)
    sig = sigmoid(z)
    return sig*(1-sig)
end

function sigmoid_double_derivative(z)
    sig = sigmoid(z)
    return sig*(1-sig)*(1-2*sig)
end

nn = initializeNNAnalytical(4, 2, 5, 5, sigmoid)
# println(wavefunctionValue(nn,  [1,2,3,4,5,6,7,8]))
nnAnalyticalComputeParameterGradient!(nn,[1,2,3,4,5,6,7,8])
# @time for i=1:100000
#     wavefunctionValue(nn, [1,2,3,4,5,6,7,8])
#     # println(nn.z1)
# end

# @time for i=1:100000
#     nnAnalyticalComputeGradient!(nn)
# end

# @time for i=1:100000
#     nnAnalyticalComputeLaplacian!(nn)
# end

@time for i=1:100000
    nnAnalyticalComputeParameterGradient!(nn, [1,2,3,4,5,6,7,8])
end
#END MODULE
end