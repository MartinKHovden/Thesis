module neuralNetworkAnalytical

using LinearAlgebra

export computePsi, nnAnalyticalComputeRatio!, nnAnalyticalComputeGradient!, nnAnalyticalComputeLaplacian!, nnAnalyticalComputeParameterGradient!
export sigmoid, sigmoid_derivative, sigmoid_double_derivative, nnAnalyticalComputeDriftForce!
export nnAnalyticalComputePsi!

function computePsi!(system, position)
    model = system.nn
    x = vec(reshape(position', 1,:))
    mul!(model.z1, model.w1, x)
    model.z1[:] += model.b1
    map!(model.activationFunction, model.a1, model.z1)

    mul!(model.z2, model.w2, model.a1)
    model.z2[:] += model.b2
    map!(model.activationFunction, model.a2, model.z2)

    mul!(model.z3, model.w3, model.a2)
    model.z3[:] += model.b3
    # map!(model.activationFunction, model.a3, model.z3)
    map!(x->x, model.a3, model.z3)


    return model.a3[1]
end

function nnAnalyticalComputePsi!(system, position)
    model = system.nn
    x = vec(reshape(position', 1,:))
    mul!(model.z1, model.w1, x)
    model.z1[:] += model.b1
    map!(model.activationFunction, model.a1, model.z1)

    mul!(model.z2, model.w2, model.a1)
    model.z2[:] += model.b2
    map!(model.activationFunction, model.a2, model.z2)

    mul!(model.z3, model.w3, model.a2)
    model.z3[:] += model.b3
    # map!(model.activationFunction, model.a3, model.z3)
    map!(x -> x, model.a3, model.z3)


    return model.a3[1]
end

function nnAnalyticalComputeRatio!(system, oldPosition)
    oldWavefunctionValue = computePsi!(system, oldPosition)
    newWavefunctionValue = computePsi!(system, system.particles)
    ratio = exp(newWavefunctionValue*2 - oldWavefunctionValue*2)
    return ratio
end

function nnAnalyticalComputeDriftForce!(system, particleToUpdate, coordinateToUpdate)
    return 2*nnAnalyticalComputeGradient!(system)[(particleToUpdate - 1)*system.numDimensions + coordinateToUpdate]
end

function nnAnalyticalComputeGradient!(system)

    model = system.nn

    gradient = I

    mul!(model.a1_grad, model.w1, gradient)
    broadcast!(*, model.a1_grad, sigmoid_derivative.(model.z1), model.a1_grad)

    mul!(model.a2_grad, model.w2, model.a1_grad)
    broadcast!(*, model.a2_grad,sigmoid_derivative.(model.z2), model.a2_grad)

    mul!(model.a3_grad, model.w3, model.a2_grad)
    # broadcast!(*, model.a3_grad, sigmoid_derivative.(model.z3),model.a3_grad)
    broadcast!(*, model.a3_grad, 1.0, model.a3_grad)



    return model.a3_grad[:]
end

function nnAnalyticalComputeLaplacian!(system)
    model = system.nn

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
    # broadcast!(*, model.a3_double_grad_temp, sigmoid_derivative.(model.z3), model.a3_double_grad_temp)
    broadcast!(*, model.a3_double_grad_temp, 1.0, model.a3_double_grad_temp)


    mul!(model.a3_double_grad, model.w3, model.a2_grad)
    map!(x->x^2, model.a3_double_grad, model.a3_double_grad)
    # broadcast!(*, model.a3_double_grad, sigmoid_double_derivative.(model.z3), model.a3_double_grad)
    broadcast!(*, model.a3_double_grad, 0.0, model.a3_double_grad)


    broadcast!(+, model.a3_double_grad, model.a3_double_grad, model.a3_double_grad_temp)

    return_value = model.a3_double_grad

    # resetArrays!(system)

    return return_value
end

function nnAnalyticalComputeParameterGradient!(system)
    model = system.nn

    x = reshape(system.particles', 1,:)'

    # model.delta3[:] = sigmoid_derivative.(model.z3)

    # map!(sigmoid_derivative, model.delta3, model.z3)
    map!(x -> 1, model.delta3, model.z3)

    println(model.delta3)

    model.w3_grad[:,:] = model.delta3'.*model.a2

    # broadcast!(*, model.w3_grad, model.delta3', model.a2)
    model.b3_grad[:] = copy(model.delta3)

    # model.delta2[:] = (model.w3'*model.delta3).*sigmoid_derivative.(model.z2)
    mul!(model.delta2, model.w3', model.delta3)
    broadcast!(*, model.delta2, sigmoid_derivative.(model.z2), model.delta2)
    model.w2_grad[:,:] = model.delta2'.*model.a1
    model.b2_grad[:] = copy(model.delta2)

    # model.delta1[:] = (model.w2'*model.delta2).*sigmoid_derivative.(model.z1)
    mul!(model.delta1, model.w2', model.delta2)
    broadcast!(*, model.delta1, sigmoid_derivative.(model.z1), model.delta1)
    model.w1_grad[:,:] = model.delta1'.*x
    model.b1_grad[:] = copy(model.delta1)

    return_values = copy(model.w1_grad), copy(model.w2_grad), copy(model.w3_grad), copy(model.b1_grad), copy(model.b2_grad), copy(model.b3_grad)
    # println(size(model.w3_grad))
    resetArrays!(system)
    return return_values
end

function resetArrays!(system)
    model = system.nn

    fill!(model.a1, zero(Float64))
    fill!(model.a2, zero(Float64))
    fill!(model.a3, zero(Float64))

    fill!(model.z1, zero(Float64))
    fill!(model.z2, zero(Float64))
    fill!(model.z3, zero(Float64))

    fill!(model.a1_grad, zero(Float64))
    fill!(model.a2_grad, zero(Float64))
    fill!(model.a3_grad, zero(Float64))

    fill!(model.a1_double_grad, zero(Float64))
    fill!(model.a2_double_grad, zero(Float64))
    fill!(model.a3_double_grad, zero(Float64))

    fill!(model.a1_double_grad_temp, zero(Float64))
    fill!(model.a2_double_grad_temp, zero(Float64))
    fill!(model.a3_double_grad_temp, zero(Float64))

    fill!(model.w1_grad, zero(Float64))
    fill!(model.w2_grad, zero(Float64))
    fill!(model.w3_grad, zero(Float64))

    fill!(model.b1_grad, zero(Float64))
    fill!(model.b2_grad, zero(Float64))
    fill!(model.b3_grad, zero(Float64))

    fill!(model.delta1, zero(Float64))
    fill!(model.delta2, zero(Float64))
    fill!(model.delta3, zero(Float64))

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
#END MODULE
end 