module nn 

export NN, computeRatio, computeGradient, computeLaplacian, computeParameterGradient, computePsi!

using StaticArrays
using LinearAlgebra
using ..wavefunction

""" 
    NN 

Struct for storing the information about, and the state of, the neural network. 
The network is restricted to two hidden layers, however, the number of nodes and 
the activation function can be changed as wanted. 

"""
mutable struct NN
    variationalParameter
    variationalParameterGradient

    a 
    z 

    aGrad 
    aDoubleGrad 
    aDoubleGradTemp

    delta

    activationFunction 
    activationFunctionDerivative 
    activationFunctionDoubleDerivative
    activationFunctionName

    localEnergyPsiParameterDerivativeSum 
    psiParameterDerivativeSum


    function NN(system, 
                numNodesLayer1::Int64, 
                numNodesLayer2::Int64, 
                activationFunction::String)
        numParticles = system.numParticles
        numDimensions = system.numDimensions

        # Initialization of the weights
        w1 = randn(numNodesLayer1, numParticles*numDimensions)
        w1 *= 0.01
        w2 = randn(numNodesLayer2, numNodesLayer1)
        w2*= 0.01
        w3 = randn(1, numNodesLayer2)
        w3*= 0.01
    
        # Initialization of the biases
        b1 = randn(numNodesLayer1)
        b2 = randn(numNodesLayer2)
        b3 = randn(1)

        # Placing the weights and biases in an Array. Alternating between w and b. 
        params = []
        push!(params, w1)
        push!(params, b1)
        push!(params, w2)
        push!(params, b2)
        push!(params, w3)
        push!(params, b3)


        w1_grad = randn(numNodesLayer1, numParticles*numDimensions) 
        w2_grad = randn(numNodesLayer2, numNodesLayer1) 
        w3_grad = randn(1, numNodesLayer2) 
    
        b1_grad = randn(numNodesLayer1)
        b2_grad = randn(numNodesLayer2)
        b3_grad = randn(1)

        paramGrads = []
        push!(paramGrads, w1_grad)
        push!(paramGrads, b1_grad)
        push!(paramGrads, w2_grad)
        push!(paramGrads, b2_grad)
        push!(paramGrads, w3_grad)
        push!(paramGrads, b3_grad)


        a1 = zeros(numNodesLayer1)
        a2 = zeros(numNodesLayer2)
        a3 = zeros(1)

        a = []
        push!(a, a1)
        push!(a, a2)
        push!(a, a3)

        z1 = zeros(numNodesLayer1)
        z2 = zeros(numNodesLayer2)
        z3 = zeros(1)

        z = []
        push!(z, z1)
        push!(z, z2)
        push!(z, z3)
    
        a1_grad = zeros(numNodesLayer1, numParticles*numDimensions)
        a2_grad = zeros(numNodesLayer2, numParticles*numDimensions)
        a3_grad = zeros(1, numParticles*numDimensions)

        aGrad = []
        push!(aGrad, a1_grad)
        push!(aGrad, a2_grad)
        push!(aGrad, a3_grad)

        a1_double_grad = zeros(numNodesLayer1, numParticles*numDimensions)
        a2_double_grad = zeros(numNodesLayer2, numParticles*numDimensions)
        a3_double_grad = zeros(1, numParticles*numDimensions)

        aDoubleGrad = []
        push!(aDoubleGrad, a1_double_grad)
        push!(aDoubleGrad, a2_double_grad)
        push!(aDoubleGrad, a3_double_grad)

        a1_double_grad_temp = zeros(numNodesLayer1, numParticles*numDimensions)
        a2_double_grad_temp = zeros(numNodesLayer2, numParticles*numDimensions)
        a3_double_grad_temp = zeros(1, numParticles*numDimensions)

        aDoubleGradTemp = []
        push!(aDoubleGradTemp, a1_double_grad_temp)
        push!(aDoubleGradTemp, a2_double_grad_temp)
        push!(aDoubleGradTemp, a3_double_grad_temp)

        delta1 = zeros(numNodesLayer1)
        delta2 = zeros(numNodesLayer2)
        delta3 = zeros(1)

        delta = []
        push!(delta, delta1)
        push!(delta, delta2)
        push!(delta, delta3)

        aFunction, aFunctionDerivative, aFunctionDoubleDerivative = getActivationFunctions(activationFunction)

        return new(params, 
                paramGrads, 
                a, 
                z, 
                aGrad, 
                aDoubleGrad, 
                aDoubleGradTemp, 
                delta, 
                aFunction, 
                aFunctionDerivative, 
                aFunctionDoubleDerivative, 
                activationFunction,
                0 .*paramGrads,
                0 .*paramGrads)
    end
end 

function wavefunction.computeRatio(system, 
                                wavefunctionElement::NN, 
                                particleToUpdate, 
                                coordinateToUpdate, 
                                oldPosition)
    return nnAnalyticalComputeRatio!(system, wavefunctionElement, oldPosition)
end

function computePsi!(model::NN, position)
    x = vec(reshape(position', 1,:))

    mul!(model.z[1], model.variationalParameter[1], x)
    model.z[1][:] += model.variationalParameter[2]
    map!(model.activationFunction, model.a[1], model.z[1])

    mul!(model.z[2], model.variationalParameter[3], model.a[1])
    model.z[2][:] += model.variationalParameter[4]
    map!(model.activationFunction, model.a[2], model.z[2])

    mul!(model.z[3], model.variationalParameter[5], model.a[2])
    model.z[3][:] += model.variationalParameter[6]
    # map!(model.activationFunction, model.a3, model.z3)
    map!(x->x, model.a[3], model.z[3])

    return model.a[3][1]
end

function nnAnalyticalComputeRatio!(system, model::NN, oldPosition)
    oldWavefunctionValue = computePsi!(model, oldPosition)
    newWavefunctionValue = computePsi!(model, system.particles)
    ratio = exp(newWavefunctionValue*2 - oldWavefunctionValue*2)
    return ratio
end

function wavefunction.computeGradient(system, wavefunctionElement::NN)
    return nnAnalyticalComputeGradient!(system, wavefunctionElement)
end

function nnAnalyticalComputeGradient!(system, model::NN)

    gradient = I

    mul!(model.aGrad[1], model.variationalParameter[1], gradient)
    broadcast!(*, model.aGrad[1], model.activationFunctionDerivative.(model.z[1]), model.aGrad[1])

    mul!(model.aGrad[2], model.variationalParameter[3], model.aGrad[1])
    broadcast!(*, model.aGrad[2],model.activationFunctionDerivative.(model.z[2]), model.aGrad[2])

    mul!(model.aGrad[3], model.variationalParameter[5], model.aGrad[2])
    # broadcast!(*, model.a3_grad, sigmoid_derivative.(model.z3),model.a3_grad)
    broadcast!(*, model.aGrad[3], 1.0, model.aGrad[3])

    return model.aGrad[3][:]
end

function wavefunction.computeLaplacian(system, wavefunctionElement::NN)
    return nnAnalyticalComputeLaplacian!(wavefunctionElement)
end 

function nnAnalyticalComputeLaplacian!(model::NN)
    gradient = I

    mul!(model.aDoubleGrad[1], model.variationalParameter[1], gradient)
    map!(x->x^2, model.aDoubleGrad[1], model.aDoubleGrad[1])
    broadcast!(*, model.aDoubleGrad[1], model.activationFunctionDoubleDerivative.(model.z[1]), model.aDoubleGrad[1])

    mul!(model.aDoubleGradTemp[2], model.variationalParameter[3], model.aDoubleGrad[1])
    broadcast!(*, model.aDoubleGradTemp[2], model.activationFunctionDerivative.(model.z[2]), model.aDoubleGradTemp[2])

    mul!(model.aDoubleGrad[2], model.variationalParameter[3], model.aGrad[1])
    map!(x->x^2, model.aDoubleGrad[2], model.aDoubleGrad[2])
    broadcast!(*, model.aDoubleGrad[2], model.activationFunctionDoubleDerivative.(model.z[2]), model.aDoubleGrad[2])

    broadcast!(+, model.aDoubleGrad[2], model.aDoubleGrad[2], model.aDoubleGradTemp[2])

    mul!(model.aDoubleGradTemp[3], model.variationalParameter[5], model.aDoubleGrad[2])
    # broadcast!(*, model.a3_double_grad_temp, sigmoid_derivative.(model.z3), model.a3_double_grad_temp)
    broadcast!(*, model.aDoubleGradTemp[3], 1.0, model.aDoubleGradTemp[3])


    mul!(model.aDoubleGrad[3], model.variationalParameter[5], model.aGrad[2])
    map!(x->x^2, model.aDoubleGrad[3], model.aDoubleGrad[3])
    # broadcast!(*, model.a3_double_grad, sigmoid_double_derivative.(model.z3), model.a3_double_grad)
    broadcast!(*, model.aDoubleGrad[3], 0.0, model.aDoubleGrad[3])


    broadcast!(+, model.aDoubleGrad[3], model.aDoubleGrad[3], model.aDoubleGradTemp[3])

    return_value = model.aDoubleGrad[3]

    # resetArrays!(system)

    return sum(return_value)
end

function wavefunction.computeParameterGradient(system, wavefunctionElement::NN)
    return_value = nnAnalyticalComputeParameterGradient!(system, wavefunctionElement)
    return return_value
end

function nnAnalyticalComputeParameterGradient!(system, model::NN)
    x = reshape(system.particles', 1,:)'

    map!(x -> 1, model.delta[3], model.z[3])

    # model.variationalParameterGradient[5][:,:] = copy(model.delta[3]'.*model.a[2])
    copy!(model.variationalParameterGradient[5], reshape(model.delta[3]'.*model.a[2], size(model.variationalParameterGradient[5])))


    # model.variationalParameterGradient[6][:] = copy(model.delta[3])
    copy!(model.variationalParameterGradient[6], model.delta[3])

    mul!(model.delta[2], model.variationalParameter[5]', model.delta[3])
    broadcast!(*, model.delta[2], model.activationFunctionDerivative.(model.z[2]), model.delta[2])

    # model.variationalParameterGradient[3][:,:] = copy(model.delta[2].*(model.a[1]'))
    copy!(model.variationalParameterGradient[3], model.delta[2].*(model.a[1]'))
    # model.variationalParameterGradient[4][:] = copy(model.delta[2])
    copy!(model.variationalParameterGradient[4], model.delta[2])


    mul!(model.delta[1], model.variationalParameter[3]', model.delta[2])

    broadcast!(*, model.delta[1], model.activationFunctionDerivative.(model.z[1]), model.delta[1])

    # model.variationalParameterGradient[1][:,:] = copy(model.delta[1].*(x'))
    copy!(model.variationalParameterGradient[1], model.delta[1].*(x'))
    # model.variationalParameterGradient[2][:] = copy(model.delta[1])
    copy!(model.variationalParameterGradient[2], model.delta[1])

    returnValues = [copy(model.variationalParameterGradient[1]), 
                copy(model.variationalParameterGradient[2]), 
                copy(model.variationalParameterGradient[3]), 
                copy(model.variationalParameterGradient[4]), 
                copy(model.variationalParameterGradient[5]), 
                copy(model.variationalParameterGradient[6])]

    resetArrays!(model)

    return returnValues
end

# function nnAnalyticalComputeParameterGradient!(system, model::NN)
#     x = reshape(system.particles', 1,:)'

#     map!(x -> 1, model.delta[3], model.z[3])

#     # model.variationalParameterGradient[5][:,:] = copy(model.delta[3]'.*model.a[2])
#     copyto!(model.variationalParameterGradient[5][:,:], model.delta[3]'.*model.a[2])

#     # model.variationalParameterGradient[6][:] = copy(model.delta[3])
#     copyto!(model.variationalParameterGradient[6][:], model.delta[3])

#     mul!(model.delta[2], model.variationalParameter[5]', model.delta[3])
#     broadcast!(*, model.delta[2], model.activationFunctionDerivative.(model.z[2]), model.delta[2])

#     # model.variationalParameterGradient[3][:,:] = copy(model.delta[2].*(model.a[1]'))
#     copyto!(model.variationalParameterGradient[3][:,:], model.delta[2].*(model.a[1]'))
#     # model.variationalParameterGradient[4][:] = copy(model.delta[2])
#     copyto!(model.variationalParameterGradient[4][:], model.delta[2])


#     mul!(model.delta[1], model.variationalParameter[3]', model.delta[2])

#     broadcast!(*, model.delta[1], model.activationFunctionDerivative.(model.z[1]), model.delta[1])

#     # model.variationalParameterGradient[1][:,:] = copy(model.delta[1].*(x'))
#     copyto!(model.variationalParameterGradient[1][:,:], model.delta[1].*(x'))
#     # model.variationalParameterGradient[2][:] = copy(model.delta[1])
#     copyto!(model.variationalParameterGradient[2][:], model.delta[1])

#     returnValues = [copy(model.variationalParameterGradient[1]), 
#                 copy(model.variationalParameterGradient[2]), 
#                 copy(model.variationalParameterGradient[3]), 
#                 copy(model.variationalParameterGradient[4]), 
#                 copy(model.variationalParameterGradient[5]), 
#                 copy(model.variationalParameterGradient[6])]

#     resetArrays!(model)

#     return returnValues
# end

function wavefunction.computeDriftForce(system, element::NN, particleToUpdate, coordinateToUpdate)
    return 2*nnAnalyticalComputeGradient!(system, element)[(particleToUpdate - 1)*system.numDimensions + coordinateToUpdate]
end

function wavefunction.computeDriftForceFull(system, element::NN, particleToUpdate)
    return 2*nnAnalyticalComputeGradient!(system, element)#[(particleToUpdate - 1)*system.numDimensions + 1: (particleToUpdate - 1)*system.numDimensions + system.numDimensions]
end

function wavefunction.updateElement!(system, wavefunctionElement::NN, particle::Int64)
end

function resetArrays!(model::NN)
    fill!(model.a[1], zero(Float64))
    fill!(model.a[2], zero(Float64))
    fill!(model.a[3], zero(Float64))

    fill!(model.z[1], zero(Float64))
    fill!(model.z[2], zero(Float64))
    fill!(model.z[3], zero(Float64))

    fill!(model.aGrad[1], zero(Float64))
    fill!(model.aGrad[2], zero(Float64))
    fill!(model.aGrad[3], zero(Float64))

    fill!(model.aDoubleGrad[1], zero(Float64))
    fill!(model.aDoubleGrad[2], zero(Float64))
    fill!(model.aDoubleGrad[3], zero(Float64))

    fill!(model.aDoubleGradTemp[1], zero(Float64))
    fill!(model.aDoubleGradTemp[2], zero(Float64))
    fill!(model.aDoubleGradTemp[3], zero(Float64))

    fill!(model.variationalParameterGradient[1], zero(Float64))
    fill!(model.variationalParameterGradient[3], zero(Float64))
    fill!(model.variationalParameterGradient[5], zero(Float64))

    fill!(model.variationalParameterGradient[2], zero(Float64))
    fill!(model.variationalParameterGradient[4], zero(Float64))
    fill!(model.variationalParameterGradient[6], zero(Float64))

    fill!(model.delta[1], zero(Float64))
    fill!(model.delta[2], zero(Float64))
    fill!(model.delta[3], zero(Float64))
end

function getActivationFunctions(activationFunction::String)
    if activationFunction == "sigmoid"
        return sigmoid, sigmoidDerivative, sigmoidDoubleDerivative
    elseif activationFunction == "tanh"
        return tanH, tanHDerivative, tanHDoubleDerivative
    else
        println("The acitvation function is not implemented. Please use one of the following: sigmoid, ")
    end
end

function sigmoid(z)
    return 1/(1 + exp(-z))
end

function sigmoidDerivative(z)
    sig = sigmoid(z)
    return sig*(1-sig)
end

function sigmoidDoubleDerivative(z)
    sig = sigmoid(z)
    return sig*(1-sig)*(1-2*sig)
end

function tanH(z)
    temp = exp(2*z)
    return (temp - 1)/(temp + 1)
end 

function tanHDerivative(z)
    return 1 - tanH(z)^2
end 

function tanHDoubleDerivative(z)
    return -2*tanH(z)*(1 - tanH(z)^2)
end

function sortInput(x)
    return 
end

end