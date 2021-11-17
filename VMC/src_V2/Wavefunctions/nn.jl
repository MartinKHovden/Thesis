module nn 

export NN

using StaticArrays
using ..wavefunction

""" 
    struct NN 

Struct for storing the information about, and the state of, the neural network. 
The network is restricted to two hidden layers, however, the number of nodes and 
the activation function can be changed as wanted. 

"""
struct NN
    variationalParameter::Array
    variationalParameterGradient::Array

    a::Array 
    z::Array 

    aGrad::Array 
    aDoubleGrad::Array 
    aDoubleGradTemp::Array

    delta::Array

    activationFunction 
    activationFunctionDerivative 
    activationFunctionDoubleDerivative

    function NN(system, numNodesLayer1::Int64, numNodesLayer2::Int64, activationFunction::String)
        numParticles = system.numParticles
        numDimensions = system.numDimensions

        # Initialization of the weights
        w1 = @MMatrix randn(numNodesLayer1, numParticles*numDimensions)
        w1 *= 0.01
        w2 = @MMatrix randn(numNodesLayer2, numNodesLayer1)
        w2*= 0.01
        w3 = @MMatrix randn(1, numNodesLayer2)
        w3*= 0.01
    
        # Initialization of the biases
        b1 = @MVector randn(numNodesLayer1)
        b2 = @MVector randn(numNodesLayer2)
        b3 = @MVector randn(1)

        # Placing the weights and biases in an Array. 
        params = []
        push!(params, w1)
        push!(params, b1)
        push!(params, w2)
        push!(params, b2)
        push!(params, w3)
        push!(params, b3)

        w1_grad = @MMatrix randn(numNodesLayer1, numParticles*numDimensions) 
        w2_grad = @MMatrix randn(numNodesLayer2, numNodesLayer1) 
        w3_grad = @MMatrix randn(1, numNodesLayer2) 
    
        b1_grad = @MVector randn(numNodesLayer1)
        b2_grad = @MVector randn(numNodesLayer2)
        b3_grad = @MVector randn(1)

        paramGrads = []
        push!(paramGrads, w1_grad)
        push!(paramGrads, b1_grad)
        push!(paramGrads, w2_grad)
        push!(paramGrads, b2_grad)
        push!(paramGrads, w3_grad)
        push!(paramGrads, b3_grad)

        a1 = @MVector zeros(numNodesLayer1)
        a2 = @MVector zeros(numNodesLayer2)
        a3 = @MVector zeros(1)

        a = []
        push!(a, a1)
        push!(a, a2)
        push!(a, a3)

        z1 = @MVector zeros(numNodesLayer1)
        z2 = @MVector zeros(numNodesLayer2)
        z3 = @MVector zeros(1)

        z = []
        push!(z, a1)
        push!(z, a2)
        push!(z, a3)
    
        a1_grad = @MMatrix zeros(numNodesLayer1, numParticles*numDimensions)
        a2_grad = @MMatrix zeros(numNodesLayer2, numParticles*numDimensions)
        a3_grad = @MMatrix zeros(1, numParticles*numDimensions)

        aGrad = []
        push!(aGrad, a1_grad)
        push!(aGrad, a2_grad)
        push!(aGrad, a3_grad)

        a1_double_grad = @MMatrix zeros(numNodesLayer1, numParticles*numDimensions)
        a2_double_grad = @MMatrix zeros(numNodesLayer2, numParticles*numDimensions)
        a3_double_grad = @MMatrix zeros(1, numParticles*numDimensions)

        aDoubleGrad = []
        push!(aDoubleGrad, a1_double_grad)
        push!(aDoubleGrad, a2_double_grad)
        push!(aDoubleGrad, a3_double_grad)

        a1_double_grad_temp = @MMatrix zeros(numNodesLayer1, numParticles*numDimensions)
        a2_double_grad_temp = @MMatrix zeros(numNodesLayer2, numParticles*numDimensions)
        a3_double_grad_temp = @MMatrix zeros(1, numParticles*numDimensions)

        aDoubleGradTemp = []
        push!(aDoubleGradTemp, a1_double_grad_temp)
        push!(aDoubleGradTemp, a2_double_grad_temp)
        push!(aDoubleGradTemp, a3_double_grad_temp)

        delta1 = @MVector zeros(numNodesLayer1)
        delta2 = @MVector zeros(numNodesLayer2)
        delta3 = @MVector zeros(1)

        delta = []
        push!(delta, delta1)
        push!(delta, delta2)
        push!(delta, delta3)

        aFunction, aFunctionDerivative, aFunctionDoubleDerivative = getActivationFunctions(activationFunction)

        return new(params, paramGrads, a, z, aGrad, aDoubleGrad, aDoubleGradTemp, delta, aFunction, aFunctionDerivative, aFunctionDoubleDerivative)
    end
end 

function wavefunction.computeRatio()
    return 1.0
end

function wavefunction.computeGradient()
end

function wavefunction.computeLaplacian()
end 

function wavefunction.computeParameterGradient()
end

function getActivationFunctions(activationFunction::String)
    if activationFunction == "sigmoid"
        return sigmoid, sigmoid_derivative, sigmoid_double_derivative
    else
        println("The acitvation function is not implemented. Please use one of the following: sigmoid, ")
    end
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

end