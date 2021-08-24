module initializeSystem 

export slater, slaterJastrow, slaterRBM, slaterNN, slaterNNAnalytical, gaussianJastrow, gaussianNNAnalytical
export slaterJastrowNNAnalytical
export initializeSystemSlater, initializeSystemSlaterJastrow
export initializeSystemSlaterRBM, initializeSystemSlaterNN
export initializeSystemSlaterNNAnalytical, initializeSystemSlaterJastrowNNAnalytical
export initializeSystemGaussianJastrow, initializeSystemGaussianNNAnalytical
export getVariationalParameters, setVariationalParameters!

include("Various/quantumNumbers.jl")
include("Wavefunctions/singleParticle.jl")

using Random
using .quantumNumbers
using .singleParticle
using Flux
using LinearAlgebra
using StaticArrays
# using ..neuralNetwork

"""
    preAllocations

Struct for storing matrices that can be pre allocated to save time and space. 
"""
struct preallocations 
    H
    H_gradient#::Array{Float64, 1}
    H_doubleDerivative
end


""" 
    initializeParticlesNormalDist(numParticles::Int64, numDimensions::Int64)

Returns a random 2D array of position of particles. 
"""
function initializeParticlesNormalDist(numParticles::Int64, numDimensions::Int64)
    rng = MersenneTwister(1234)
    particles = 0.05*randn(rng, Float64, (numParticles, numDimensions))
    return particles
end

"""
    slater(particles::Array{Float64, 2}, numParticles::Int64, numDimensions::Int64 
        alpha::Float64, omega::Float64, beta::Float64, interacting::Bool, 
        slaterMatrixSpinUp::Array{Float64, 2}, slaterMatrixSpinDown::Array{Float64, 2}
        inverseSlaterMatrixSpinUp::Array{Float64, 2}, inverseSlaterMatrixSpinDown::Array{Float64, 2}) 

Struct for storing the information about the wavefunction and the system. 
"""
mutable struct slater
    particles::Array{Float64, 2}
    numParticles::Int64 
    numDimensions::Int64

    alpha::Float64
    omega::Float64
    beta::Float64

    interacting::Bool

    slaterMatrixSpinUp::Array{Float64, 2}
    slaterMatrixSpinDown::Array{Float64, 2}

    inverseSlaterMatrixSpinUp::Array{Float64, 2}
    inverseSlaterMatrixSpinDown::Array{Float64, 2}

    prealloc::preallocations
end 

"""
    initializeSystemSlater(numParticles, numDimensions; alpha = 1.0, omega = 1.0,
    beta = 1.0, interacting = false)

Initializes the slater system struct with given constants and slater matrices. 
"""
function initializeSystemSlater(numParticles, numDimensions; alpha = 1.0, omega = 1.0, beta = 1.0, interacting = false)
    particles = initializeParticlesNormalDist(numParticles, numDimensions)
    sSU, sSD, iSSU, iSSD = initializeSlaterMatrix(particles, numParticles, numDimensions, alpha, omega)
    pA = preallocations(zeros(numDimensions), zeros(numDimensions), zeros(numDimensions) )
    system = slater(particles, numParticles, numDimensions, alpha, omega, beta, interacting, sSU, sSD, iSSU, iSSD, pA)
end

function initializeSlaterMatrix(particles, numParticles, numDimensions, alpha, omega)
    slaterMatrixSpinUp = zeros(Int(numParticles/2),  Int(numParticles/2))

    for row=1:size(slaterMatrixSpinUp)[1]
        for col=1:size(slaterMatrixSpinUp)[2]
            qN = getQuantumNumbers(col, numDimensions)
            slaterMatrixSpinUp[row, col] = singleParticleHermitian(particles[row, :], qN, alpha, omega)
        end 
    end

    slaterMatrixSpinDown = zeros(Int(numParticles/2), Int(numParticles/2))

    for row=1:size(slaterMatrixSpinDown)[1]
        for col=1:size(slaterMatrixSpinDown)[2]
            qN = getQuantumNumbers(col, numDimensions)
            slaterMatrixSpinDown[row, col] = singleParticleHermitian(particles[Int(row + numParticles/2),:], qN, alpha, omega)
        end 
    end

    invSlaterMatrixSpinUp = inv(slaterMatrixSpinUp)
    invSlaterMatrixSpinDown = inv(slaterMatrixSpinDown)

    return (slaterMatrixSpinUp, slaterMatrixSpinDown, invSlaterMatrixSpinUp, invSlaterMatrixSpinDown)
end

struct Jastrow 
    kappa::Array{Float64, 2}
    distanceMatrix::Array{Float64, 2}
end

mutable struct slaterJastrow
    particles::Array{Float64, 2}
    numParticles::Int64 
    numDimensions::Int64

    alpha::Float64 
    omega::Float64
    beta::Float64

    interacting::Bool

    slaterMatrixSpinUp::Array{Float64, 2}
    slaterMatrixSpinDown::Array{Float64, 2}

    inverseSlaterMatrixSpinUp::Array{Float64, 2}
    inverseSlaterMatrixSpinDown::Array{Float64, 2}

    jastrowFactor::Jastrow
end 

function initializeSystemSlaterJastrow(numParticles, numDimensions; alpha = 1.0, omega = 1.0, beta = 1.0, interacting = false)
    particles = initializeParticlesNormalDist(numParticles, numDimensions)
    sSU, sSD, iSSU, iSSD = initializeSlaterMatrix(particles, numParticles, numDimensions, alpha, omega) 
    distanceMatrix = zeros(numParticles, numParticles)
    println(distanceMatrix)
    for i=1:numParticles
        for j=i:numParticles 
            difference = particles[i, :] - particles[j, :]
            distance = sqrt(dot(difference, difference))
            distanceMatrix[i,j] = distance
        end 
    end 
    display(distanceMatrix)
    distanceMatrix = distanceMatrix + distanceMatrix'
    display(distanceMatrix)
    rng = MersenneTwister(123)
    kappa = randn(rng, Float64, (numParticles, numParticles))
    kappa = 0.5*(kappa + kappa')
    jastrowFactor = Jastrow(kappa, distanceMatrix)
    display(particles)
    system = slaterJastrow(particles, numParticles, numDimensions, alpha, omega, beta, interacting, sSU, sSD, iSSU, iSSD, jastrowFactor)
    return system
end

struct simpleJastrow 
    beta::Array{Float64,1}
    distanceMatrix::Array{Float64, 2}
end 

struct gaussianJastrow 
    particles::Array{Float64, 2}
    numParticles::Int64 
    numDimensions::Int64

    alpha::Float64 
    omega::Float64

    interacting::Bool

    jastrowFactor::simpleJastrow
end

function initializeSystemGaussianJastrow(numParticles, numDimensions; alpha = 1.0, omega = 1.0, beta = 1.0, interacting = false)
    particles = initializeParticlesNormalDist(numParticles, numDimensions)

    distanceMatrix = zeros(numParticles, numParticles)
    println(distanceMatrix)
    for i=1:numParticles
        for j=i:numParticles 
            difference = particles[i, :] - particles[j, :]
            distance = sqrt(dot(difference, difference))
            distanceMatrix[i,j] = distance
        end 
    end 
    display(distanceMatrix)
    distanceMatrix = distanceMatrix + distanceMatrix'
    display(distanceMatrix)


    jastrowFactor = simpleJastrow([beta], distanceMatrix)
    system = gaussianJastrow(particles, numParticles, numDimensions, alpha, omega, interacting, jastrowFactor)
    return system
end

struct NQS
    num_particles::Int64
    num_dims::Int64

    # Bias for hidden layer
    b::Array{Float64, 2}
    # Bias for visible layer
    a::Array{Float64, 2}

    # Weights
    w::Array{Float64, 2}

    # Hidden layer
    h::Array{Float64, 2}

    sigma_squared::Float64
    interacting::Bool
end

mutable struct slaterRBM 
    particles::Array{Float64, 2}
    numParticles::Int64 
    numDimensions::Int64

    alpha::Float64 
    omega::Float64
    beta::Float64

    slaterMatrixSpinUp::Array{Float64, 2}
    slaterMatrixSpinDown::Array{Float64, 2}

    inverseSlaterMatrixSpinUp::Array{Float64, 2}
    inverseSlaterMatrixSpinDown::Array{Float64, 2}

    nqs::NQS
end 

function initializeRBM(position, num_particles::Int64, num_dims::Int64, M::Int64, N::Int64, sig_sq::Float64 = 0.5, inter::Bool = false)
    # Initializes the biases
    b = randn(Float64, N, 1)*0.05 
    a = randn(Float64, M, 1)*0.05

    # Initializes the weights.
    w = randn(Float64, M, N)*0.05 

    # Initializes the hidden layer.
    h = rand(0:1, N, 1)

    interacting = inter

    return NQS(num_particles, num_dims, b, a, w, h, sig_sq, interacting)
end

function initializeSystemSlaterRBM(numParticles, numDimensions, numHidden; alpha = 1.0, omega = 1.0, beta = 1.0, sigmaSquared = 1.0, interacting = false)
    particles = initializeParticlesNormalDist(numParticles, numDimensions)
    sSU, sSD, iSSU, iSSD = initializeSlaterMatrix(particles, numParticles, numDimensions, alpha, omega) 
    nqs = initializeRBM(particles, numParticles, numDimensions, numParticles*numDimensions, numHidden, sigmaSquared, interacting)
    system = slaterRBM(particles, numParticles, numDimensions, alpha, omega, beta, sSU, sSD, iSSU, iSSD, nqs)
end

struct NN
    model
end

struct slaterNN 
    particles::Array{Float64, 2}
    numParticles::Int64 
    numDimensions::Int64

    alpha::Float64 
    omega::Float64
    beta::Float64

    interacting::Bool

    slaterMatrixSpinUp::Array{Float64, 2}
    slaterMatrixSpinDown::Array{Float64, 2}

    inverseSlaterMatrixSpinUp::Array{Float64, 2}
    inverseSlaterMatrixSpinDown::Array{Float64, 2}

    nn::NN
end 

function initializeNN(numParticles, numDimensions, numHiddenNeurons)
    nn = NN(Chain(Dense(numParticles*numDimensions, numHiddenNeurons, sigmoid), Dense(numHiddenNeurons,numHiddenNeurons, sigmoid), Dense(numHiddenNeurons, 1, sigmoid)))
end 

function initializeSystemSlaterNN(numParticles, numDimensions; alpha = 1.0, omega = 1.0, beta = 1.0, interacting = false, numHiddenNeurons = 10)
    particles = initializeParticlesNormalDist(numParticles, numDimensions)
    sSU, sSD, iSSU, iSSD = initializeSlaterMatrix(particles, numParticles, numDimensions, alpha, omega) 
    nn = initializeNN(numParticles, numDimensions, numHiddenNeurons)
    system = slaterNN(particles, numParticles, numDimensions, alpha, omega, beta, interacting, sSU, sSD, iSSU, iSSD, nn)
end

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
    activationFunctionDerivative 
    activationFunctionDoubleDerivative
end

function initializeNNAnalytical(numParticles, numDimensions, numNodesLayer1, numNodesLayer2, activationFunction, activationFunctionDerivative, activationFunctionDoubleDerivative)
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
                            activationFunction, activationFunctionDerivative, activationFunctionDoubleDerivative)
end

struct slaterNNAnalytical
    particles::Array{Float64, 2}
    numParticles::Int64 
    numDimensions::Int64

    alpha::Float64 
    omega::Float64
    beta::Float64

    interacting::Bool

    slaterMatrixSpinUp::Array{Float64, 2}
    slaterMatrixSpinDown::Array{Float64, 2}

    inverseSlaterMatrixSpinUp::Array{Float64, 2}
    inverseSlaterMatrixSpinDown::Array{Float64, 2}

    nn::NNAnalytical
end 

function initializeSystemSlaterNNAnalytical(numParticles, numDimensions; alpha = 1.0, omega = 1.0, beta = 1.0, interacting = false, numNodesLayer1 = 10, numNodesLayer2 = 10)
    particles = initializeParticlesNormalDist(numParticles, numDimensions)
    sSU, sSD, iSSU, iSSD = initializeSlaterMatrix(particles, numParticles, numDimensions, alpha, omega) 
    nn = initializeNNAnalytical(numParticles, numDimensions, numNodesLayer1, numNodesLayer2, sigmoid, sigmoid_derivative, sigmoid_double_derivative)
    system = slaterNNAnalytical(particles, numParticles, numDimensions, alpha, omega, beta, interacting, sSU, sSD, iSSU, iSSD, nn)
end

struct gaussianNNAnalytical 
    particles::Array{Float64, 2}
    numParticles::Int64 
    numDimensions::Int64

    alpha::Float64 
    omega::Float64

    interacting::Bool

    nn::NNAnalytical
end

function initializeSystemGaussianNNAnalytical(numParticles, numDimensions; alpha = 1.0, omega = 1.0, beta = 1.0, interacting = false, numNodesLayer1 = 10, numNodesLayer2 = 10)
    particles = initializeParticlesNormalDist(numParticles, numDimensions)
    nn = initializeNNAnalytical(numParticles, numDimensions, numNodesLayer1, numNodesLayer2, sigmoid, sigmoid_derivative, sigmoid_double_derivative)
    return gaussianNNAnalytical(particles, numParticles, numDimensions, alpha, omega, interacting, nn)
end


struct slaterJastrowNNAnalytical  
    particles::Array{Float64, 2}
    numParticles::Int64 
    numDimensions::Int64

    alpha::Float64 
    omega::Float64
    beta::Float64

    interacting::Bool

    slaterMatrixSpinUp::Array{Float64, 2}
    slaterMatrixSpinDown::Array{Float64, 2}

    inverseSlaterMatrixSpinUp::Array{Float64, 2}
    inverseSlaterMatrixSpinDown::Array{Float64, 2}

    jastrowFactor::Jastrow

    nn::NNAnalytical
end

function initializeSystemSlaterJastrowNNAnalytical()
    particles = initializeParticlesNormalDist(numParticles, numDimensions)
    sSU, sSD, iSSU, iSSD = initializeSlaterMatrix(particles, numParticles, numDimensions, alpha, omega) 
    distanceMatrix = zeros(numParticles, numParticles)
    println(distanceMatrix)
    for i=1:numParticles
        for j=i:numParticles 
            difference = particles[i, :] - particles[j, :]
            distance = sqrt(dot(difference, difference))
            distanceMatrix[i,j] = distance
        end 
    end 
    display(distanceMatrix)
    distanceMatrix = distanceMatrix + distanceMatrix'
    display(distanceMatrix)
    rng = MersenneTwister(123)
    kappa = randn(rng, Float64, (numParticles, numParticles))
    kappa = 0.5*(kappa + kappa')
    jastrowFactor = Jastrow(kappa, distanceMatrix)
    nn = initializeNNAnalytical(numParticles, numDimensions, numNodesLayer1, numNodesLayer2, sigmoid, sigmoid_derivative, sigmoid_double_derivative)
    return  slaterJastrowNNAnalytical(particles, numParticles, numDimensions, alpha, omega, beta, interacting, sSU, sSD, iSSU, iSSD, jastrowFactor, nn)  
end

function setVariationalParameters!(system::slater, parameters)
    if haskey(parameters, "alpha") 
        system.alpha = parameters["alpha"]
    end 
end

function setVariationalParameters!(system::slaterJastrow, parameters)
    if haskey(parameters, "alpha") 
        system.alpha = parameters["alpha"]
    end 
    if haskey(parameters, "kappa")
        try 
            system.jastrowFactor.kappa = parameters["kappa"]
        catch 
            println("Kappa has to be a matrix of dimensions N x N")
        end 
    end
end

function getVariationalParameters(system::slater)
    parameters = Dict()
    parameters["alpha"] = system.alpha
    return parameters
end 

function getVariationalParameters(system::slaterJastrow)
    parameters = Dict()
    parameters["alpha"] = system.alpha
    parameters["kappa"] = system.jastrowFactor.kappa
    return parameters
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

end #MODULE
