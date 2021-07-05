module initializeSystem 

export slater, slaterJastrow, slaterRBM, slaterNN
export initializeSystemSlater, initializeSystemSlaterJastrow
export initializeSystemSlaterRBM, initializeSystemSlaterNN

include("Various/quantumNumbers.jl")
include("Wavefunctions/singleParticle.jl")

using Random
using .quantumNumbers
using .singleParticle
using Flux
using LinearAlgebra
# using ..neuralNetwork

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
end 

"""
    initializeSystemSlater(numParticles, numDimensions; alpha = 1.0, omega = 1.0,
    beta = 1.0, interacting = false)

Initializes the slater system struct with given constants and slater matrices. 
"""
function initializeSystemSlater(numParticles, numDimensions; alpha = 1.0, omega = 1.0, beta = 1.0, interacting = false)
    particles = initializeParticlesNormalDist(numParticles, numDimensions)
    sSU, sSD, iSSU, iSSD = initializeSlaterMatrix(particles, numParticles, numDimensions, alpha, omega) 
    system = slater(particles, numParticles, numDimensions, alpha, omega, beta, interacting, sSU, sSD, iSSU, iSSD)
end

function initializeSlaterMatrix(particles, numParticles, numDimensions, alpha, omega)
    slaterMatrixSpinUp = zeros(Int(numParticles/2),  Int(numParticles/2))

    for row=1:size(slaterMatrixSpinUp)[1]
        for col=1:size(slaterMatrixSpinUp)[2]
            # nx = quantumNumbers2D[col,1]
            # ny = quantumNumbers2D[col, 2]
            # slaterMatrixSpinUp[row, col] = singleParticleHermitian(particles[row, :], nx, ny, alpha, omega)
            qN = getQuantumNumbers(col, numDimensions)
            slaterMatrixSpinUp[row, col] = singleParticleHermitian(particles[row, :], qN, alpha, omega)

        end 
    end

    slaterMatrixSpinDown = zeros(Int(numParticles/2), Int(numParticles/2))

    for row=1:size(slaterMatrixSpinDown)[1]
        for col=1:size(slaterMatrixSpinDown)[2]
            # nx = quantumNumbers2D[col,1]
            # ny = quantumNumbers2D[col, 2]
            # slaterMatrixSpinDown[row, col] = singleParticleHermitian(particles[Int(row + numParticles/2),:], nx, ny, alpha, omega)
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
    b = rand(Float64, N, 1) .-0.5
    a = rand(Float64, M, 1) .-0.5

    # Initializes the weights.
    w = rand(Float64, M, N) .-0.5

    # Initializes the visble and the hidden layer.
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
    nn = NN(Chain(Dense(numParticles*numDimensions, numHiddenNeurons, sigmoid), Dense(numHiddenNeurons, 1)))
end 

function initializeSystemSlaterNN(numParticles, numDimensions; alpha = 1.0, omega = 1.0, beta = 1.0, interacting = false, numHiddenNeurons = 10)
    particles = initializeParticlesNormalDist(numParticles, numDimensions)
    sSU, sSD, iSSU, iSSD = initializeSlaterMatrix(particles, numParticles, numDimensions, alpha, omega) 
    nn = initializeNN(numParticles, numDimensions, numHiddenNeurons)
    system = slaterNN(particles, numParticles, numDimensions, alpha, omega, beta, interacting, sSU, sSD, iSSU, iSSD, nn)
end



end #MODULE
