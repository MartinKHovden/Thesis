module initializeSystem 

export slater, initializeSystemSlater, slaterJatrow, slaterRBM, slaterNN

include("Various/quantumNumbers.jl")
include("Wavefunctions/singleParticle.jl")

using Random
using .quantumNumbers
using .singleParticle


# struct system 
#     particles::Array{Float64, 2}
#     numParticles::Int64 
#     numDims::Int64

#     alpha::Float64 
#     omega::Float64
#     beta::Float64

#     interacting::Bool

#     wavefunction

# end

struct slater
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

function initializeSystemSlater(numParticles, numDimensions; alpha = 1.0, omega = 1.0, beta = 1.0, interacting = false)
    particles = initializeParticlesNormalDist(numParticles, numDimensions)
    sSU, sSD, iSSU, iSSD = initializeSlaterMatrix(particles, numParticles, numDimensions, alpha, omega) 
    system = slater(particles, numParticles, numDimensions, alpha, omega, beta, interacting, sSU, sSD, iSSU, iSSD)
end

function initializeParticlesNormalDist(numParticles, numDimensions)
    rng = MersenneTwister(1234)
    particles = 0.05*randn(rng, Float64, (numParticles, numDimensions))
    return particles
end

function initializeSlaterMatrix(particles, numParticles, numDimensions, alpha, omega)
    slaterMatrixSpinUp = zeros(Int(numParticles/2),  Int(numParticles/2))

    for row=1:size(slaterMatrixSpinUp)[1]
        for col=1:size(slaterMatrixSpinUp)[2]
            nx = quantumNumbers2D[col,1]
            ny = quantumNumbers2D[col, 2]
            slaterMatrixSpinUp[row, col] = singleParticleHermitian(particles[row, :], nx, ny, alpha, omega)
        end 
    end

    slaterMatrixSpinDown = zeros(Int(numParticles/2), Int(numParticles/2))

    for row=1:size(slaterMatrixSpinDown)[1]
        for col=1:size(slaterMatrixSpinDown)[2]
            nx = quantumNumbers2D[col,1]
            ny = quantumNumbers2D[col, 2]
            slaterMatrixSpinDown[row, col] = singleParticleHermitian(particles[Int(row + numParticles/2),:], nx, ny, alpha, omega)
        end 
    end

    invSlaterMatrixSpinUp = inv(slaterMatrixSpinUp)
    invSlaterMatrixSpinDown = inv(slaterMatrixSpinDown)

    return (slaterMatrixSpinUp, slaterMatrixSpinDown, invSlaterMatrixSpinUp, invSlaterMatrixSpinDown)
end

struct slaterJastrow
    particles::Array{Float64, 2}
    n_particles::Int64 
    n_dims::Int64

    alpha::Float64 
    omega::Float64
    beta::Float64

    slaterMatrixSpinUp::Array{Float64, 2}
    slaterMatrixSpinDown::Array{Float64, 2}

    inverseSlaterMatrixSpinUp::Array{Float64, 2}
    inverseSlaterMatrixSpinDown::Array{Float64, 2}
end 

struct slaterRBM 
    particles::Array{Float64, 2}
    n_particles::Int64 
    n_dims::Int64

    alpha::Float64 
    omega::Float64
    beta::Float64

    slaterMatrixSpinUp::Array{Float64, 2}
    slaterMatrixSpinDown::Array{Float64, 2}

    inverseSlaterMatrixSpinUp::Array{Float64, 2}
    inverseSlaterMatrixSpinDown::Array{Float64, 2}

    # nqs::NQS
end 

struct slaterNN 
    particles::Array{Float64, 2}
    n_particles::Int64 
    n_dims::Int64

    alpha::Float64 
    omega::Float64
    beta::Float64

    slaterMatrixSpinUp::Array{Float64, 2}
    slaterMatrixSpinDown::Array{Float64, 2}

    inverseSlaterMatrixSpinUp::Array{Float64, 2}
    inverseSlaterMatrixSpinDown::Array{Float64, 2}

    # nn::NN
end 

end 