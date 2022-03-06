module slater

export SlaterMatrix
export computeRatio, computeGradient, computeLaplacian
export updateElement!, inverseSlaterMatrixUpdate, computeDriftForce

include("hermite.jl")
include("singleParticle.jl")

using LinearAlgebra

using .hermite
using .singleParticle

using ..wavefunction

struct SlaterMatrix
    slaterMatrixSpinUp::Array{Float64, 2}
    slaterMatrixSpinDown::Array{Float64, 2}

    inverseSlaterMatrixSpinUp::Array{Float64, 2}
    inverseSlaterMatrixSpinDown::Array{Float64, 2}

    R::Array{Float64,1}

    function SlaterMatrix(system)
        sSU, sSD, iSSU, iSSD = initializeSlaterMatrix(system.particles, 
                                                    system.numParticles, 
                                                    system.numDimensions, 
                                                    system.omega)
        return new(sSU, sSD, iSSU, iSSD, [1.0])
    end
end

function initializeSlaterMatrix(particles, numParticles, numDimensions, omega)
    slaterMatrixSpinUp = zeros(Int(numParticles/2),  Int(numParticles/2))
    for row=1:size(slaterMatrixSpinUp)[1]
        for col=1:size(slaterMatrixSpinUp)[2]
            qN = getQuantumNumbers(col, numDimensions)
            slaterMatrixSpinUp[row, col] = singleParticleHermitian(particles[row, :], qN, omega)
        end 
    end

    slaterMatrixSpinDown = zeros(Int(numParticles/2), Int(numParticles/2))
    for row=1:size(slaterMatrixSpinDown)[1]
        for col=1:size(slaterMatrixSpinDown)[2]
            qN = getQuantumNumbers(col, numDimensions)
            slaterMatrixSpinDown[row, col] = singleParticleHermitian(particles[Int(row + numParticles/2),:], qN, omega)
        end 
    end

    invSlaterMatrixSpinUp = inv(slaterMatrixSpinUp)
    invSlaterMatrixSpinDown = inv(slaterMatrixSpinDown)

    return (slaterMatrixSpinUp, 
        slaterMatrixSpinDown, 
        invSlaterMatrixSpinUp, 
        invSlaterMatrixSpinDown)
end

function wavefunction.computeRatio(system, 
                                wavefunctionElement::SlaterMatrix, 
                                particleToUpdate::Int64, 
                                coordinateToUpdate::Int64, 
                                oldPosition)
    R = slaterMatrixComputeRatio(system, wavefunctionElement, particleToUpdate)
    wavefunctionElement.R[1] = R
    return R^2
end

function slaterMatrixComputeRatio(system, slater::SlaterMatrix, particleMoved)
    if particleMoved <= system.numParticles/2
        newSlaterMatrixSpinUp = deepcopy(slater.slaterMatrixSpinUp)
        oldInverseSlaterMatrixSpinUp = deepcopy(slater.inverseSlaterMatrixSpinUp)
        R = dot(newSlaterMatrixSpinUp[particleMoved, :], slater.inverseSlaterMatrixSpinUp[:,particleMoved])
    else 
        newSlaterMatrixSpinDown = deepcopy(slater.slaterMatrixSpinDown)
        oldInverseSlaterMatrixSpinDown = deepcopy(slater.inverseSlaterMatrixSpinDown)
        R = dot(newSlaterMatrixSpinDown[Int64(particleMoved - system.numParticles/2), :], slater.inverseSlaterMatrixSpinDown[:, Int64(particleMoved - system.numParticles/2)])
    end 
    return R
end

function wavefunction.computeGradient(system, wavefunctionElement::SlaterMatrix)
    numParticles = system.numParticles
    numDimensions = system.numDimensions
    gradient = zeros(numParticles*numDimensions)
    for particle=1:numParticles
        gradient[(particle-1)*numDimensions + 1: (particle-1)*numDimensions + numDimensions] = slaterDeterminantComputeGradient(system, wavefunctionElement, particle)
    end
    return gradient
end

function slaterDeterminantComputeGradient(system, slater::SlaterMatrix, particle_num)
    if particle_num <= system.numParticles/2
        return slaterDeterminantSpinUpComputeGradient(system, slater, particle_num)
    else 
        return slaterDeterminantSpinDownComputeGradient(system, slater, particle_num)
    end 
end

function slaterDeterminantSpinUpComputeGradient(system, 
                                            slater::SlaterMatrix, 
                                            particle_num::Int64)
    d = system.numDimensions
    N = Int64(system.numParticles/2)
    omega = system.omega
    grad = zeros(d)
    for j=1:N
        grad[:] += singleParticleHermitianGradient(system.particles[particle_num,:], getQuantumNumbers(j, d), omega)*slater.inverseSlaterMatrixSpinUp[j, particle_num]
    end
    return grad
end 

function slaterDeterminantSpinDownComputeGradient(system,slater::SlaterMatrix, 
                                                particle_num::Int64)
    d = system.numDimensions
    row = Int(particle_num - system.numParticles/2)
    N = Int64(system.numParticles/2)
    omega = system.omega
    grad = zeros(d)
    for j=1:N
        grad[:] += singleParticleHermitianGradient(system.particles[particle_num,:], getQuantumNumbers(j, d), omega)*slater.inverseSlaterMatrixSpinDown[j, row]
    end
    return grad
end

function wavefunction.computeLaplacian(system, wavefunctionElement::SlaterMatrix)
    numParticles = system.numParticles
    numDimensions = system.numDimensions
    laplacian = 0
    for particle=1:numParticles
        laplacian += slaterDeterminantComputeLaplacian(system, 
                                                    wavefunctionElement::SlaterMatrix, 
                                                    particle::Int64)
    end
    return laplacian
end 

function slaterDeterminantComputeLaplacian(system, 
                                        slater::SlaterMatrix, 
                                        particle_num::Int64)
    if particle_num <= system.numParticles/2
        return slaterDeterminantSpinUpComputeLaplacian(system, slater, particle_num)
    else 
        return slaterDeterminantSpinDownComputeLaplacian(system, slater, particle_num)
    end 
end

function slaterDeterminantSpinUpComputeLaplacian(system, 
                                            slater::SlaterMatrix, 
                                            particle_num::Int64)
    d = system.numDimensions
    N = Int64(system.numParticles/2)
    omega = system.omega
    laplacian = 0
    for j=1:N
        laplacian += (singleParticleHermitianLaplacian(system.particles[particle_num,:], getQuantumNumbers(j, system.numDimensions), omega)*slater.inverseSlaterMatrixSpinUp[j, particle_num])
    end

    temp = zeros(d)
    for j=1:N 
        temp += singleParticleHermitianGradient(system.particles[particle_num,:], getQuantumNumbers(j, system.numDimensions), omega)*slater.inverseSlaterMatrixSpinUp[j, particle_num]
    end
    return laplacian - dot(temp, temp)
end

function slaterDeterminantSpinDownComputeLaplacian(system, 
                                                slater::SlaterMatrix, 
                                                particle_num::Int64)
    d = system.numDimensions
    row = Int(particle_num - system.numParticles/2)
    N = Int64(system.numParticles/2)
    omega = system.omega
    laplacian = 0
    for j=1:N
        laplacian += (singleParticleHermitianLaplacian(system.particles[particle_num,:], getQuantumNumbers(j, d), omega)*slater.inverseSlaterMatrixSpinDown[j, row])

    end
    temp = zeros(d)
    for j=1:N 
        temp += (singleParticleHermitianGradient(system.particles[particle_num,:], getQuantumNumbers(j, d), omega)*slater.inverseSlaterMatrixSpinDown[j, row])
    end
    return laplacian - dot(temp, temp)
end

function wavefunction.updateElement!(system, 
                                wavefunctionElement::SlaterMatrix, 
                                particleNum::Int64)
    slaterMatrixUpdate!(system, wavefunctionElement, particleNum)
end

function slaterMatrixUpdate!(system, slater::SlaterMatrix, particle)
    numParticles = system.numParticles

    if particle <= numParticles/2
        slaterMatrixSpinUpUpdateRow(system, slater, particle)
    else 
        slaterMatrixSpinDownUpdateRow(system, slater, Int64(particle-numParticles/2))
    end
end

function slaterMatrixSpinUpUpdateRow(system, slater::SlaterMatrix, row)
    coordinates = system.particles[row, :]
    omega = system.omega
    for col=1:size(slater.slaterMatrixSpinUp)[2]
        slater.slaterMatrixSpinUp[row, col] = singleParticleHermitian(coordinates, getQuantumNumbers(col, system.numDimensions), omega)
    end
end 

function slaterMatrixSpinDownUpdateRow(system, slater::SlaterMatrix, row)
    coordinates = system.particles[Int(row + system.numParticles/2), :]
    omega = system.omega
    for col=1:size(slater.slaterMatrixSpinDown)[2]
        slater.slaterMatrixSpinDown[row, col] = singleParticleHermitian(coordinates, getQuantumNumbers(col, system.numDimensions), omega)
    end
end 


function inverseSlaterMatrixUpdate(system, slater::SlaterMatrix, col, R)
    R = R[1]
    if col <= system.numParticles/2
        inverseSlaterMatrixSpinUpUpdateCol(system, slater, col, R)
    else 
        inverseSlaterMatrixSpinDownUpdateCol(system, slater, Int64(col - system.numParticles/2), R)
    end 

    return nothing
end

function inverseSlaterMatrixSpinUpUpdateCol(system, slater::SlaterMatrix, col, R)
    newSlater = deepcopy(slater.slaterMatrixSpinUp)
    oldSlaterInverse = deepcopy(slater.inverseSlaterMatrixSpinUp)
    N = system.numParticles/2
    for j = 1:N
        if j != col
            j=Int(j)
            S_j = dot(newSlater[col,:], oldSlaterInverse[:, j])
            value = oldSlaterInverse[:, j] - (S_j/R)*oldSlaterInverse[:, col] 
            slater.inverseSlaterMatrixSpinUp[:, j] = copy(value)
        end 
    end
    value = (1/R)*oldSlaterInverse[:,col]
    slater.inverseSlaterMatrixSpinUp[:, col] = copy(value)

    return nothing
end

function inverseSlaterMatrixSpinDownUpdateCol(system, slater::SlaterMatrix, col, R)
    newSlater = deepcopy(slater.slaterMatrixSpinDown)
    oldSlaterInverse = deepcopy(slater.inverseSlaterMatrixSpinDown)
    N = system.numParticles/2
    for j = 1:N
        if j != col
            j = Int(j)
            S_j = dot(newSlater[col, :], oldSlaterInverse[:, j])
            value = oldSlaterInverse[:, j] - (S_j/R)*oldSlaterInverse[:, col] 
            slater.inverseSlaterMatrixSpinDown[:, j] = copy(value)
        end 
    end
    value = (1/R)*oldSlaterInverse[:,col]
    slater.inverseSlaterMatrixSpinDown[:, col] = copy(value)

    return nothing
end

function wavefunction.computeDriftForce(system, 
                                    element::SlaterMatrix, 
                                    particleToUpdate, 
                                    coordinateToUpdate)
    return 2*slaterDeterminantComputeGradient(system, element, particleToUpdate)[coordinateToUpdate]
end

function wavefunction.computeDriftForceFull(system, 
                                    element::SlaterMatrix, 
                                    particleToUpdate)
    return 2*slaterDeterminantComputeGradient(system, element, particleToUpdate)
end

const quantumNumbers1D = [0
1
2
3
4
5
6
7
8
9]

const quantumNumbers2D = [0 0
1 0 
0 1 
2 0 
1 1 
0 2 
3 0 
2 1 
1 2 
0 3]

const quantumNumbers3D = [0 0 0
1 0 0
0 1 0
0 0 1
2 0 0
0 2 0
0 0 2
1 1 0
1 0 1
0 1 1]

function getQuantumNumbers(col::Int64, numDimensions::Int64)
    if numDimensions == 1
        return quantumNumbers1D[col,:]
    elseif numDimensions == 2
        return quantumNumbers2D[col,:]
    elseif numDimensions == 3
        return quantumNumbers3D[col,:]
    else 
        println("Use dim = 2 or dim = 3")
    end
end 

#End module
end