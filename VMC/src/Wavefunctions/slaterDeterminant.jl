module slaterDeterminant

export slaterWaveFunction, slaterGaussianWaveFunction
export slaterMatrixComputeRatio, slaterGaussianComputeRatio
export slaterMatrixUpdate
export inverseSlaterMatrixUpdate
export slaterDeterminantComputeGradient
export slaterDeterminantComputeLaplacian
export slaterGaussianComputeGradient
export slaterGaussianComputeLaplacian
export slaterGaussianComputeParameterGradient

include("singleParticle.jl")

using Random
using LinearAlgebra
using .singleParticle

quantumNumbers = [0 0
                1 0 
                0 1 
                2 0 
                1 1 
                0 2 
                3 0 
                2 1 
                1 2 
                0 3]

"""
    slaterWaveFunction(system)

Manual computation of the wavefunction value for the slater determinant, with
the gaussian part of the single particle wavefunction factored out. 
"""
function slaterWaveFunction(system)
    N = system.n_particles
    val = det(copy(system.slaterMatrixSpinUp))*det(copy(system.slaterMatrixSpinDown))
    return val
end 

"""
    slaterGaussianWaveFunction(system)

Manual computation of the gaussian part of the slater determinant. 
"""
function slaterGaussianWaveFunction(system)
    numParticles = system.numParticles
    numDimensions  = system.numDimensions

    exp_argument = 0

    for i=1:numParticles
        for j=1:numDimensions
            exp_argument += system.particles[i, j]^2
        end 
    end 
    return exp(-0.5*system.omega*system.alpha*exp_argument)
end 

"""
    slaterMatrixComputeRatio(system, particleMoved)

Fast computation of ratio of the slater part of the wavefunction. Works only
when one particle is moved at the time. 
"""
function slaterMatrixComputeRatio(system, particleMoved)
    if particleMoved <= system.numParticles/2
        newSlaterMatrixSpinUp = deepcopy(system.slaterMatrixSpinUp)
        oldInverseSlaterMatrixSpinUp = deepcopy(system.inverseSlaterMatrixSpinUp)
        R = dot(newSlaterMatrixSpinUp[particleMoved, :], system.inverseSlaterMatrixSpinUp[:,particleMoved])
    else 
        newSlaterMatrixSpinDown = deepcopy(system.slaterMatrixSpinDown)
        oldInverseSlaterMatrixSpinDown = deepcopy(system.inverseSlaterMatrixSpinDown)
        R = dot(newSlaterMatrixSpinDown[Int64(particleMoved - system.numParticles/2), :], system.inverseSlaterMatrixSpinDown[:, Int64(particleMoved - system.numParticles/2)])
    end 
    return R
end

"""
    slaterGaussianComputeRatio(system, oldPosition, particleMoved, dimensionMoved)

Fast computation of ratio of the gaussian part of the slater matrix. 
"""
function slaterGaussianComputeRatio(system, oldPosition, particleMoved, dimensionMoved)
    return exp(system.omega*system.alpha*(oldPosition[particleMoved,dimensionMoved]^2 - system.particles[particleMoved,dimensionMoved]^2))
end

"""
    slaterMatrixUpdate(system, particle)

Updates the slater matrix when one particle is moved. 
"""
function slaterMatrixUpdate(system, particle)
    numParticles = system.numParticles

    if particle <= numParticles/2
        slaterMatrixSpinUpUpdateRow(system, particle)
    else 
        slaterMatrixSpinDownUpdateRow(system, Int64(particle-numParticles/2))
    end
end


function slaterMatrixSpinUpUpdateRow(system, row)
    coordinates = system.particles[row, :]

    omega = system.omega
    alpha = system.alpha

    for col=1:size(system.slaterMatrixSpinUp)[2]
        nx = quantumNumbers[col,1]
        ny = quantumNumbers[col, 2]
        system.slaterMatrixSpinUp[row, col] = singleParticleHermitian(coordinates, nx, ny, alpha, omega)
    end
end 


function slaterMatrixSpinDownUpdateRow(system, row)
    coordinates = system.particles[Int(row + system.numParticles/2), :]

    omega = system.omega
    alpha = system.alpha

    for col=1:size(system.slaterMatrixSpinDown)[2]
        nx = quantumNumbers[col,1]
        ny = quantumNumbers[col, 2]
        system.slaterMatrixSpinDown[row, col] = singleParticleHermitian(coordinates, nx, ny, alpha, omega)
    end
end 


function inverseSlaterMatrixUpdate(system, col, R)
    # println(R, system.inverseSlaterMatrixSpinDown, system.inverseSlaterMatrixSpinUp)
    if col <= system.numParticles/2
        inverseSlaterMatrixSpinUpUpdateCol(system, col, R)
    else 
        inverseSlaterMatrixSpinDownUpdateCol(system, Int64(col - system.numParticles/2), R)
    end 

    return nothing
end


function inverseSlaterMatrixSpinUpUpdateCol(system, col, R)
    newSlater = deepcopy(system.slaterMatrixSpinUp)
    oldSlaterInverse = deepcopy(system.inverseSlaterMatrixSpinUp)
    N = system.numParticles/2
    for j = 1:N
        if j != col
            j=Int(j)
            S_j = dot(newSlater[col,:], oldSlaterInverse[:, j])
            value = oldSlaterInverse[:, j] - (S_j/R)*oldSlaterInverse[:, col] 
            system.inverseSlaterMatrixSpinUp[:, j] = copy(value)
        end 
    end
    value = (1/R)*oldSlaterInverse[:,col]
    system.inverseSlaterMatrixSpinUp[:, col] = copy(value)

    return nothing
end


function inverseSlaterMatrixSpinDownUpdateCol(system, col, R)
    newSlater = deepcopy(system.slaterMatrixSpinDown)
    oldSlaterInverse = deepcopy(system.inverseSlaterMatrixSpinDown)
    N = system.numParticles/2
    for j = 1:N
        if j != col
            j = Int(j)
            S_j = dot(newSlater[col, :], oldSlaterInverse[:, j])
            value = oldSlaterInverse[:, j] - (S_j/R)*oldSlaterInverse[:, col] 
            system.inverseSlaterMatrixSpinDown[:, j] = copy(value)
        end 
    end
    value = (1/R)*oldSlaterInverse[:,col]
    system.inverseSlaterMatrixSpinDown[:, col] = copy(value)

    return nothing
end


function slaterDeterminantComputeGradient(system, particle_num)
    if particle_num <= system.numParticles/2
        return slaterDeterminantSpinUpComputeGradient(system, particle_num)
    else 
        return slaterDeterminantSpinDownComputeGradient(system, particle_num)
    end 
end


function slaterDeterminantSpinUpComputeGradient(system, particle_num)
    d = system.numDimensions
    N = Int64(system.numParticles/2)
    omega = system.omega
    alpha = system.alpha
    grad = zeros(d)
    particles = system.particles
    for j=1:N
        nx = quantumNumbers[j,1]
        ny = quantumNumbers[j,2]
        grad[:] += singleParticleHermitianGradient(particles[particle_num,:], nx, ny, alpha, omega)*system.inverseSlaterMatrixSpinUp[j, particle_num]
    end
    return grad
end 


function slaterDeterminantSpinDownComputeGradient(system, particle_num)
    d = system.numDimensions
    row = Int(particle_num - system.numParticles/2)
    N = Int64(system.numParticles/2)
    omega = system.omega
    alpha = system.alpha
    grad = zeros(d)
    particles = system.particles
    for j=1:N
        nx = quantumNumbers[j, 1] 
        ny = quantumNumbers[j, 2]
        temp = singleParticleHermitianGradient(particles[particle_num,:], nx, ny, alpha, omega)*system.inverseSlaterMatrixSpinDown[j, row]
        grad += temp
    end
    return grad
end


function slaterDeterminantComputeLaplacian(system, particle_num)
    if particle_num <= system.numParticles/2
        return slaterDeterminantSpinUpComputeLaplacian(system, particle_num)
    else 
        return slaterDeterminantSpinDownComputeLaplacian(system, particle_num)
    end 
end


function slaterDeterminantSpinUpComputeLaplacian(system, particle_num)

    N = Int64(system.numParticles/2)
    omega = system.omega
    alpha = system.alpha
    laplacian = 0
    particles = system.particles
    for j=1:N
        nx = quantumNumbers[j,1]
        ny = quantumNumbers[j,2]
        laplacian += (singleParticleHermitianLaplacian(particles[particle_num,:], nx, ny, alpha, omega)*system.inverseSlaterMatrixSpinUp[j, particle_num])
    end

    temp = zeros(2)
    for j=1:N 
        nx = quantumNumbers[j,1]
        ny = quantumNumbers[j,2]

        temp += singleParticleHermitianGradient(particles[particle_num,:], nx, ny, alpha, omega)*system.inverseSlaterMatrixSpinUp[j, particle_num]
    end
    return laplacian - dot(temp, temp)
end


function slaterDeterminantSpinDownComputeLaplacian(system, particle_num)
    # d = system.numDimensions
    row = Int(particle_num - system.numParticles/2)
    N = Int64(system.numParticles/2)
    omega = system.omega
    alpha = system.alpha
    laplacian = 0
    particles = system.particles
    for j=1:N
        nx = quantumNumbers[j, 1] 
        ny = quantumNumbers[j, 2]
        laplacian += (singleParticleHermitianLaplacian(particles[particle_num,:], nx, ny, alpha, omega)*system.inverseSlaterMatrixSpinDown[j, row])
    end

    temp = zeros(2)
    for j=1:N 
        nx = quantumNumbers[j,1]
        ny = quantumNumbers[j,2]
        temp+= (singleParticleHermitianGradient(particles[particle_num,:], nx, ny, alpha, omega)*system.inverseSlaterMatrixSpinDown[j, row])
    end
    return laplacian - dot(temp, temp)
end


function slaterGaussianComputeGradient(system, particle_num)
    coordinates = system.particles[particle_num,:]

    omega = system.omega 
    alpha = system.alpha

    grad = -omega*alpha*coordinates

    return grad
end 


function slaterGaussianComputeLaplacian(system)
    d = system.numDimensions 
    alpha = system.alpha 
    omega = system.omega

    return -alpha*omega*d
end

function slaterGaussianComputeParameterGradient(system)
    return -0.5*system.omega*sum(system.particles.^2)
end 

end