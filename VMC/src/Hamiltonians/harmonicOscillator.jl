module harmonicOscillator 

export computeLocalEnergy

# include("../initializeSystem.jl")
# include("../Wavefunctions/slaterDeterminant.jl")

using ..initializeSystem
using ..slaterDeterminant
using ..jastrow
using ..neuralNetwork

# function harmonicTerm(coordinates)

function computeLocalEnergy(system::slater, interacting = false)
    N = system.numParticles
    localEnergy = 0
    harmonicTerm = 0
    omega = system.omega
    particleCoordinates = system.particles
    for i=1:N 
        laplacianSlaterDeterminant =  slaterDeterminantComputeLaplacian(system, i) 
        gradientSlaterDeterminant = slaterDeterminantComputeGradient(system, i)

        gradientSlaterGaussian = slaterGaussianComputeGradient(system, i)
        laplacialSlaterGaussian =  slaterGaussianComputeLaplacian(system)
        
        coordinates = particleCoordinates[i,:]
        r_i_squared = coordinates[1]^2 + coordinates[2]^2
        harmonicTerm += omega*omega*r_i_squared

        temp =  gradientSlaterGaussian + gradientSlaterDeterminant 

        localEnergy += laplacianSlaterDeterminant + laplacialSlaterGaussian + temp[1]^2 + temp[2]^2
    end 

    return -0.5*localEnergy + 0.5*harmonicTerm
end

function computeLocalEnergy(system::slaterJastrow, interacting = false)
    N = system.numParticles
    localEnergy = 0
    harmonicTerm = 0
    omega = system.omega
    particleCoordinates = system.particles

    for i=1:N 
        laplacianSlaterDeterminant =  slaterDeterminantComputeLaplacian(system, i) 
        gradientSlaterDeterminant = slaterDeterminantComputeGradient(system, i)

        gradientSlaterGaussian = slaterGaussianComputeGradient(system, i)
        laplacialSlaterGaussian =  slaterGaussianComputeLaplacian(system)

        gradientJastrow = jastrowComputeGradient(system, i)
        laplacianJastrow = jastrowComputeLaplacian(system, i)
        
        coordinates = particleCoordinates[i,:]
        r_i_squared = coordinates[1]^2 + coordinates[2]^2
        harmonicTerm += omega*omega*r_i_squared

        grad =  gradientSlaterGaussian + gradientSlaterDeterminant + gradientJastrow

        laplacian = laplacianSlaterDeterminant + laplacialSlaterGaussian + laplacianJastrow

        localEnergy += laplacian + grad[1]^2 + grad[2]^2
    end 

    return -0.5*localEnergy + 0.5*harmonicTerm
end

function computeLocalEnergy(system::slaterNN, interacting = false)
    N = system.numParticles
    localEnergy = 0
    harmonicTerm = 0
    omega = system.omega
    particleCoordinates = system.particles

    for i=1:N 
        laplacianSlaterDeterminant =  slaterDeterminantComputeLaplacian(system, i) 
        gradientSlaterDeterminant = slaterDeterminantComputeGradient(system, i)

        gradientSlaterGaussian = slaterGaussianComputeGradient(system, i)
        laplacialSlaterGaussian =  slaterGaussianComputeLaplacian(system)

        gradientNN = nnComputeGradient(system, i)
        laplacianNN = nnComputeLaplacian(system, i)
        
        coordinates = particleCoordinates[i,:]
        r_i_squared = coordinates[1]^2 + coordinates[2]^2
        harmonicTerm += omega*omega*r_i_squared

        grad =  gradientSlaterGaussian + gradientSlaterDeterminant + gradientNN

        laplacian = laplacianSlaterDeterminant + laplacialSlaterGaussian + laplacianNN

        localEnergy += laplacian + grad[1]^2 + grad[2]^2
    end 

    return -0.5*localEnergy + 0.5*harmonicTerm
end

# END MODULE
end 