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

function computeLocalEnergy(system::slaterRBM, interacting = false)
    N = system.numParticles
    numDimensions = system.numDimensions
    localEnergy = 0
    harmonicTerm = 0
    omega = system.omega
    particleCoordinates = system.particles

    for i=1:N 
        laplacianSlaterDeterminant =  slaterDeterminantComputeLaplacian(system, i) 
        gradientSlaterDeterminant = slaterDeterminantComputeGradient(system, i)

        gradientSlaterGaussian = slaterGaussianComputeGradient(system, i)
        laplacialSlaterGaussian =  slaterGaussianComputeLaplacian(system)

        gradientRBM = 0
        laplacianRBM = 0
        
        coordinates = particleCoordinates[i,:]
        r_i_squared = coordinates[1]^2 + coordinates[2]^2
        harmonicTerm += omega*omega*r_i_squared

        grad =  gradientSlaterGaussian + gradientSlaterDeterminant + gradientNN

        laplacian = laplacianSlaterDeterminant + laplacialSlaterGaussian + laplacianNN

        localEnergy += laplacian + grad[1]^2 + grad[2]^2
    end 

    return -0.5*localEnergy + 0.5*harmonicTerm
end

function computeLocalEnergy(system::slaterNN, interacting = false)
    N = system.numParticles
    numDimensions = system.numDimensions
    localEnergy = 0
    harmonicTerm = 0
    omega = system.omega
    particleCoordinates = system.particles

    fullGradientNN = nnComputeGradient(system)
    fullLaplacianNN = nnComputeLaplacian(system)

    # println("fullGrad = ", fullGradientNN, fullLaplacianNN)


    for i=1:N 
        laplacianSlaterDeterminant =  slaterDeterminantComputeLaplacian(system, i) 
        gradientSlaterDeterminant = slaterDeterminantComputeGradient(system, i)

        gradientSlaterGaussian = slaterGaussianComputeGradient(system, i)
        laplacialSlaterGaussian =  slaterGaussianComputeLaplacian(system)

        gradientNN = fullGradientNN[(i-1)*numDimensions + 1: (i-1)*numDimensions + numDimensions]
        laplacianNN = sum(fullLaplacianNN[(i-1)*numDimensions + 1: (i-1)*numDimensions + numDimensions])
        
        # println("grad = ", gradientNN, laplacianNN)

        coordinates = particleCoordinates[i,:]
        r_i_squared = coordinates[1]^2 + coordinates[2]^2
        harmonicTerm += omega*omega*r_i_squared

        grad =  gradientSlaterGaussian + gradientSlaterDeterminant + gradientNN

        laplacian = laplacianSlaterDeterminant + laplacialSlaterGaussian + laplacianNN

        localEnergy += laplacian + grad[1]^2 + grad[2]^2
    end 

    return -0.5*localEnergy + 0.5*harmonicTerm
end

function particleInteraction(system)
    interaction = 0
    numParticles = system.numParticles
    particles = system.particles
    for i=1:numParticles
        for j=i+1:numParticles
            difference = particles[i] - particles[j]
            distance = sqrt(dot(difference, difference))
            interaction += 1/distance 
        end 
    end
    return interaction 
end

# END MODULE
end 