module harmonicOscillator

export computeLocalEnergy

using LinearAlgebra
using ..initializeSystem
using ..slaterDeterminant
using ..jastrow
using ..boltzmannMachine
using ..neuralNetwork

function computeLocalEnergy(system::slater, interacting = false)
    N = system.numParticles
    localEnergy = 0
    harmonicTerm = 0
    omega = system.omega
    particleCoordinates = system.particles
    for i=1:N 
        laplacianSlaterDeterminant = slaterDeterminantComputeLaplacian(system, i) 
        gradientSlaterDeterminant = slaterDeterminantComputeGradient(system, i)

        gradientSlaterGaussian = slaterGaussianComputeGradient(system, i)
        laplacialSlaterGaussian = slaterGaussianComputeLaplacian(system)
        
        coordinates = particleCoordinates[i,:]
        r_i_squared = sum(coordinates.^2)#coordinates[1]^2 + coordinates[2]^2
        harmonicTerm += omega*omega*r_i_squared

        grad =  gradientSlaterGaussian + gradientSlaterDeterminant 

        localEnergy += laplacianSlaterDeterminant + laplacialSlaterGaussian + sum(grad.^2)#grad[1]^2 + grad[2]^2
    end
    
    interactionTerm = 0

    if interacting
        # println("Here")
        interactionTerm = computeParticleInteraction(system)
    end

    return -0.5*localEnergy + 0.5*harmonicTerm + interactionTerm
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

        gradientSlaterGaussian =  slaterGaussianComputeGradient(system, i)
        laplacialSlaterGaussian =  slaterGaussianComputeLaplacian(system)

        gradientJastrow = jastrowComputeGradient(system, i)
        laplacianJastrow = jastrowComputeLaplacian(system, i)
        
        coordinates = particleCoordinates[i,:]
        r_i_squared = sum(coordinates.^2)
        harmonicTerm += omega*omega*r_i_squared

        grad =  gradientSlaterGaussian + gradientSlaterDeterminant + gradientJastrow

        laplacian = laplacianSlaterDeterminant + laplacialSlaterGaussian + laplacianJastrow

        localEnergy += laplacian + sum(grad.^2)
    end 

    interactionTerm = 0

    if interacting
        interactionTerm += computeParticleInteraction(system)
    end

    return -0.5*localEnergy + 0.5*harmonicTerm + interactionTerm
end

function computeLocalEnergy(system::slaterRBM, interacting = false)
    N = system.numParticles
    numDimensions = system.numDimensions
    localEnergy = 0
    harmonicTerm = 0
    omega = system.omega
    particleCoordinates = system.particles

    fullGradientRBM = rbmComputeGradient(system)
    fullLaplacianRBM = rbmComputeLaplacian(system)

    for i=1:N 
        laplacianSlaterDeterminant =  slaterDeterminantComputeLaplacian(system, i)
        gradientSlaterDeterminant = slaterDeterminantComputeGradient(system, i)

        # gradientSlaterGaussian =  slaterGaussianComputeGradient(system, i)
        # laplacialSlaterGaussian =  slaterGaussianComputeLaplacian(system)

        gradientRBM = fullGradientRBM[(i-1)*numDimensions + 1: (i-1)*numDimensions + numDimensions]
        laplacianRBM = sum(fullLaplacianRBM[(i-1)*numDimensions + 1: (i-1)*numDimensions + numDimensions])
        
        coordinates = particleCoordinates[i,:]
        r_i_squared = sum(coordinates.^2)
        harmonicTerm += omega*omega*r_i_squared

        grad =  gradientSlaterDeterminant + gradientRBM  # + gradientSlaterGaussian

        laplacian = laplacianSlaterDeterminant + laplacianRBM  #+ laplacialSlaterGaussian

        localEnergy += laplacian + sum(grad.^2)
    end 

    interactionTerm = 0

    if interacting
        interactionTerm += computeParticleInteraction(system)
    end

    return -0.5*localEnergy + 0.5*harmonicTerm + interactionTerm
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

    for i=1:N 
        laplacianSlaterDeterminant =  slaterDeterminantComputeLaplacian(system, i) 
        gradientSlaterDeterminant = slaterDeterminantComputeGradient(system, i)

        gradientSlaterGaussian = slaterGaussianComputeGradient(system, i)
        laplacialSlaterGaussian =  slaterGaussianComputeLaplacian(system)

        gradientNN = fullGradientNN[(i-1)*numDimensions + 1: (i-1)*numDimensions + numDimensions]
        laplacianNN = sum(fullLaplacianNN[(i-1)*numDimensions + 1: (i-1)*numDimensions + numDimensions])
        
        coordinates = particleCoordinates[i,:]
        r_i_squared = sum(coordinates.^2)
        harmonicTerm += omega*omega*r_i_squared

        grad =   gradientSlaterDeterminant + gradientNN + gradientSlaterGaussian 

        laplacian = laplacianSlaterDeterminant  + laplacianNN  + laplacialSlaterGaussian

        localEnergy += laplacian + sum(grad.^2)
    end
    
    interactionTerm = 0

    if interacting
        interactionTerm += computeParticleInteraction(system)
    end

    return -0.5*localEnergy + 0.5*harmonicTerm + interactionTerm
end

function computeParticleInteraction(system)
    interaction = 0
    numParticles = system.numParticles
    particles = system.particles
    for i=1:numParticles
        for j=i+1:numParticles
            difference = particles[i,:] - particles[j,:]
            distance = sqrt(dot(difference, difference))
            interaction += 1.0/distance 
        end 
    end
    return interaction 
end

# END MODULE
end 