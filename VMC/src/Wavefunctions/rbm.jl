module rbm 

export RBM, computeRatio, computeGradient

using ..wavefunction

mutable struct RBM 
    numParticles::Int64
    numDimensions::Int64

    variationalParameter::Array
    variationalParameterGradient::Array

    # Visible layer
    x::Array{Float64, 2}
    # Hidden layer
    h::Array{Float64, 2}

    sigma_squared::Float64

    function RBM(system, numHiddenNodes, sigmaSquared)
        numParticles = system.numParticles
        numDimensions = system.numDimensions
        numVisibleNodes = numParticles*numDimensions

        # Initializes the biases
        a = randn(Float64, numVisibleNodes, 1)*0.05     # Visible layer bias
        b = randn(Float64, numHiddenNodes, 1)*0.05      # Hidden layer bias

        # Initializes the weights.
        w = randn(Float64, numVisibleNodes, numHiddenNodes)*0.05 

        # Initializes the hidden layer.
        h = rand(0:1, numHiddenNodes, 1)

        variationalParameter = []
        push!(variationalParameter, a)
        push!(variationalParameter, b)
        push!(variationalParameter, w)

        variationalParameterGradient = 0. *variationalParameter

        return new(numParticles, 
                numDimensions, 
                variationalParameter, 
                variationalParameterGradient, 
                system.particles, 
                h, 
                sigmaSquared)

    end
end

function wavefunction.computeRatio(system, 
                                wavefunctionElement::RBM, 
                                particleToUpdate, 
                                coordinateToUpdate, 
                                oldPosition)
    ratio = rbmComputeRatio(system, wavefunctionElement, oldPosition)
    return ratio
end

function rbmComputePsi(system, rbm::RBM, x)
    x = reshape(x', 1,:)'

    num_visible = rbm.numParticles*rbm.numDimensions
    num_hidden = size(rbm.h)[1]

    variationalParameter = rbm.variationalParameter

    a = variationalParameter[1]
    b = variationalParameter[2]
    w = variationalParameter[3]

    precalc::Array{Float64, 2} = b + transpose((1.0/rbm.sigma_squared)*(transpose(x)* w))

    exp_argument = 0

    for i=1:num_visible
        exp_argument += (x[i] - a[i])^2
    end

    exp_argument /= (2*rbm.sigma_squared)

    prod_term = 1.0

    for j=1:num_hidden
        prod_term *= (1.0 + exp(precalc[j]))
    end

    return exp(-exp_argument)*prod_term
end

""" 
    rbmComputeRatio(system, oldPosition)

Computes the ratio given the old position.
"""
function rbmComputeRatio(system, rbm::RBM, oldPosition)
    oldWavefunctionValue = rbmComputePsi(system, rbm, oldPosition)
    newWavefunctionValue = rbmComputePsi(system, rbm, system.particles)
    ratio = (newWavefunctionValue^2)/(oldWavefunctionValue^2)
    return ratio
end 

function wavefunction.computeGradient(system, wavefunctionElement::RBM)
    return rbmComputeGradient(system, wavefunctionElement)
end

function rbmComputeGradient(system, rbm::RBM)
    numDimensions = system.numDimensions
    numParticles = system.numParticles
    numVisible = numDimensions*numParticles
    numHidden = size(rbm.h)[1]
    sigmaSquared = rbm.sigma_squared

    particles = system.particles
    x = reshape(particles', 1,:)'

    variationalParameter = rbm.variationalParameter
    a = variationalParameter[1]
    b = variationalParameter[2]
    w = variationalParameter[3]

    grads = zeros(numVisible)

    precalc =  b + transpose((1.0/sigmaSquared)*(transpose(x)* w))

    firstTerm = -(1/(sigmaSquared))*(x - a)

    for m=1:numVisible
        secondTerm = 0
        for n=1:numHidden 
            secondTerm += w[m, n]/(exp(-precalc[n])+1)
        end 
        grads[m] = firstTerm[m] + (1/sigmaSquared)*secondTerm
    end 

    return grads
end

function wavefunction.computeLaplacian(system, wavefunctionElement::RBM)
    return sum(rbmComputeLaplacian(system, wavefunctionElement))
end

function rbmComputeLaplacian(system, rbm::RBM)
    numDimensions = system.numDimensions
    numParticles = system.numParticles
    numVisible = numDimensions*numParticles
    numHidden = size(rbm.h)[1]
    sigmaSquared = rbm.sigma_squared

    particles = system.particles
    x = reshape(particles', 1,:)'

    variationalParameter = rbm.variationalParameter
    a = variationalParameter[1]
    b = variationalParameter[2]
    w = variationalParameter[3]

    doubleGrads = zeros(numVisible)

    precalc =  b + transpose((1.0/sigmaSquared)*(transpose(x) * w))

    firstTerm = -1/sigmaSquared

    for m=1:numVisible
        secondTerm = 0
        for n=1:numHidden 
            secondTerm += ((w[m,n]^2)/(sigmaSquared^2))*(exp(precalc[n])/((exp(precalc[n])+1)^2))
        end 
        doubleGrads[m] = firstTerm + secondTerm
    end 
    return doubleGrads
end 

function wavefunction.computeParameterGradient(system, wavefunctionElement::RBM)
    rbmComputeParameterGradient!(system, wavefunctionElement)
    return wavefunctionElement.variationalParameterGradient
end

function rbmComputeParameterGradient!(system, rbm::RBM)
    x = reshape(system.particles', 1,:)'

    num_visible::Int64 = size(x)[1]
    num_hidden::Int64 = size(rbm.h)[1]

    variationalParameter = rbm.variationalParameter
    a = variationalParameter[1]
    b = variationalParameter[2]
    w = variationalParameter[3]

    precalc =  b + transpose((1.0/rbm.sigma_squared)*(transpose(x) * w))

    rbm.variationalParameterGradient[1][:,1] = (1.0/rbm.sigma_squared)*(x - a)

    for n = 1:num_hidden
        rbm.variationalParameterGradient[2][n, 1] =  1.0/((exp(-precalc[n]) + 1.0)) 
    end

    for n=1:num_hidden
        for m=1:num_visible
            rbm.variationalParameterGradient[3][m,n] =  x[m,1]/(rbm.sigma_squared*(exp(-precalc[n]) + 1.0)) 
        end
    end
end

function wavefunction.computeDriftForce(system, element::RBM, particleToUpdate, coordinateToUpdate)
    return 2*rbmComputeGradient(system, element)[(particleToUpdate - 1)*system.numDimensions + coordinateToUpdate]
end

function wavefunction.computeDriftForceFull(system, element::RBM, particleToUpdate)
    return 2*rbmComputeGradient(system, element)
end

function wavefunction.updateElement!(system, wavefunctionElement::RBM, particle)
end
end