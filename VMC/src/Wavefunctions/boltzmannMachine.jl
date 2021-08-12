module boltzmannMachine 

export NQS, setUpSystemRandomUniform, computePsi, optimizationStep, computeRBMParameterDerivative
export rbmComputeGradient, rbmComputeLaplacian, rbmComputeRatio, rbmComputeParameterGradient!
export rbmComputeDriftForce

"""
    NQS 

Struct for storing the information about the Resticted Boltzmann machine.
"""
struct NQS
    num_particles::Int64
    num_dims::Int64

    # Bias for hidden layer
    b::Array{Float64, 2}
    # Bias for visible layer
    a::Array{Float64, 2}

    # Weights
    w::Array{Float64, 2}

    # Visible layer
    x::Array{Float64, 2}
    # Hidden layer
    h::Array{Float64, 2}

    sigma_squared::Float64
    interacting::Bool
end

"""
    rbmComputeGradient(system)

Computes the gradient of the RBM with respect to coordinates.
"""
function rbmComputeGradient(system)
    numDimensions = system.numDimensions
    numParticles = system.numParticles
    numVisible = numDimensions*numParticles
    numHidden = size(system.nqs.h)[1]
    sigmaSquared = system.nqs.sigma_squared

    particles = system.particles
    x = reshape(particles', 1,:)'

    nqs = system.nqs
    a = nqs.a
    w = nqs.w

    grads = zeros(numVisible)

    precalc =  nqs.b + transpose((1.0/sigmaSquared)*(transpose(x)* nqs.w))

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

""" 
    rbmComputeDriftForce(system)

Computes the drift force used in importance sampling. 
"""
function rbmComputeDriftForce(system, particleToUpdate, coordinateToUpdate)
    m = particleToUpdate
    nqs = system.nqs

    x = reshape(system.particles', 1,:)'

    precalc = nqs.b + transpose((1.0/nqs.sigma_squared)*(transpose(x)* nqs.w))

    #Extracts the number of hidden and visible units.
    num_visible = size(system.nqs.x)[1]
    num_hidden = size(system.nqs.h)[1]

    #Calculates the first term in the drift force.
    drift_force = -(1.0/nqs.sigma_squared)*(nqs.x[m]- nqs.a[m])

    #Calculates the second term in the drift force.
    for n=1:num_hidden
        drift_force += (1.0/nqs.sigma_squared)*nqs.w[m,n]/(exp(-precalc[n]) + 1.0)
    end

    drift_force*=2.0

    return drift_force
    # return 2*rbmComputeGradient(system)[(particleToUpdate - 1)*numDimensions + coordinateToUpdate]
end

"""
    rbmComputeLaplacian(system)

Computes the laplacian of the RBM with respect to coordinates.
"""
function rbmComputeLaplacian(system)
    numDimensions = system.numDimensions
    numParticles = system.numParticles
    numVisible = numDimensions*numParticles
    numHidden = size(system.nqs.h)[1]
    sigmaSquared = system.nqs.sigma_squared

    particles = system.particles
    x = reshape(particles', 1,:)'

    nqs = system.nqs
    a = nqs.a
    w = nqs.w

    doubleGrads = zeros(numVisible)

    precalc =  nqs.b + transpose((1.0/sigmaSquared)*(transpose(x) * nqs.w))

    firstTerm = -1/sigmaSquared

    for m=1:numVisible
        secondTerm = 0
        for n=1:numHidden 
            secondTerm += ((w[m,n]^2)/(sigmaSquared^2))*(exp(precalc[n])/((exp(precalc[n])+1)^2))
        end 
        doubleGrads[m] = firstTerm + secondTerm#(1/(sigmaSquared^2))*secondTerm
    end 


    return doubleGrads
end 

"""
    rbmComputeParameterGradient!(system, psi_derivative_a, psi_derivative_b, psi_derivative_w, precalc::Array{Float64, 2})

Computes the gradient of the RBM with respect to the variational parameters. Updates
the matrices in-place to save space and time. 
"""
function rbmComputeParameterGradient!(system, psi_derivative_a, psi_derivative_b, psi_derivative_w, precalc::Array{Float64, 2})
    nqs = system.nqs
    x = reshape(system.particles', 1,:)'

    num_visible::Int64 = size(x)[1]
    num_hidden::Int64 = size(nqs.h)[1]

    psi_derivative_a[:,1] = (1.0/nqs.sigma_squared)*(x - nqs.a)

    for n = 1:num_hidden
        psi_derivative_b[n, 1] =  1.0/((exp(-precalc[n]) + 1.0)) 
    end

    for n=1:num_hidden
        for m=1:num_visible
            psi_derivative_w[m,n] =  x[m,1]/(nqs.sigma_squared*(exp(-precalc[n]) + 1.0)) 
        end
    end
end

"""
    rbmComputePsi(system, x)

Computes the value of the RBM wavefunction given the coordinates x of the particles.
"""
function rbmComputePsi(system, x)
    nqs = system.nqs
    x = reshape(x', 1,:)'

    num_visible = nqs.num_particles*nqs.num_dims
    num_hidden = size(nqs.h)[1]

    precalc::Array{Float64, 2} = nqs.b + transpose((1.0/nqs.sigma_squared)*(transpose(x)* nqs.w))

    exp_argument = 0

    for i=1:num_visible
        exp_argument += (x[i] - nqs.a[i])^2
    end

    exp_argument /= (2*nqs.sigma_squared)

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
function rbmComputeRatio(system, oldPosition)
    oldWavefunctionValue = rbmComputePsi(system, oldPosition)
    newWavefunctionValue = rbmComputePsi(system, system.particles)
    ratio = (newWavefunctionValue^2)/(oldWavefunctionValue^2)
    return ratio
end 

end