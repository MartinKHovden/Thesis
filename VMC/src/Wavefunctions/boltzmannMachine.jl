module boltzmannMachine 

export NQS, setUpSystemRandomUniform, computePsi, optimizationStep, computeRBMParameterDerivative
export rbmComputeGradient, rbmComputeLaplacian, rbmComputeRatio, rbmComputeParameterGradient!

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

    precalc =  nqs.b + transpose((1.0/sigmaSquared)*(transpose(x)* nqs.w))

    firstTerm = -1/sigmaSquared

    for m=1:numVisible
        secondTerm = 0
        for n=1:numHidden 
            secondTerm += ((w[m,n]^2)/(sigmaSquared^2))*(exp(precalc[n])/((exp(precalc[n])+1)^2))
        end 
        doubleGrads[m] = firstTerm + (1/(sigmaSquared^2))*secondTerm
    end 

    return doubleGrads
end 

function rbmComputeParameterGradient!(system, psi_derivative_a, psi_derivative_b, psi_derivative_w, precalc::Array{Float64, 2})
    nqs = system.nqs
    x = reshape(system.particles', 1,:)'

    num_visible::Int64 = size(x)[1]
    num_hidden::Int64 = size(nqs.h)[1]

    psi_derivative_parameter_a = (1.0/nqs.sigma_squared)*(x - nqs.a)

    for n = 1:num_hidden
        psi_derivative_b[n, 1] = 1.0/((exp(-precalc[n]) + 1.0))
    end

    for n=1:num_hidden
        for m=1:num_visible
            psi_derivative_w[m,n] = x[m,1]/(nqs.sigma_squared*(exp(-precalc[n]) + 1.0))
        end
    end

end

function computePsi(nqs::NQS)
    num_visible = nqs.num_particles*nqs.num_dims
    num_hidden = size(nqs.h)[1]

    precalc::Array{Float64, 2} = nqs.b + transpose((1.0/nqs.sigma_squared)*(transpose(nqs.x)* nqs.w))


    exp_argument = 0

    for i=1:num_visible
        exp_argument += (nqs.x[i] - nqs.a[i])^2
    end

    exp_argument /= (2*nqs.sigma_squared)

    prod_term = 1.0

    for j=1:num_hidden
        prod_term *= (1.0 + exp(precalc[j]))
    end

    return exp(-exp_argument)*prod_term
end

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

function rbmComputeRatio(system, oldPosition)
    oldWavefunctionValue = rbmComputePsi(system, oldPosition)
    newWavefunctionValue = rbmComputePsi(system, system.particles)
    ratio = (newWavefunctionValue^2)/(oldWavefunctionValue^2)
    return ratio
end 

# function computeInteractionTerm(nqs::NQS)
    #     interaction_term = 0
    
    #     for i = 1:nqs.num_dims:nqs.num_particles*nqs.num_dims
    #         for j = i+nqs.num_dims:nqs.num_dims:nqs.num_particles*nqs.num_dims
    #             r_ij = 0
    #             for k = 0:nqs.num_dims-1
    #                 r_ij += (nqs.x[i + k] - nqs.x[j + k])^2
    #             end
    #             r_ij = sqrt(r_ij)
    #             interaction_term += 1.0/r_ij
    #         end
    #     end
    
    #     return interaction_term
    # end

# function setUpSystemRandomUniform(position, num_particles::Int64, num_dims::Int64, M::Int64, N::Int64, sig_sq::Float64 = 0.5, inter::Bool = false)

#     # Initializes the biases
#     b = rand(Float64, N, 1) .-0.5
#     a = rand(Float64, M, 1) .-0.5

#     # Initializes the weights.
#     w = rand(Float64, M, N) .-0.5

#     # Initializes the visble and the hidden layer.
#     x = reshape(position', 1,:)'
#     h = rand(0:1, N, 1)

#     interacting = inter

#     return NQS(num_particles, num_dims, b, a, w, x, h, sig_sq, interacting)

# end

# function computeLocalEnergy(nqs::NQS, precalc)

#     #Extracts the number of hidden and visible units.
#     num_visible = size(nqs.x)[1]
#     num_hidden = size(nqs.h)[1]

#     local_energy::Float64 = 0

#     # Computes the local energy by looping over the visible and hidden nodes.
#     for m = 1:num_visible
#         # Computes the first part of the derivative and double derivative of the log of the wavefunction.
#         ln_psi_derivative = -(1.0/nqs.sigma_squared)*(nqs.x[m] - nqs.a[m])
#         ln_psi_double_derivative = -1.0/nqs.sigma_squared
#         for n=1:num_hidden
#             # Adds the rest of the derivatives that varies with the hidden layer.
#             ln_psi_derivative += (1.0/nqs.sigma_squared)*nqs.w[m,n]/(exp(-precalc[n]) + 1.0)
#             ln_psi_double_derivative += (1.0/nqs.sigma_squared^2)*(exp(precalc[n])/((exp(precalc[n])+1)^2))*(nqs.w[m,n]^2)
#         end
#         local_energy += -(ln_psi_derivative)^2 - ln_psi_double_derivative + nqs.x[m]^2
#     end

#     local_energy *= 0.5

#     # Add the interaction term if the system should account for interactions.
#     if nqs.interacting
#         local_energy += computeInteractionTerm(nqs)
#     end

#     return local_energy
# end

# function rbmComputeParameterGradient(system)
#     nqs = system.nqs

#     precalc::Array{Float64, 2} = nqs.b + transpose((1.0/nqs.sigma_squared)*(transpose(nqs.x)* nqs.w))

#     num_visible::Int64 = size(nqs.x)[1]
#     num_hidden::Int64 = size(nqs.h)[1]

#     #Computes the derivatives of psi with repsect to a
#     psi_derivative_a = (1.0/nqs.sigma_squared)*(nqs.x - nqs.a)

#     #Computes the derivatives of psi with respect to b
#     psi_derivative_b::Array{Float64, 2} = zeros(Float64, size(nqs.b))
    
#     for n = 1:num_hidden
#         psi_derivative_b[n, 1] = 1.0/((exp(-precalc[n]) + 1.0))
#     end

#     #Computes the derivatives of psi with respect to w
#     psi_derivative_w::Array{Float64, 2} = zeros(Float64, size(nqs.w))

#     for n=1:num_hidden
#         for m=1:num_visible
#             psi_derivative_w[m,n] = nqs.x[m,1]/(nqs.sigma_squared*(exp(-precalc[n]) + 1.0))
#         end
#     end

#     return psi_derivative_a, psi_derivative_b, psi_derivative_w
# end

# function optimizationStep(system, grad_a::Array{Float64, 2}, grad_b::Array{Float64, 2}, grad_w::Array{Float64, 2}, learning_rate::Float64)
#     system.nqs.a[:] = system.nqs.a - learning_rate*grad_a
#     system.nqs.b[:] = system.nqs.b - learning_rate*grad_b
#     system.nqs.w[:,:] = system.nqs.w - learning_rate*grad_w
# end

end