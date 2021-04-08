module boltzmannMachine 

export NQS, setUpSystemRandomUniform, computePsi, optimizationStep, computeRBMParameterDerivative

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

function setUpSystemRandomUniform(num_particles::Int64, num_dims::Int64, M::Int64, N::Int64, sig_sq::Float64 = 0.5, inter::Bool = false)

    # Initializes the biases
    b = rand(Float64, N, 1) .-0.5
    a = rand(Float64, M, 1) .-0.5

    # Initializes the weights.
    w = rand(Float64, M, N) .-0.5

    # Initializes the visble and the hidden layer.
    x = rand(Float64, M, 1) .-0.5
    h = rand(0:1, N, 1)

    interacting = inter

    return NQS(num_particles, num_dims, b, a, w, x, h, sig_sq, interacting)

end

function computeInteractionTerm(nqs::NQS)

    interaction_term = 0

    # Loops over the outer sum
    for i = 1:nqs.num_dims:nqs.num_particles*nqs.num_dims

        # Loops over the inner sum from the current value of the outer sum
        for j = i+nqs.num_dims:nqs.num_dims:nqs.num_particles*nqs.num_dims

            r_ij = 0

            for k = 0:nqs.num_dims-1

                r_ij += (nqs.x[i + k] - nqs.x[j + k])^2

            end

            r_ij = sqrt(r_ij)

            interaction_term += 1.0/r_ij

        end

    end

    return interaction_term



end

function computeRBMParameterDerivative(nqs::NQS)

    precalc::Array{Float64, 2} = nqs.b + transpose((1.0/nqs.sigma_squared)*(transpose(nqs.x)* nqs.w))

    num_visible::Int64 = size(nqs.x)[1]
    num_hidden::Int64 = size(nqs.h)[1]

    #Computes the derivatives of psi with repsect to a
    psi_derivative_a = (1.0/nqs.sigma_squared)*(nqs.x - nqs.a)

    #Computes the derivatives of psi with respect to b
    psi_derivative_b::Array{Float64, 2} = zeros(Float64, size(nqs.b))
    
    for n = 1:num_hidden
        psi_derivative_b[n, 1] = 1.0/((exp(-precalc[n]) + 1.0))
    end

    #Computes the derivatives of psi with respect to w
    psi_derivative_w::Array{Float64, 2} = zeros(Float64, size(nqs.w))

    for n=1:num_hidden

        for m=1:num_visible

            psi_derivative_w[m,n] = nqs.x[m,1]/(nqs.sigma_squared*(exp(-precalc[n]) + 1.0))

        end

    end

    return psi_derivative_a, psi_derivative_b, psi_derivative_w

end

function optimizationStep(system, grad_a::Array{Float64, 2}, grad_b::Array{Float64, 2}, grad_w::Array{Float64, 2}, learning_rate::Float64)

    system.nqs.a[:] = system.nqs.a - learning_rate*grad_a
    system.nqs.b[:] = system.nqs.b - learning_rate*grad_b
    system.nqs.w[:,:] = system.nqs.w - learning_rate*grad_w

end

function computeLocalEnergy(nqs::NQS, precalc)

    #Extracts the number of hidden and visible units.
    num_visible = size(nqs.x)[1]
    num_hidden = size(nqs.h)[1]

    local_energy::Float64 = 0

    # Computes the local energy by looping over the visible and hidden nodes.
    for m = 1:num_visible

        # Computes the first part of the derivative and double derivative of the log of the wavefunction.
        ln_psi_derivative = -(1.0/nqs.sigma_squared)*(nqs.x[m] - nqs.a[m])
        ln_psi_double_derivative = -1.0/nqs.sigma_squared

        for n=1:num_hidden

            # Adds the rest of the derivatives that varies with the hidden layer.
            ln_psi_derivative += (1.0/nqs.sigma_squared)*nqs.w[m,n]/(exp(-precalc[n]) + 1.0)
            ln_psi_double_derivative += (1.0/nqs.sigma_squared^2)*(exp(precalc[n])/((exp(precalc[n])+1)^2))*(nqs.w[m,n]^2)

        end

        local_energy += -(ln_psi_derivative)^2 - ln_psi_double_derivative + nqs.x[m]^2

    end

    local_energy *= 0.5

    # Add the interaction term if the system should account for interactions.
    if nqs.interacting
        local_energy += computeInteractionTerm(nqs)
    end

    return local_energy

end

function computePsi(nqs::NQS)

    #Extracts the number of hidden and visible units.
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

function computeRatio()
    return 0
end 

end