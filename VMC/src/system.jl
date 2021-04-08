module system
export initializeSystem, slaterMatrixUpdate, System, slaterMatrixSpinUpUpdateRow, slaterMatrixSpinDownUpdateRow, inverseSlaterMatrixUpdate, slaterMatrixComputeRatio
export slaterGaussianComputeRatio, slaterMatrixComputeRatio


#Library for simulating fermions in a potential trap. Different methods are 
#used for representing the wavefunction elements. Currently implemented:
#    Restricted Boltzmann Machine 

# TODO: 
#- Return nothing for functions with no return value (println already returns nothing). 
#- Can use f.ex map(x -> x^2 + 2x -1 , [1,2,3])

# Author: Martin Krokan Hovden

include("Wavefunctions/hermite.jl")
include("Wavefunctions/singleParticle.jl")
include("Wavefunctions/boltzmannMachine.jl")
include("Wavefunctions/slaterDeterminant.jl")

using Random
using LinearAlgebra
using .singleParticle
using .hermite
using .boltzmannMachine
using .slaterDeterminant


function metropolisStepBruteForce(step_length, system)
    n_particles = system.n_particles 
    n_dims = system.n_dims

    # #Chooses one coordinate randomly to update.
    coordinateToUpdate::Int64 = rand(1:n_dims)
    particleToUpdate::Int64 = rand(1:n_particles)

    # #Update the coordinate:
    old_position = copy(system.particles)
    # println("Coors ", system.particles)
    # old_slater_wf = slaterWaveFunction(system)
    # old_rbm_wf = computePsi(system.nqs)
    # println(old_slater_wf)
    system.particles[particleToUpdate, coordinateToUpdate] += (rand(Float64) - 0.5)*step_length
    # system.nqs.x[:] = reshape(system.particles', 1,:)

    slaterMatrixUpdate(system, particleToUpdate)
    # new_slater_wf = slaterWaveFunction(system)
    # new_rbm_wf = computePsi(system.nqs)
    # println(new_slater_wf)
    # ratioSlaterDeterminant = (abs(new_slater_wf)^2)/(abs(old_slater_wf)^2)
    # println("\n Analytic = ", (abs(new_slater_wf)^2)/(abs(old_slater_wf)^2))

    # ratioRBM = (abs(new_rbm_wf)^2)/(abs(old_rbm_wf)^2)
    ratioSlaterDeterminant = slaterMatrixComputeRatio(system, particleToUpdate)
    # println("Fast = ", ratioSlaterDeterminant)

    # if ratioSlaterDeterminant > 10
    #     println(" Analytic = ", ratioSlaterDeterminant)
    #     # println(" Num = ", slaterMatrixComputeRatio(system, particleToUpdate))

    # end
    ratioSlaterGaussian = slaterGaussianComputeRatio(system, old_position, particleToUpdate, coordinateToUpdate)

    U = rand(Float64)

    ratio = (ratioSlaterDeterminant^2)*ratioSlaterGaussian#*ratioRBM

    if U < ratio
        # println(1)
        inverseSlaterMatrixUpdate(system, particleToUpdate, (ratioSlaterDeterminant))

        # system.inverseSlaterMatrixSpinUp[:, :] = inv(system.slaterMatrixSpinUp)
        # system.inverseSlaterMatrixSpinDown[:, :] = inv(system.slaterMatrixSpinDown)
    else 
        # println(2)
        system.particles[particleToUpdate, coordinateToUpdate] = old_position[particleToUpdate, coordinateToUpdate]
        slaterMatrixUpdate(system, particleToUpdate)
        # system.nqs.x[:] = reshape(system.particles', 1,:)
        # println(computeLocalEnergy(system))
    end
end


function jastrowComputeRatio(system, old_position, particleMoved)
    changeInDistanceSum = 0
    new_position = system.particles
    for i=1:system.n_particles
        if i != particleMoved
            new_difference = system.particles[i, :] - system.particles[particleMoved, :]
            new_distance = sqrt(dot(new_difference, new_difference))

            old_difference = old_position[i, :] - old_position[particleMoved, :]
            old_distance = sqrt(dot(old_difference, old_difference))

            changeInDistanceSum += system.beta*(new_distance - old_distance)
        end 
    end

    return exp(2*changeInDistanceSum)
end 


function rbmComputeLaplacian(system, particle_num)
    nqs = system.nqs
    n_dims = system.n_dims
    laplacian = 0
    precalc::Array{Float64, 2} = nqs.b + transpose((1.0/nqs.sigma_squared)*(transpose(nqs.x)* nqs.w))
    num_hidden = size(nqs.h)[1]


    if particle_num ==1
        start_index = 1
    else 
        start_index = n_dims*particle_num - 1
    end

    for m=start_index:(start_index + n_dims-1)
        laplacian += -1.0/nqs.sigma_squared
        for n=1:num_hidden
            laplacian += (1.0/nqs.sigma_squared^2)*(exp(precalc[n])/((exp(precalc[n])+1)^2))*(nqs.w[m,n]^2)
        end
    end 

    return laplacian
end 

function rbmComputeGradient(system, particle_num)
    nqs = system.nqs
    n_dims = system.n_dims
    gradient = zeros(n_dims)
    precalc::Array{Float64, 2} = nqs.b + transpose((1.0/nqs.sigma_squared)*(transpose(nqs.x)* nqs.w))
    num_hidden = size(nqs.h)[1]


    if particle_num ==1
        start_index = 1
    else 
        start_index = n_dims*particle_num - 1
    end

    i = 1
    for m=start_index:(start_index + n_dims-1)
        ln_psi_derivative = -(1.0/nqs.sigma_squared)*(nqs.x[m] - nqs.a[m])
        for n=1:num_hidden
            ln_psi_derivative += (1.0/nqs.sigma_squared)*nqs.w[m,n]/(exp(-precalc[n]) + 1.0)
        end
        gradient[i] = ln_psi_derivative
        i+=1
    end 
    return gradient
end 


function computeLocalEnergy(system)
    # println("START")
    N = system.n_particles
    E_L = 0
    harmonic_term = 0
    omega = system.omega
    particle_coordinates = system.particles
    # println(system.inverseSlaterMatrixSpinUp, system.inverseSlaterMatrixSpinDown)
    for i=1:N 
        laplacianSlaterDeterminant =  slaterDeterminantComputeLaplacian(system, i) # (1/sqrt(factorial(N)))*
        gradientSlaterDeterminant = slaterDeterminantComputeGradient(system, i)

        gradientSlaterGaussian = slaterGaussianComputeGradient(system, i)
        laplacialSlaterGaussian =  slaterGaussianComputeLaplacian(system)
        
        # laplacianRBM = rbmComputeLaplacian(system, i)
        # gradientRBM = rbmComputeGradient(system, i)


        coordinates = particle_coordinates[i,:]
        r_i_squared = coordinates[1]^2 + coordinates[2]^2
        harmonic_term += omega*omega*r_i_squared

        temp =  gradientSlaterGaussian + gradientSlaterDeterminant #+ gradientRBM

        # println(laplacialSlaterGaussian," ", laplacianSlaterDeterminant, gradientSlaterGaussian, gradientSlaterDeterminant, dot(temp,temp)," ", harmonic_term)

        E_L +=   laplacianSlaterDeterminant + laplacialSlaterGaussian + temp[1]^2 + temp[2]^2
        # E_L += laplacianSlaterDeterminant + laplacialSlaterGaussian + 2*dot(gradientSlaterDeterminant, gradientSlaterGaussian)

        # println(E_L)



    end 

    # println(-0.5*E_L+ 0.5*harmonic_term)

    # println("END")

    return -0.5*E_L + 0.5*harmonic_term
end 


struct System 
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

    nqs::NQS
end


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


function initializeSystem(alpha)
    n_particles = 4
    n_dims = 2

    alpha = alpha
    omega = 1.0
    beta = 1.0

    wavefunction = ["slater", "bmm"]
    
    rng = MersenneTwister(1234)
    particles = 0.05*randn(rng, Float64, (n_particles, n_dims))

    slaterMatrixSpinUp = zeros(Int(n_particles/2),  Int(n_particles/2))

    for row=1:size(slaterMatrixSpinUp)[1]
        for col=1:size(slaterMatrixSpinUp)[2]
            nx = quantumNumbers[col,1]
            ny = quantumNumbers[col, 2]
            slaterMatrixSpinUp[row, col] = singleParticleHermitian(particles[row, :], nx, ny, alpha, omega)
        end 
    end

    slaterMatrixSpinDown = zeros(Int(n_particles/2), Int(n_particles/2))

    for row=1:size(slaterMatrixSpinDown)[1]
        for col=1:size(slaterMatrixSpinDown)[2]
            nx = quantumNumbers[col,1]
            ny = quantumNumbers[col, 2]
            slaterMatrixSpinDown[row, col] = singleParticleHermitian(particles[Int(row + n_particles/2),:], nx, ny, alpha, omega)
        end 
    end

    # println(slaterMatrixSpinDown, slaterMatrixSpinUp)

    slaterMatrixInverseSpinUp = inv(slaterMatrixSpinUp)
    slaterMatrixInverseSpinDown = inv(slaterMatrixSpinDown)

    # TODO 
    nqs = setUpSystemRandomUniform(n_particles, n_dims, n_particles*n_dims, 10)
    nqs.x[:] = reshape(particles', 1,:)

    return System(particles, n_particles, n_dims, alpha, omega, beta, slaterMatrixSpinUp, slaterMatrixSpinDown, slaterMatrixInverseSpinUp, slaterMatrixInverseSpinDown, nqs)
end

function runMetropolis()
    a_vals = [0.8, 0.9, 1.0, 1.1, 1.2]
    for a in a_vals
        s = initializeSystem(a)

        # println(s.nqs.x, s.particles)

        println(a)
        N = 1000000
        E = 0
        for i=0:N
            if i>5000
                metropolisStepBruteForce(0.1, s)
                temp = computeLocalEnergy(s)
                E += temp
            end
        end
        println("Local Energy = ", E/(N-5000))
        E = 0
    end
end

function computePsiParameterDerivative(system)
    grad_a, grad_b, grad_w = computeRBMParameterDerivative(system.nqs)
    sWF = slaterWaveFunction(system)*slaterGaussianWaveFunction(system)
    return sWF*grad_a, sWF*grad_b, sWF*grad_w
end

function runMetorpolisBruteForce(system, num_mc_iterations::Int64, burn_in::Float64, step_length::Float64)

    local_energy_sum::Float64 = 0.0

    nqs = system.nqs

    #Initializes the arrays and matrices to save the derivatives and the sums.
    local_energy_psi_derivative_a_sum::Array{Float64, 2} = zeros(Float64, size(nqs.a))
    local_energy_psi_derivative_b_sum::Array{Float64, 2} = zeros(Float64, size(nqs.b))
    local_energy_psi_derivative_w_sum::Array{Float64, 2} = zeros(Float64, size(nqs.w))

    psi_derivative_a_sum::Array{Float64, 2} = zeros(Float64, size(nqs.a))
    psi_derivative_b_sum::Array{Float64, 2} = zeros(Float64, size(nqs.b))
    psi_derivative_w_sum::Array{Float64, 2} = zeros(Float64, size(nqs.w))

    #Vector to store the energies for each step.
    local_energies::Array{Float64, 1} = zeros(Float64, Int(num_mc_iterations))

    start = time()

    for i = 1:num_mc_iterations

        # Does one step with the brute force method.
        metropolisStepBruteForce(step_length, system)

        # precalc = nqs.b + transpose((1.0/nqs.sigma_squared)*(transpose(nqs.x)* nqs.w))

        # Computes the contribution to Monte carlo estimate of the local energy given the new system configuration.
        local_energy = computeLocalEnergy(system)

        local_energies[i] = local_energy

        # Computes the contribution to the gradients given the new system configuration.
        psi_derivative_a, psi_derivative_b, psi_derivative_w = computePsiParameterDerivative(system)

        # Calculates the estimates of the energy and derivatives. Uses only those after the burn-in period.
        if i > burn_in*num_mc_iterations


            local_energy_sum += local_energy

            local_energy_psi_derivative_a_sum += local_energy*psi_derivative_a
            local_energy_psi_derivative_b_sum += local_energy*psi_derivative_b
            local_energy_psi_derivative_w_sum += local_energy*psi_derivative_w

            psi_derivative_a_sum += psi_derivative_a
            psi_derivative_b_sum += psi_derivative_b
            psi_derivative_w_sum += psi_derivative_w

        end

    end

    runtime = time() - start

    # Updates the final estimates of local energy and gradients.
    samples = num_mc_iterations - burn_in*num_mc_iterations

    mc_local_energy = local_energy_sum/samples

    mc_local_energy_psi_derivative_a = local_energy_psi_derivative_a_sum/samples
    mc_local_energy_psi_derivative_b = local_energy_psi_derivative_b_sum/samples
    mc_local_energy_psi_derivative_w = local_energy_psi_derivative_w_sum/samples

    mc_psi_derivative_a = psi_derivative_a_sum/samples
    mc_psi_derivative_b = psi_derivative_b_sum/samples
    mc_psi_derivative_w = psi_derivative_w_sum/samples

    local_energy_derivative_a = 2*(mc_local_energy_psi_derivative_a - mc_local_energy*mc_psi_derivative_a)
    local_energy_derivative_b = 2*(mc_local_energy_psi_derivative_b - mc_local_energy*mc_psi_derivative_b)
    local_energy_derivative_w = 2*(mc_local_energy_psi_derivative_w - mc_local_energy*mc_psi_derivative_w)

    return mc_local_energy, local_energy_derivative_a, local_energy_derivative_b, local_energy_derivative_w

end

function runOptimizationBruteForce(system, num_iterations::Int64, num_mc_iterations::Int64, mc_burn_in::Float64, mc_step_length::Float64, learning_rate::Float64)

    local_energies::Array{Float64, 2} = zeros(Float64, (num_iterations, 1))

    # Loops for running multiple gradient descent steps.
    for k = 1:num_iterations
        local_energy,_grad_a,  _grad_b, _grad_w = runMetorpolisBruteForce(system, num_mc_iterations, mc_burn_in, mc_step_length)
        optimizationStep(system, _grad_a, _grad_b, _grad_w, learning_rate)
        local_energies[k] = local_energy
        println("Iteration = ", k, "    E = ", local_energy)

    end

    return local_energies

end

# s = initializeSystem(1.0)
# runOptimizationBruteForce(s, 100, 100000, 0.1, 0.1, 0.01)

runMetropolis()

#End module
end



















