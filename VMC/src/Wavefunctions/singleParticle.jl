module singleParticle

include("hermite.jl")
using .hermite

export singleParticleHermitian, singleParticleHermitianGradient, singleParticleHermitianLaplacian

function prefactor(omega, n)
    return (omega/(4*pi))^(1.0/4)*(1.0/(sqrt(factorial(n)*(2^n))))
end

function singleParticleHermitian(particle_coordinates, qN::Array{Int64, 1}, omega)
    omegaSqrt = sqrt(omega)
    H = prod(@. prefactor(omega,qN)*hermitePolynomial(omegaSqrt*particle_coordinates, qN))
    return H
end

function singleParticleHermitian(particle_coordinates, qN::Array{Int64, 1}, omega, H)
    omegaSqrt = sqrt(omega)
    H = prod(@. prefactor(omega,qN)*hermitePolynomial(omegaSqrt*particle_coordinates, qN))
end

function singleParticleHermitianGradient(particle_coordinates::Array{Float64,1}, qN::Array{Int64, 1}, omega)
    numDimensions = length(qN)
    omegaSqrt = sqrt(omega)
    H = @. prefactor(omega, qN)*hermitePolynomial(omegaSqrt*particle_coordinates, qN)

    H_gradient = @. prefactor(omega, qN)*hermitePolynomialDerivative(omegaSqrt*particle_coordinates, qN)

    grad = zeros(numDimensions)
    for i=1:numDimensions
        grad[i] = H_gradient[i]
        for j=1:numDimensions
            if i!=j
                grad[i] *= H[j]
            end 
        end 
    end
    grad*=omegaSqrt
    return grad
end 

function singleParticleHermitianLaplacian(particle_coordinates::Array{Float64,1}, qN::Array{Int64,1}, omega)
    numDimensions = length(qN)
    omega_sqrt = sqrt(omega)
    H = @. prefactor(omega, qN)*hermitePolynomial(omega_sqrt*particle_coordinates, qN)

    H_gradient = @. prefactor(omega, qN)*hermitePolynomialDerivative(omega_sqrt*particle_coordinates, qN)

    H_doubleDerivative = @. prefactor(omega, qN)*hermitePolynomialDoubleDerivative(omega_sqrt*particle_coordinates, qN)

    double_grads = zeros(numDimensions)
    for i=1:numDimensions
        double_grads[i] = H_doubleDerivative[i]
        for j=1:numDimensions
            if i!=j
                double_grads[i] *= H[j]
            end 
        end 
    end
    double_grads*=omega^2
    return sum(double_grads)*omega^2
end

end