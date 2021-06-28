module singleParticle

include("hermite.jl")
using .hermite

export singleParticleHermitian, singleParticleHermitianGradient, singleParticleHermitianLaplacian

function prefactor(omega, n)
    return (omega/4*pi)^(1.0/4)*(1/(sqrt(factorial(n))))
end

# function singleParticleHermitian(particle_coordinates, nx, ny, alpha, omega)
#     omega_sqrt = sqrt(omega)

#     x = particle_coordinates[1]
#     y = particle_coordinates[2]

#     H_nx = prefactor(omega, nx)*hermitePolynomial(omega_sqrt*x, nx)
#     H_ny = prefactor(omega, ny)*hermitePolynomial(omega_sqrt*y, ny)

#     return H_nx*H_ny
# end

function singleParticleHermitian(particle_coordinates, nx, ny, alpha, omega)
    omega_sqrt = sqrt(omega)

    qN = [nx, ny]

    H = prod(prefactor.(omega,qN).*hermitePolynomial.(omega_sqrt*particle_coordinates, qN))

    return H
end

# function singleParticleHermitianGradient(particle_coordinates, nx, ny, alpha, omega)
#     grad = zeros(2)

#     omega_sqrt = sqrt(omega)

#     x = particle_coordinates[1]
#     y = particle_coordinates[2]

#     H_nx = prefactor(omega, nx)*hermitePolynomial(omega_sqrt*x, nx)
#     H_ny = prefactor(omega, ny)*hermitePolynomial(omega_sqrt*y, ny)

#     H_nx_derivative = prefactor(omega, nx)*hermitePolynomialDerivative(omega_sqrt*x, nx)
#     H_ny_derivative = prefactor(omega, ny)*hermitePolynomialDerivative(omega_sqrt*y, ny)


#     grad[1] = omega_sqrt*H_nx_derivative*H_ny
#     grad[2] = omega_sqrt*H_ny_derivative*H_nx

#     return grad
# end 

function singleParticleHermitianGradient(particle_coordinates, nx, ny, alpha, omega)
    grad = zeros(2)

    omega_sqrt = sqrt(omega)

    qN = [nx, ny]

    H = prefactor.(omega, qN).*hermitePolynomial(omega_sqrt*particle_coordinates, qN)

    H_gradient = prefactor.(omega, qN)*hermitePolynomialDerivative(omega_sqrt*particle_coordinates, qN)

    grad = omega_sqrt*H.*reverse(H_gradient)

    return grad
end 

function singleParticleHermitianLaplacian(particle_coordinates, nx, ny, alpha, omega)
    omega_sqrt = sqrt(omega)

    x = particle_coordinates[1]
    y = particle_coordinates[2]

    H_nx = prefactor(omega, nx)*hermitePolynomial(omega_sqrt*x, nx)
    H_ny = prefactor(omega, ny)*hermitePolynomial(omega_sqrt*y, ny) 
    
    H_nx_derivative = prefactor(omega, nx)*hermitePolynomialDerivative(omega_sqrt*x, nx)
    H_ny_derivative = prefactor(omega, ny)*hermitePolynomialDerivative(omega_sqrt*y, ny)

    H_nx_doubleDerivative = prefactor(omega, nx)*hermitePolynomialDoubleDerivative(omega_sqrt*x, nx)
    H_ny_doubleDerivative = prefactor(omega, ny)*hermitePolynomialDoubleDerivative(omega_sqrt*y, ny)

    return H_nx_doubleDerivative*H_ny*omega^2 + H_ny_doubleDerivative*H_nx*omega^2

end

end