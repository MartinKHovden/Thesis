include("hermite.jl")

using .hermite
function prefactor(omega, n)
    return (omega/4*pi)^(1.0/4)*(1/(sqrt(factorial(n))))
end

function singleParticleHermitianGradient(particle_coordinates, nx, ny, alpha, omega)
    grad = zeros(2)

    omega_sqrt = sqrt(omega)

    x = particle_coordinates[1]
    y = particle_coordinates[2]

    H_nx = prefactor(omega, nx)*hermitePolynomial(omega_sqrt*x, nx)
    H_ny = prefactor(omega, ny)*hermitePolynomial(omega_sqrt*y, ny)

    println([H_nx, H_ny])

    H_nx_derivative = prefactor(omega, nx)*hermitePolynomialDerivative(omega_sqrt*x, nx)
    H_ny_derivative = prefactor(omega, ny)*hermitePolynomialDerivative(omega_sqrt*y, ny)

    println("grad: ",[H_nx_derivative, H_ny_derivative])


    grad[1] = omega_sqrt*H_nx_derivative*H_ny
    grad[2] = omega_sqrt*H_ny_derivative*H_nx

    return grad
end 

function singleParticleHermitianGradientV2(particle_coordinates, nx, ny, alpha, omega)
    grad = zeros(2)

    omega_sqrt = sqrt(omega)

    qN = [nx, ny]

    H = prefactor.(omega, qN).*hermitePolynomial.(omega_sqrt*particle_coordinates, qN)

    println(H)

    H_gradient = prefactor.(omega, qN).*hermitePolynomialDerivative.(omega_sqrt*particle_coordinates, qN)

    println("gradV2", H_gradient, reverse(H_gradient))
    grad = omega_sqrt*H.*reverse(H_gradient)

    return reverse(grad)
end 


println(singleParticleHermitianGradient([0.6,0.1], 1, 0, 1, 1))
println(singleParticleHermitianGradientV2([0.6,0.1], 1, 0, 1, 1))