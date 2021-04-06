include("system.jl")
using .system
using Test
using Random
using LinearAlgebra

function testInitializeSystem()
    s = initializeSystem(1.1)
    println("Particles = ", s.particles)
    println("Slater spin up = ", s.slaterMatrixSpinUp)
    println("Slater spin down = ", s.slaterMatrixSpinDown)
    println("Inverse slater spin up = ", s.inverseSlaterMatrixSpinUp)
    println("Inverse slater spin down = ", s.inverseSlaterMatrixSpinDown)
end

function testInverseFast()
    s = initializeSystem(1.1)
    println("Coors old = ", s.particles)
    println("Slater up = ", s.slaterMatrixSpinUp)
    println("Inverse up = ", s.inverseSlaterMatrixSpinUp)
    println("Slater down = ", s.slaterMatrixSpinDown)
    println("Inverse down = ", s.inverseSlaterMatrixSpinDown)
    old_position = copy(s.particles)
    particle_to_update = 2
    coordinate_to_update = 1
    s.particles[particle_to_update,coordinate_to_update] += (rand(Float64)-0.5)*0.1
    println("Coors new = ", s.particles)
    slaterMatrixUpdate(s, particle_to_update)
    println("Inv analytic up =", inv(s.slaterMatrixSpinUp))
    println("Inv analytic down = ", inv(s.slaterMatrixSpinDown))
    ratioSlater = slaterMatrixComputeRatio(s, particle_to_update)
    inverseSlaterMatrixUpdate(s, particle_to_update, ratioSlater)
    println("Slater up = ", s.slaterMatrixSpinUp)
    println("Inverse up = ", s.inverseSlaterMatrixSpinUp)
    println("Slater down = ", s.slaterMatrixSpinDown)
    println("Inverse down = ", s.inverseSlaterMatrixSpinDown)
    println("Up: ", isapprox(inv(s.slaterMatrixSpinUp), s.inverseSlaterMatrixSpinUp ), "  Down: ", isapprox(inv(s.slaterMatrixSpinDown), s.inverseSlaterMatrixSpinDown))
end

function gaussianWaveFunction(alpha, omega, particles)
    N = size(particles)[1]
    println(N)
    temp = 0
    for i=1:N 
        temp += -alpha*omega*(particles[i,1]^2 + particles[i,2]^2)
    end
    return exp(0.5*temp)
end 

function testSlaterGaussianRatio()
    alpha = 1.0
    s = initializeSystem(alpha)
    old_position = copy(s.particles)
    particle_to_update = 2
    coordinate_to_update = 1
    old_wf = gaussianWaveFunction(alpha, 1, s.particles)
    s.particles[particle_to_update,coordinate_to_update] += (rand(Float64)-0.5)
    new_wf = gaussianWaveFunction(alpha, 1, s.particles)
    

    println(slaterGaussianComputeRatio(s, old_position, particle_to_update, coordinate_to_update))
    println((abs(new_wf)^2)/(abs(old_wf)^2))

end 

function slaterWaveFunction(system)
    N = system.n_particles
    return det(system.slaterMatrixSpinUp)*det(system.slaterMatrixSpinDown)
end 

function testSlaterDeterminantRatio()
    alpha = 1.1
    s = initializeSystem(alpha)
    old_position = copy(s.particles)
    particle_to_update = 2
    coordinate_to_update = 1
    old_wf = slaterWaveFunction(s)
    s.particles[particle_to_update,coordinate_to_update] += (rand(Float64)-0.5)
    slaterMatrixUpdate(s, particle_to_update)
    new_wf = slaterWaveFunction(s)
    

    println(slaterMatrixComputeRatio(s, particle_to_update))
    println((abs(new_wf)^2)/(abs(old_wf)^2))
end

# testInitializeSystem()
testInverseFast()
# testSlaterGaussianRatio()
# testSlaterDeterminantRatio()

