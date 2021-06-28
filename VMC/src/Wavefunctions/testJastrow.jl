using Test 


include("../initializeSystem.jl")
include("slaterDeterminant.jl")
include("jastrow.jl")


using .initializeSystem
using .slaterDeterminant
using .jastrow


@testset "Test ratio" begin
    system = initializeSystemSlaterJastrow(4, 2, alpha=1.0)
    println("-----------------------------------------------------------------")
    particleMoved = 1
    dimensionsMoved = 1
    for i=1:200
        oldPosition = deepcopy(system.particles)
        oldSlaterValue = slaterWavefunction(system)
        oldJastrowValue = jastrowWavefunction(system)

        system.particles[particleMoved,dimensionsMoved] += 0.01
        jastrowUpdateDistanceMatrix(system)
        newSlaterValue =  slaterWavefunction(system)
        newJastrowValue = jastrowWavefunction(system)

        ratio = jastrowComputeRatio(system, oldPosition, particleMoved)
        ratio_analytical = ((newSlaterValue^2)*(newJastrowValue^2))/((oldSlaterValue^2)*(oldJastrowValue^2))
        println("")
        println("Ratio:            ", ratio)
        println("Ratio Analytical: ", ratio_analytical)
        @test ratio ≈ ratio_analytical
    end
end

@testset "Test gradient" begin 
    system = initializeSystemSlaterJastrow(4, 2, alpha=1.0)
    println("-----------------------------------------------------------------")
    particleMoved = 1
    dimensionsMoved = 1
    for i=1:200
        oldPosition = deepcopy(system.particles)
        oldSlaterValue = slaterWavefunction(system)
        oldJastrowValue = jastrowWavefunction(system)

        system.particles[particleMoved,dimensionsMoved] += 0.01
        jastrowUpdateDistanceMatrix(system)
        newSlaterValue =  slaterWavefunction(system)
        newJastrowValue = jastrowWavefunction(system)

        ratio = jastrowComputeRatio(system, oldPosition, particleMoved)
        ratio_analytical = ((newSlaterValue^2)*(newJastrowValue^2))/((oldSlaterValue^2)*(oldJastrowValue^2))
        println("")
        println("Ratio:            ", ratio)
        println("Ratio Analytical: ", ratio_analytical)
        @test ratio ≈ ratio_analytical
    end
end 



