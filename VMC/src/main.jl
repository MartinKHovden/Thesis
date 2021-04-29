module main

include("initializeSystem.jl")
include("Wavefunctions/slaterDeterminant.jl")
include("Samplers/metropolisBruteForce.jl")
include("Hamiltonians/harmonicOscillator.jl")

using .initializeSystem
using .slaterDeterminant
using .metropolisBruteForce
using .harmonicOscillator

function run()
    alphaValues = [0.9, 1.0, 1.1]
    for alphaValue in alphaValues
        system = initializeSystemSlater(4, 2, alpha=alphaValue)
        numSamples = 1000000
        localEnergy = 0
        for i=0:numSamples
            if i>5000
                metropolisStepBruteForce(0.1, system)
                temp = harmonicOscillator.computeLocalEnergy(system)
                localEnergy += temp
            end
        end
        println("Local Energy = ", localEnergy/(numSamples-5000))
        localEnergy = 0
    end
end

run()

end