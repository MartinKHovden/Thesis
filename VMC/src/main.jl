module main

include("initializeSystem.jl")
include("Wavefunctions/slaterDeterminant.jl")
include("Wavefunctions/jastrow.jl")

include("Wavefunctions/neuralNetwork.jl")

include("Hamiltonians/harmonicOscillator.jl")
include("Samplers/metropolisBruteForce.jl")


using .initializeSystem
using .slaterDeterminant
using .jastrow
using .neuralNetwork
using .harmonicOscillator
using .metropolisBruteForce

function runSlater()
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

function runSlaterJastrow()
    alphaValues = [0.9, 1.0, 1.1]
    for alphaValue in alphaValues
        system = initializeSystemSlaterJastrow(4, 2, alpha=alphaValue)
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

# runSlaterJastrow()

function runSlaterNN()
    system = initializeSystemSlaterNN(4, 2)
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
end 

# println("Here1")

# runSlaterNN()
# system = initializeSystemSlater(4, 2, alpha=1.0)
# runVMC(system, 10, 1000000, 0.1, 0.1)


# system = initializeSystemSlaterNN(4, 2, alpha = 1.0)
# runVMC(system, 10, 100000, 0.1, 0.1)
# println("Grads: ", nnComputeParameterGradient(system))


system = initializeSystemSlaterRBM(4, 2, 4, alpha=1)

end # MODULE