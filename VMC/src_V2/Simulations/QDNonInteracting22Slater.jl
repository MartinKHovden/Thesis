module QDNonInteracting22Slater

include("../MLVMC.jl")
using .MLVMC

#Set up the system:
numParticles = 2
numDimensions = 2
hamiltonian = "quantumDot" # Use "quantumDot" or "calogeroSutherland" or "bosons"
harmonicOscillatorFrequency = 1.0
interactingParticles = false



# #Set up the optimiser from Flux: 



# addWaveFunctionElement(s, Jastrow(s))
# addWaveFunctionElement(s, PadeJastrow( s; beta = 5000.0 ))
# addWaveFunctionElement(s, RBM(s, 4, 1.0))
# println(s)

learningrate = 0.1
optim = Descent(learningrate)

s = System(numParticles, 
    numDimensions, 
    hamiltonian, 
    omega=harmonicOscillatorFrequency, 
    interacting = interactingParticles)

#Add the wavefunction elements:
addWaveFunctionElement(s, SlaterMatrix( s ))
addWaveFunctionElement(s, Gaussian( 1.3 ))

for stepLength in [5.0, 0.5, 0.05, 0.005]


    @time runMetropolis!(s, 
                2^21,  
                stepLength, 
                sampler="is", 
                writeToFile = true, 
                calculateOnebody = false)
end


end