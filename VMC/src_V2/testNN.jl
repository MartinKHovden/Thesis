module testNN

include("Wavefunctions/wavefunction.jl")
include("Wavefunctions/slater.jl")
include("Wavefunctions/gaussian.jl")
include("Wavefunctions/gaussianSimple.jl")
include("Wavefunctions/jastrow.jl")
include("Wavefunctions/padeJastrow.jl")
include("Wavefunctions/rbm.jl")
include("Wavefunctions/nn.jl")
include("system.jl")
include("Hamiltonians/harmonicOscillator.jl")
include("Samplers/metropolis.jl")
include("VMC/vmc.jl")

using OrderedCollections
using Random
using Flux: Descent, ADAM, Momentum, RADAM, Dense, Chain, params, sigmoid, Params, gradient
using Zygote:hessian
using LinearAlgebra
using .wavefunction
using .slater
using .gaussian
using .gaussianSimple
using .jastrow
using .padeJastrow
using .rbm
using .nn
using .system
using .harmonicOscillator
using .metropolis
using .vmc


#Set up the system:
numParticles = 2
numDimensions = 2
hamiltonian = "quantumDot" # Use "quantumDot" or "calogeroSutherland" or "bosons"
harmonicOscillatorFrequency = 1.0
interactingParticles = true

numHidden1 = 2
numHidden2 = 2

# #Set up the optimiser from Flux: 

learningrate = 0.1
optim = ADAM(learningrate)

s = System(numParticles, 
    numDimensions, 
    hamiltonian, 
    omega=harmonicOscillatorFrequency, 
    interacting = interactingParticles)

addWaveFunctionElement(s, NN(s, numHidden1, numHidden2, "tanh"))
nnAnalytical = s.wavefunctionElements[1]
println(nnAnalytical)

w1 = nnAnalytical.variationalParameter[1]
w2 = nnAnalytical.variationalParameter[3]
w3 = nnAnalytical.variationalParameter[5]

b1 = nnAnalytical.variationalParameter[2]
b2 = nnAnalytical.variationalParameter[4]
b3 = nnAnalytical.variationalParameter[6]


# println("Params analytical")
# display(w1)
# display(w2)
# display(b1)
# display(b2)



nnFlux = Chain(Dense(numParticles*numDimensions, numHidden1, tanh), Dense(numHidden1,numHidden2, tanh), Dense(numHidden2, 1))
display(params(nnFlux))

params(nnFlux)[1][:,:] = w1
params(nnFlux)[3][:,:] = w2
params(nnFlux)[5][:] = w3
params(nnFlux)[2][:] = b1
params(nnFlux)[4][:] = b2
params(nnFlux)[6][:] = b3



# println("Params flux")
# display(params(nnFlux)[1][:,:])
# display(params(nnFlux)[3][:,:])
# display(params(nnFlux)[2][:,:])
# display(params(nnFlux)[4][:,:])



function nnFluxComputePsi(model, position)
    x = vec(reshape(position', 1,:))
    return model(x)[1]
end

function nnFluxComputeGradient(model, x)
    x = reshape(x', 1,:)'
    loss(x) = sum(model(x))
    grads = gradient(Params([x])) do 
        loss(x)
    end 
    return grads[x]
end

function nnFluxComputeLaplacian(model, x)
    x = reshape(x', 1,:)'
    loss(x) = sum(model(x))
    return diag(hessian(loss, x))
end

function nnFluxComputeParameterGradient(model, x)
    
    ps = params(model)
    x = reshape(x', 1,:)'
    loss(x) = sum(model(x))
    grads = gradient(ps) do 
        loss(x)
    end 
    for p in ps 
        display(grads[p])
    end
    return grads
end 

input = s.particles

# println("Flux = ", nnFluxComputePsi(nnFlux, input))
println("Analytic = ", computePsi!(nnAnalytical, input))

# println("Flux = ", nnFluxComputeGradient(nnFlux, input))
# println("Analytic = ", computeGradient(s, nnAnalytical))

# println("Flux = ", sum(nnFluxComputeLaplacian(nnFlux, input)))
# println("Analytic = ", computeLaplacian(s, nnAnalytical))

println("Params analytical")
display(w1)
display(w2)
display(w2)
display(b1)
display(b2)
display(b3)

println("Params flux")
display(params(nnFlux)[1][:,:])
display(params(nnFlux)[3][:,:])
display(params(nnFlux)[5][:,:])
display(params(nnFlux)[2][:,:])
display(params(nnFlux)[4][:,:])
display(params(nnFlux)[6][:,:])


println("Flux = ")
nnFluxComputeParameterGradient(nnFlux, input)
println("Analytic = ")
@time display( computeParameterGradient(s, nnAnalytical))


end