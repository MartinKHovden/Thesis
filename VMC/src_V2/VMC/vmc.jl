module vmc 

export runVMC!

using ..metropolis
using Flux:update!

function runVMC!(system, numOptimizationIterations, numMCMCIterations, mcmcStepLength, optimizer, sampler = "bf")
    local_energies::Array{Float64, 2} = zeros(Float64, (numMCMCIterations, 1))
    for k = 1:numOptimizationIterations
        local_energy, grads = runMetropolis!(system, numMCMCIterations, mcmcStepLength, sampler = sampler)
        println("START:")
        update!(optimizer, last(system.wavefunctionElements).variationalParameter, grads)
        local_energies[k] = local_energy
        println("Iteration = ", k, "    E = ", local_energy, "E relative = ", (local_energy - 2.0)/2.0, "alpha = ", system.wavefunctionElements[2].variationalParameter)
    end
    return local_energies
end

#End module
end