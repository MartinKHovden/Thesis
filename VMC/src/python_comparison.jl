using Random

function wavefunction(r, alpha, beta)
    r1 = r[1,1]^2 + r[1,2]^2
    r2 = r[2,1]^2 + r[2,2]^2 
    r12 = sqrt((r[1,1] - r[2,1])^2 + (r[1,2] - r[2,2])^2)
    deno = r12/(1+beta*r12)
    return exp(-0.5*alpha*(r1+r2) + deno)
end 

function localEnergy(r, alpha, beta)
    r1 = r[1,1]^2 + r[1,2]^2
    r2 = r[2,1]^2 + r[2,2]^2 
    r12 = sqrt((r[1,1] - r[2,1])^2 + (r[1,2] - r[2,2])^2)
    deno = r12/(1+beta*r12)
    deno2 = deno*deno
    return 0.5*(1-alpha*alpha)*(r1 + r2) + 2.0*alpha + 1.0/r12 + deno2*(alpha*r12 - deno2 + 2*beta*deno - 1.0/r12)
end
function runVMC(numMCCycles, stepSize, alpha, beta)
    rng = MersenneTwister(230)
    positionOld::Array{Float64, 2} = randn(rng, Float64, (2,2))
    positionNew::Array{Float64, 2} = zeros(Float64, (2,2))

    wfOld = wavefunction(positionOld, alpha, beta)

    energy = 0

    deltaE = 0.0


    for i=1:numMCCycles
        for particle=1:2
            for dim=1:2
                positionNew[particle, dim] = positionOld[particle, dim] + stepSize*(rand() - 0.5)
            end
        end
        wfNew = wavefunction(positionNew, alpha, beta)
        if rand() < wfNew^2/wfOld^2
            positionOld = copy(positionNew)
            wfOld = wfNew
            deltaE = localEnergy(positionOld, alpha, beta)
        end
        energy += deltaE
    end


    energy /= numMCCycles

    println(energy)
    return energy 
end

@time runVMC(10^4,0.5, 1.0, 1.0)
@time runVMC(10^6,0.5, 1.0, 1.0)



