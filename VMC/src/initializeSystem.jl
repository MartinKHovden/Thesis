module initializeSystem 

struct slaterJastrow
    particles::Array{Float64, 2}
    n_particles::Int64 
    n_dims::Int64

    alpha::Float64 
    omega::Float64
    beta::Float64

    slaterMatrixSpinUp::Array{Float64, 2}
    slaterMatrixSpinDown::Array{Float64, 2}

    inverseSlaterMatrixSpinUp::Array{Float64, 2}
    inverseSlaterMatrixSpinDown::Array{Float64, 2}
end 

struct slaterRBM 
    particles::Array{Float64, 2}
    n_particles::Int64 
    n_dims::Int64

    alpha::Float64 
    omega::Float64
    beta::Float64

    slaterMatrixSpinUp::Array{Float64, 2}
    slaterMatrixSpinDown::Array{Float64, 2}

    inverseSlaterMatrixSpinUp::Array{Float64, 2}
    inverseSlaterMatrixSpinDown::Array{Float64, 2}

    nqs::NQS
end 

struct slaterNN 
    particles::Array{Float64, 2}
    n_particles::Int64 
    n_dims::Int64

    alpha::Float64 
    omega::Float64
    beta::Float64

    slaterMatrixSpinUp::Array{Float64, 2}
    slaterMatrixSpinDown::Array{Float64, 2}

    inverseSlaterMatrixSpinUp::Array{Float64, 2}
    inverseSlaterMatrixSpinDown::Array{Float64, 2}

    nn::NN
end 

function initializeSlater()
    return nothing 
end

function initializeSlaterJastrow()
    return nothing 
end

function initializeSlaterRBM()
    return nothing
end 

function initializeSlaterNN()
    return nothing
end

end 