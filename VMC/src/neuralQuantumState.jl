using Flux
using Zygote

# struct NNQS
#     W::Array{Float64, 2}
#     b::Array{Float64, 1}
# end

# W = rand(2,2)
# b = rand(2)

# nn = NNQS(W, b)

x = rand(10)

# y( x) = sum(W*x .+ b)

# g = gradient(()->y(x), params([W,b]))

# println("Grads: ", g[W], g[b])

model = Chain(Dense(10, 5), Dense(5, 1))

println("Output: ", model(x))
loss(x) = sum(model(x))

grads = gradient(params(model)) do 
    loss(x)
end

for p in params(model)
    println(grads[p])
end 

for i=1:10
    