using Flux

f(x) = 3*x^2 + 2*x + 1;

df(x) = gradient(f, x)[1];

println(df(2))

W1 = rand(3,5)
b1 = rand(3)

layer1(x) = W1*x .+ b1 

W2 = rand(2,3)
b2 = rand(2)
layer2(x) = W2*x .+ b2 

model(x) = layer2(sigmoid.(layer1(x)))

model(rand(5))

# function loss(x, y)
#     y_hat = model(x) 
# end
    

df = gradient(model, W2)

function neuralWaveFunction(system)
    return 0
end