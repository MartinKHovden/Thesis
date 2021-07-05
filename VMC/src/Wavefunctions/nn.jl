module nn

struct nNet
    model 
end

function initializeModel(numInputNodes, numHiddenNodes)
    modelParams = Dict()

    for layer=1:length(numHiddenNodes)
        

end