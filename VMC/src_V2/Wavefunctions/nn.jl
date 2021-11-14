module nn 

struct NN 
    test
    variationalParameter::Array 
    variationalParameterGradient::Array

    function NN()
        w1 = @MMatrix randn(numNodesLayer1, numParticles*numDimensions)
        w1 *= 0.01
        w2 = @MMatrix randn(numNodesLayer2, numNodesLayer1)
        w2*= 0.01
        w3 = @MMatrix randn(1, numNodesLayer2)
        w3*= 0.01
    
        b1 = @MVector randn(numNodesLayer1)
        b2 = @MVector randn(numNodesLayer2)
        b3 = @MVector randn(1)
    
        a1 = @MVector zeros(numNodesLayer1)
        a2 = @MVector zeros(numNodesLayer2)
        a3 = @MVector zeros(1)
    
        z1 = @MVector zeros(numNodesLayer1)
        z2 = @MVector zeros(numNodesLayer2)
        z3 = @MVector zeros(1)
    
        a1_grad = @MMatrix zeros(numNodesLayer1, numParticles*numDimensions)
        a2_grad = @MMatrix zeros(numNodesLayer2, numParticles*numDimensions)
        a3_grad = @MMatrix zeros(1, numParticles*numDimensions)
    
        a1_double_grad = @MMatrix zeros(numNodesLayer1, numParticles*numDimensions)
        a2_double_grad = @MMatrix zeros(numNodesLayer2, numParticles*numDimensions)
        a3_double_grad = @MMatrix zeros(1, numParticles*numDimensions)
    
        a1_double_grad_temp = @MMatrix zeros(numNodesLayer1, numParticles*numDimensions)
        a2_double_grad_temp = @MMatrix zeros(numNodesLayer2, numParticles*numDimensions)
        a3_double_grad_temp = @MMatrix zeros(1, numParticles*numDimensions)
    
        w1_grad = @MMatrix randn(numNodesLayer1, numParticles*numDimensions) 
        w2_grad = @MMatrix randn(numNodesLayer2, numNodesLayer1) 
        w3_grad = @MMatrix randn(1, numNodesLayer2) 
    
        b1_grad = @MVector randn(numNodesLayer1)
        b2_grad = @MVector randn(numNodesLayer2)
        b3_grad = @MVector randn(1)
    
        delta1 = @MVector zeros(numNodesLayer1)
        delta2 = @MVector zeros(numNodesLayer2)
        delta3 = @MVector zeros(1)
    
        return NNAnalytical(w1, w2, w3, b1, b2, b3, a1, a2, a3, z1, z2, z3, 
                                a1_grad, a2_grad, a3_grad, 
                                a1_double_grad, a2_double_grad, a3_double_grad, 
                                a1_double_grad_temp, a2_double_grad_temp, a3_double_grad_temp,
                                w1_grad, w2_grad, w3_grad,
                                b1_grad, b2_grad, b3_grad,
                                delta1, delta2, delta3,
                                activationFunction, activationFunctionDerivative, activationFunctionDoubleDerivative)
    end
end 

end