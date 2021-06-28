module hermite

export hermitePolynomial, hermitePolynomialDerivative, hermitePolynomialDoubleDerivative

""" 
    hermite.jl 

Functions for calculating the hermite polynomials and it derivatives. 

"""

function hermitePolynomial(x, n)
    if n == 0
        return 1
    elseif n == 1
        return 2*x 
    elseif n==2
        return (4*x^2 - 2)
    elseif n==3
        return (8*x^3 - 12*x)
    else 
        println("The hermite polynomial is not implemented for n > 3")
        return 0
    end 
end

function hermitePolynomialDerivative(x, n)
    if n == 0
        return 0
    elseif n == 1
        return 2 
    elseif n==2
        return 8*x
    elseif n==3
        return (24*x^2 - 12)
    else 
        println("The hermite polynomial drivative is not implemented for n > 3")
        return 0
    end 
end

function hermitePolynomialDoubleDerivative(x, n)
    if n == 0
        return 0
    elseif n == 1
        return 0
    elseif n==2
        return 8
    elseif n==3
        return 48*x 
    else 
        println("The hermite polynomial double derivative is not implemented for n > 3")
        return 0
    end 
end

end