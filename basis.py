#Python imports
from numpy.polynomial import Polynomial as pm

#Defining basis choosing function (from mode choosen)
def basis(degree, mode):
    mode = mode.lower()
    
    if mode == 'test':
        #Free block
        raise "Ignore"
        
    elif mode == 'chebyshev':
        basis = [pm([-1, 2]), pm([1])]
        for i in range(degree):
            basis.append(pm([-2, 4])*basis[-1] - basis[-2])
        del basis[0]
        
    elif mode == 'legendre':
        basis = [pm([1]),pm([-1, 2])]
        for i in range(1, degree):
            basis.append((pm([-2*i - 1, 4*i + 2])*basis[-1] - i * basis[-2]) / (i + 1))
            
    elif mode == 'laguerre':
        basis = [pm([1]), pm([1, -1])]
        for i in range(1, degree):
            basis.append(pm([2*i + 1, -1])*basis[-1] - i * i * basis[-2])
                             
    elif mode == 'hermite':
        basis = [pm([0]), pm([1])]
        for i in range(degree):
            basis.append(pm([0,2])*basis[-1] - 2 * i * basis[-2])
        del basis[0]
                             
    else:
        #Exception if type is not matching
        raise ValueError("Mode "+str(mode)+" is not allowed")
    
    return basis