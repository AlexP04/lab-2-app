#Python imports
import numpy as np

#Class (private) to store polynom in and 
class _Polynom(object):
    ###Constructor
    def __init__(self, coeficients, symbol='x', subscribe=None, accuracy=0.000001):
        self.coeficients = coeficients
        self.symbol = symbol
        self.subscribe = subscribe
        self.accuracy = accuracy
        
    #Show polynom as string :: private
    def __print__(self):
        result = []
        for degree, c in reversed(list(enumerate(self.coeficients))):
            
            if len(result) == 0:
                if c < 0:
                    sign = '-'
                else:
                    sign = ''
            else:
                if c < 0:
                    sign = ' - '
                else:
                    sign =  ' + '
          
            c  = abs(c)
            
            if c < self.accuracy:
                continue
                
            if c == 1 and degree != 0:
                c = ''

            f = {0: '{}{:f}', 1: '{}{:f}'+self.symbol}.get(degree, '{}{:f}' + self.symbol + '^{{{}}}')
            res = f.format(sign, c, degree)
            res = res.replace(self.symbol, r' x_{{{}}}'.format(self.subscribe))
            result.append(res)
            
        return ''.join(result)