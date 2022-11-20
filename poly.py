#Python import
import numpy as np
import re

#Other packages
from solve import Solve
from basis import *
from polynom import _Polynom
    
#Class, that builds output for user of polynoms created
class Builder(object):
    ###Constructor
    def __init__(self, solution):   
        try:
            self._solution = solution
            degree = max(solution.degree) - 1

            self.basis = basis(degree, solution.polynomial_type) 

            self.a = solution.a.T.tolist()
            self.c = solution.c.T.tolist()
            self.minX = [X.min(axis=0).ravel() for X in solution.X_]
            self.maxX = [X.max(axis=0).ravel() for X in solution.X_]
            self.minY = solution.Y_.min(axis=0).ravel()
            self.maxY = solution.Y_.max(axis=0).ravel()
        except:
            raise "Construction error"
    
    #Standartize coeficients for polynom :: private
    def __standardtize__(self, c):
        std_coeffs = np.zeros(c.shape)
        for index in range(c.shape[0]):
            cp = self.basis[index].coef.copy()
            cp.resize(c.shape)
            if type(c) is np.matrix:
                std_coeffs += c[index].getA1() * cp[0]
            else:
                std_coeffs += c[index] * cp
        return std_coeffs.squeeze()
    
    #Find lamdas (lowest level aggregation) for each and every X_i to summarize functions further:: private
    def __compose_lambdas__(self):
        self.lvl1 = list()
        for i in range(self._solution.Y.shape[1]):
            current_1 = list()
            shift = 0
            for j in range(3): 
                current_2 = list()
                for k in range(self._solution.dim[j]):
                    current_3 = self._solution.L[shift:shift + self._solution.degree[j], i].ravel()
                    shift += self._solution.degree[j]
                    current_2.append(current_3)
                current_1.append(current_2)
            self.lvl1.append(current_1)
    
    #Print first-level aggregation results as a string o use further with different depths :: private
    def __print_1__(self, mode = 1, i=0, j=0, k=0):
        texts = list()
        
        if mode == 1:
            for n in range(len(self.lvl1[i][j][k])):
                texts.append(r'{0:.6f}\cdot P_{{{deg}}}(x_{{{1}{2}}})'.format(
                    self.lvl1[i][j][k][n], 
                    j+1, k+1, deg=n
                ))
                
        elif mode == 2:
            for k in range(len(self.lvl1[i][j])):
                shift = sum(self._solution.dim[:j]) + k
                for n in range(len(self.lvl1[i][j][k])):
                    texts.append(r'{0:.6f}\cdot P_{{{deg}}}(x_{{{1}{2}}})'.format(
                        self.a[i][shift] * self.lvl1[i][j][k][n],
                        j+1, k+1, deg=n
                    ))
                    
        else:
            for j in range(3):
                for k in range(len(self.lvl1[i][j])):
                    shift = sum(self._solution.dim[:j]) + k
                    for n in range(len(self.lvl1[i][j][k])):
                        texts.append(r'{0:.6f}\cdot P_{{{deg}}}(x_{{{1}{2}}})'.format(
                            self.c[i][j] * self.a[i][shift] * self.lvl1[i][j][k][n],
                            j + 1, k + 1, deg=n
                        ))
                        
        res = ' + '.join(texts).replace('+ -', ' -')
        return res

    #Prints F-function in special form :: private
    def __print_final_1__(self, i):
        texts = list()
        for j in range(3):
            for k in range(len(self.lvl1[i][j])):
                shift = sum(self._solution.dim[:j]) + k
                raw_coeffs = self.__standardtize__(self.c[i][j] * self.a[i][shift] * self.lvl1[i][j][k])
                diff = self.maxX[j][k] - self.minX[j][k]
                mult_poly = np.poly1d([1 / diff, - self.minX[j][k]] / diff)
                add_poly = np.poly1d([1])
                current_poly = np.poly1d([0])
                for n in range(len(raw_coeffs)):
                    current_poly += add_poly * raw_coeffs[n]
                    add_poly *= mult_poly
                 
                
                current_poly = current_poly * (self.maxY[i] - self.minY[i]) + self.minY[i]
                current_poly = np.poly1d(current_poly.coeffs, variable='(x_{0}{1})'.format(j+1, k+1))
                
                texts.append(str(_Polynom(
                    current_poly, 
                    symbol='(x_{0}{1})'.format(j+1, k+1),
                    subscribe='{0}{1}'.format(j+1, k+1))))
                
        res = ' + '.join(texts).replace('+ -', '- ')
        return res

    #Prints F-function in special form (just another form) :: private
    def __print_2__(self, i):
        texts = list()
        for j in range(3):
            for k in range(len(self.lvl1[i][j])):
                shift = sum(self._solution.dim[:j]) + k
                current_polynom = np.poly1d(self.__standardtize__(self.c[i][j] * self.a[i][shift] *
                                                                     self.lvl1[i][j][k])[::-1],
                                         variable='(x_{0}{1})'.format(j+1, k+1))
                texts.append(str(_Polynom(
                    current_polynom, 
                    symbol="(x_"+str(j+1)+str(k+1)+")".format(j+1, k+1),
                    subscribe='{0}{1}'.format(j+1, k+1))))
        res = ' + '.join(texts).replace('+ -', '- ')
        return res
    
    # Prints F-function in special form (just another form) :: private
    def __print_final_2__(self, i):
        res = ''
        for j in range(3):
            coef = self.c[i][j]
            if coef >= 0:
                res += f'+ {coef:.6f} \\cdot \\Phi_{{{i+1}{j+1}}} (x_{j+1}) '
            else:
                res += f'- {-coef:.6f} \\cdot \\Phi_{{{i+1}{j+1}}} (x_{j+1})'
        if self.c[i][0] >= 0:
            return res[2:-1]
        else:
            return res[:-1]
        
    # Method to get refined result, generates final string of result :: public
    def get_results(self):

        self.__compose_lambdas__()
        lvl1_texts = [r'$\Psi_{{{1}{2}}}^{{[{0}]}}(x_{{{1}{2}}}) = {result}$'.format(i+1, j+1, k+1, result=self.__print_1__(1, i, j, k)) + '\n' for i in range(self._solution.Y.shape[1]) for j in range(3) for k in range(self._solution.dim[j])]
        
        lvl2_texts = [r'$\Phi_{{{0}{1}}}(x_{{{1}}}) = {result}$'.format(i+1, j+1, result=self.__print_1__(2, i, j)) + '\n'
                       for i in range(self._solution.Y.shape[1])
                       for j in range(3)]
        f_texts = [r'$\Phi_{{{0}}}(x_1, x_2, x_3) = {result}$'.format(i + 1, result=self.__print_1__(3, i)) + '\n'
                     for i in range(self._solution.Y.shape[1])]
        f_texts_t = [r'$\Phi_{{{0}}}(x_1, x_2, x_3) = {result}$'.format(i + 1,result=self.__print_final_1__(i)) + '\n' for i in range(self._solution.Y.shape[1])]
        f_texts_td = [r'$\Phi_{{{0}}}(x_1, x_2, x_3) = {result}$'.format(
                                            i+1, result=self.__print_2__(i)) + '\n'
                                            for i in range(self._solution.Y.shape[1])]
        f_texts_l = [r'$\Phi_{i}(x_1, x_2, x_3) = {result}$'.format(i=i+1, result=self.__print_final_2__(i)) + '\n' 
                                for i in range(self._solution.Y.shape[1])]
        
        return '\n'.join(
            [r'$\Phi_{i1}(x_1)$, $\Phi_{i2}(x_2)$, $\Phi_{i3}(x_3)$:' + '\n'] + f_texts_l +
            [r'$\Phi_i$' + f'from polinom {self._solution.polynomial_type}:' + '\n'] + f_texts + 
            [r'$\Phi_i$ not normalized:' + '\n'] + f_texts_td+
            [r'$\Phi_i$ normalized:' + '\n'] + f_texts_t
        )
#             [r'Проміжні функції $\Phi$:' + '\n'] + phi_strings +
#             [r'Проміжні функції $\Psi$:' + '\n'] + psi_strings)