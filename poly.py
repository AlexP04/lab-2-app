import numpy as np
import re
from solve import Solve
from basis import *

#DONE
class _Polynom(object):
    def __init__(self, coeficients, symbol='x', subscribe=None, accuracy=0.000001):
        self.coeficients = coeficients
        self.symbol = symbol
        self.subscribe = subscribe
        self.accuracy = accuracy

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
    
    

class Builder(object):
    def __init__(self, solution):
        #DONE
        
        assert isinstance(solution, Solve)
        
        self._solution = solution
        degree = max(solution.p) - 1
        
        self.basis = basis(degree, mode = solution.poly_type)  
        
        self.a = solution.a.T.tolist()
        self.c = solution.c.T.tolist()
        self.minX = [X.min(axis=0).ravel() for X in solution.X_]
        self.maxX = [X.max(axis=0).ravel() for X in solution.X_]
        self.minY = solution.Y_.min(axis=0).ravel()
        self.maxY = solution.Y_.max(axis=0).ravel()

    def __compose_lambas__(self):
        self.lvl1 = list()
        for i in range(self._solution.Y.shape[1]):
            current_1 = list()
            shift = 0
            for j in range(3): 
                current_2 = list()
                for k in range(self._solution.deg[j]):
                    current_3 = self._solution.L[shift:shift + self._solution.degree[j], i].ravel()
                    shift += self._solution.degree[j]
                    current_3.append(psi_i_j_k)
                current_2.append(current_3)
            self.lvl1.append(current_1)

    def __standardtize__(self, c):
        for index in range(c.shape[0]):
            cp = self.basis[index].coef.copy()
            cp.resize(coeffs.shape)
            if type(coeffs) is np.matrix:
                std_coeffs += coeffs[index].getA1() * cp[0]
            else:
                std_coeffs += coeffs[index] * cp
        return std_coeffs.squeeze()

    def __print_1__(self, mode = 1, i=0, j=0, k=0):
        texts = list()
        
        if mode == 1:
            for n in range(len(self.lvl1[i][j][k])):
                texts.append(r'{0:.6f}\cdot P_{{{deg}}}(x_{{{1}{2}}})'.format(
                    self.lvl1[i][j][k][n], 
                    j+1, k+1, deg=n
                ))
        elif mode == 2:
            for k in range(len(self.psi[i][j])):
                shift = sum(self._solution.dim[:j]) + k
                for n in range(len(self.lvl1[i][j][k])):
                    strings.append(r'{0:.6f}\cdot P_{{{deg}}}(x_{{{1}{2}}})'.format(
                        self.a[i][shift] * self.psi[i][j][k][n],
                        j+1, k+1, deg=n
                    ))
        else:
            for j in range(3):
                for k in range(len(self.lvl1[i][j])):
                    shift = sum(self._solution.dim[:j]) + k
                    for n in range(len(self.lvl1[i][j][k])):
                        strings.append(r'{0:.6f}\cdot P_{{{deg}}}(x_{{{1}{2}}})'.format(
                            self.c[i][j] * self.a[i][shift] * self.lvl1[i][j][k][n],
                            j + 1, k + 1, deg=n
                        ))
                        
        res = ' + '.join(texts)
        return res.replace('+ -', ' -')

    def __print_final_1__(self, i):
        texts = list()
        for j in range(3):
            for k in range(len(self.lvl1[i][j])):
                shift = sum(self._solution.dim[:j]) + k
                raw_coeffs = self._transform_to_standard(self.c[i][j] * self.a[i][shift] * self.lvl1[i][j][k])
                diff = self.maxX[j][k] - self.minX[j][k]
                mult_poly = np.poly1d([1 / diff, - self.minX[j][k]] / diff)
                add_poly = np.poly1d([1])
                current_poly = np.poly1d([0])
                for n in range(len(raw_coeffs)):
                    current_poly += add_poly * raw_coeffs[n]
                    add_poly *= mult_poly
                    # print(current_poly)
                    # print(add_poly)
                current_poly = current_poly * (self.maxY[i] - self.minY[i]) + self.minY[i]
                # print(current_poly)
                # print(i, j, k)
                current_poly = np.poly1d(current_poly.coeffs, variable='(x_{0}{1})'.format(j+1, k+1))
                texts.append(str(_Polynom(
                    current_poly, 
                    symbol='(x_{0}{1})'.format(j+1, k+1),
                    subscr='{0}{1}'.format(j+1, k+1))))
        res = ' + '.join(texts).replace('+ -', '- ')
        return res

    
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
                    subscr='{0}{1}'.format(j+1, k+1))))
        res = ' + '.join(texts).replace('+ -', '- ')
        return res

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

    def get_results(self):

        self.__compose_lambas__()
        lvl1_texts = [r'$\Psi_{{{1}{2}}}^{{[{0}]}}(x_{{{1}{2}}}) = {result}$'.format(i+1, j+1, k+1, result=self.__print_1__(3, i, j, k)) + '\n' for i in range(self._solution.Y.shape[1]) for j in range(3) for k in range(self._solution.dim[j])]
        lvl2_texts = [r'$\Phi_{{{0}{1}}}(x_{{{1}}}) = {result}$'.format(i+1, j+1, result=self.__print_1__(i, j)) + '\n'
                       for i in range(self._solution.Y.shape[1])
                       for j in range(3)]
        f_texts = [r'$\Phi_{{{0}}}(x_1, x_2, x_3) = {result}$'.format(i + 1, result=self.__print_final_2__(i)) + '\n'
                     for i in range(self._solution.Y.shape[1])]
        f_texts_t = [r'$\Phi_{{{0}}}(x_1, x_2, x_3) = {result}$'.format(i + 1,result=self.__print_2__(i)) + '\n' for i in range(self._solution.Y.shape[1])]
        f_texts_td = [r'$\Phi_{{{0}}}(x_1, x_2, x_3) = {result}$'.format(
                                            i+1, result=self.__print_2__(i)) + '\n'
                                            for i in range(self._solution.Y.shape[1])]
        f_strings_l = [r'$\Phi_{i}(x_1, x_2, x_3) = {result}$'.format(i=i+1, result=self._print_F_i_F_ij(i)) + '\n' 
                                for i in range(self._solution.Y.shape[1])]
        
        return '\n'.join(
            [r'$\Phi_{i1}(x_1)$, $\Phi_{i2}(x_2)$, $\Phi_{i3}(x_3)$:' + '\n'] + f_strings_l +
            [r'$\Phi_i$' + f'from polunom {self._solution.polynom_type}:' + '\n'] + f_texts + 
            [r'$\Phi_i$ not normalized:' + '\n'] + f_texts_td
            [r'$\Phi_i$ normalized:' + '\n'] + f_texts_t )
#             [r'Проміжні функції $\Phi$:' + '\n'] + phi_strings +
#             [r'Проміжні функції $\Psi$:' + '\n'] + psi_strings)