import numpy as np
import re
from solve import Solve
import basis as b_gen

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
    

class PolynomialBuilder(object):
    def __init__(self, solution):
        assert isinstance(solution, Solve)
        self._solution = solution
        max_degree = max(solution.p) - 1
        self.basis = b_gen.basis(degree, solution.polynomial_type) 
        if solution.poly_type == 'Chebyshev':
            self.symbol = 'T'
        elif solution.poly_type == 'Legendre':
            self.symbol = 'P'
        elif solution.poly_type == 'Laguerre':
            self.symbol = 'L'
        elif solution.poly_type == 'Hermite':
            self.symbol = 'H'

        self.a = solution.a.T.tolist()
        self.c = solution.c.T.tolist()
        self.minX = [X.min(axis=0).ravel() for X in solution.X_]
        self.maxX = [X.max(axis=0).ravel() for X in solution.X_]
        self.minY = solution.Y_.min(axis=0).ravel()
        self.maxY = solution.Y_.max(axis=0).ravel()

    def _form_lamb_lists(self):
        """
        Generates specific basis coefficients for Psi functions
        """
        self.psi = list()
        for i in range(self._solution.Y.shape[1]):  # `i` is an index for Y
            psi_i = list()
            shift = 0
            for j in range(3):  # `j` is an index to choose vector from X
                psi_i_j = list()
                for k in range(self._solution.deg[j]):  # `k` is an index for vector component
                    psi_i_j_k = self._solution.Lamb[shift:shift + self._solution.p[j], i].ravel()
                    shift += self._solution.p[j]
                    psi_i_j.append(psi_i_j_k)
                psi_i.append(psi_i_j)
            self.psi.append(psi_i)

    def _transform_to_standard(self, coeffs):
        """
        Transforms special polynomial to standard
        :param coeffs: coefficients of special polynomial
        :return: coefficients of standard polynomial
        """
        std_coeffs = np.zeros(coeffs.shape)
        for index in range(coeffs.shape[0]):
            cp = self.basis[index].coef.copy()
            cp.resize(coeffs.shape)
            if type(coeffs) is np.matrix:
                std_coeffs += coeffs[index].getA1() * cp[0]
            else:
                std_coeffs += coeffs[index] * cp
        return std_coeffs.squeeze()

    def _print_psi_i_j_k(self, i, j, k):
        """
        Returns string of Psi function in special polynomial form
        :param i: an index for Y
        :param j: an index to choose vector from X
        :param k: an index for vector component
        :return: result string
        """
        strings = list()
        for n in range(len(self.psi[i][j][k])):
            strings.append(r'{0:.6f}\cdot {symbol}_{{{deg}}}(x_{{{1}{2}}})'.format(
                self.psi[i][j][k][n], 
                j+1, k+1,symbol=self.symbol, deg=n
            ))
        res = ' + '.join(strings)
        return res.replace('+ -', '- ')

    def _print_phi_i_j(self, i, j):
        """
        Returns string of Phi function in special polynomial form
        :param i: an index for Y
        :param j: an index to choose vector from X
        :return: result string
        """
        strings = list()
        for k in range(len(self.psi[i][j])):
            shift = sum(self._solution.deg[:j]) + k
            for n in range(len(self.psi[i][j][k])):
                strings.append(r'{0:.6f}\cdot {symbol}_{{{deg}}}(x_{{{1}{2}}})'.format(
                    self.a[i][shift] * self.psi[i][j][k][n],
                    j+1, k+1, symbol=self.symbol, deg=n
                ))
        res = ' + '.join(strings)
        return res.replace('+ -', '- ')

    def _print_F_i(self, i):
        """
        Returns string of F function in special polynomial form
        :param i: an index for Y
        :return: result string
        """
        strings = list()
        for j in range(3):
            for k in range(len(self.psi[i][j])):
                shift = sum(self._solution.deg[:j]) + k
                for n in range(len(self.psi[i][j][k])):
                    strings.append(r'{0:.6f}\cdot {symbol}_{{{deg}}}(x_{{{1}{2}}})'.format(
                        self.c[i][j] * self.a[i][shift] * self.psi[i][j][k][n],
                        j + 1, k + 1, symbol=self.symbol, deg=n
                    ))
        res = ' + '.join(strings)
        return res.replace('+ -', '- ')

    def _print_F_i_transformed_denormed(self, i):
        """
        Returns string of F function in special polynomial form
        :param i: an index for Y
        :return: result string
        """
        strings = list()
        for j in range(3):
            for k in range(len(self.psi[i][j])):
                shift = sum(self._solution.deg[:j]) + k
                raw_coeffs = self._transform_to_standard(self.c[i][j] * self.a[i][shift] * self.psi[i][j][k])
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
                strings.append(str(_Polynom(
                    current_poly, 
                    symbol='(x_{0}{1})'.format(j+1, k+1),
                    subscr='{0}{1}'.format(j+1, k+1))))
        res = ' + '.join(strings)
        return res.replace('+ -', '- ')

    def _print_F_i_transformed(self, i):
        """
        Returns string of F function in special polynomial form
        :param i: an index for Y
        :return: result string
        """
        strings = list()
        for j in range(3):
            for k in range(len(self.psi[i][j])):
                shift = sum(self._solution.deg[:j]) + k
                current_poly = np.poly1d(self._transform_to_standard(self.c[i][j] * self.a[i][shift] *
                                                                     self.psi[i][j][k])[::-1],
                                         variable='(x_{0}{1})'.format(j+1, k+1))
                strings.append(str(_Polynom(
                    current_poly, 
                    symbol='(x_{0}{1})'.format(j+1, k+1),
                    subscr='{0}{1}'.format(j+1, k+1))))
        res = ' + '.join(strings)
        return res.replace('+ -', '- ')

    def _print_F_i_F_ij(self, i):
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
        """
        Generates results based on given solution
        :return: Results string
        """
        self._form_lamb_lists()
        psi_strings = [r'$\Psi_{{{1}{2}}}^{{[{0}]}}(x_{{{1}{2}}}) = {result}$'.format(i+1, j+1, k+1, result=self._print_psi_i_j_k(i, j, k)) + '\n'
                       for i in range(self._solution.Y.shape[1])
                       for j in range(3)
                       for k in range(self._solution.deg[j])]
        phi_strings = [r'$\Phi_{{{0}{1}}}(x_{{{1}}}) = {result}$'.format(i+1, j+1, result=self._print_phi_i_j(i, j)) + '\n'
                       for i in range(self._solution.Y.shape[1])
                       for j in range(3)]
        f_strings = [r'$\Phi_{{{0}}}(x_1, x_2, x_3) = {result}$'.format(i + 1, result=self._print_F_i(i)) + '\n'
                     for i in range(self._solution.Y.shape[1])]
        f_strings_transformed = [r'$\Phi_{{{0}}}(x_1, x_2, x_3) = {result}$'.format(i + 1, result=self._print_F_i_transformed(i)) + '\n'
                                 for i in range(self._solution.Y.shape[1])]
        f_strings_transformed_denormed = [r'$\Phi_{{{0}}}(x_1, x_2, x_3) = {result}$'.format(
                                            i+1, result=self._print_F_i_transformed_denormed(i)) + '\n'
                                            for i in range(self._solution.Y.shape[1])]
        f_strings_from_f_ij = [r'$\Phi_{i}(x_1, x_2, x_3) = {result}$'.format(i=i+1, result=self._print_F_i_F_ij(i)) + '\n' 
                                for i in range(self._solution.Y.shape[1])]
        
        return '\n'.join(
            [r'$\Phi_i$ через $\Phi_{i1}(x_1)$, $\Phi_{i2}(x_2)$, $\Phi_{i3}(x_3)$:' + '\n'] + f_strings_from_f_ij +
            [r'$\Phi_i$' + f'через поліноми {self._solution.poly_type}:' + '\n'] + f_strings + 
            [r'$\Phi_i$ у звичайному вигляді (нормовані):' + '\n'] + f_strings_transformed + 
            [r'$\Phi_i$ у звичайному вигляді (відновлені):' + '\n'] + f_strings_transformed_denormed + 
            [r'Проміжні функції $\Phi$:' + '\n'] + phi_strings +
            [r'Проміжні функції $\Psi$:' + '\n'] + psi_strings)