import numpy as np
import pandas as pd
from datetime import datetime
from copy import deepcopy
from tabulate import tabulate as tb
from scipy import special
from openpyxl import Workbook
from scipy.sparse.linalg import cg
from sklearn.preprocessing import normalize
from basis import *

class Solve(object):
    def __init__(self, user_input):
        ###DONE
        self.dim = user_input['dimensions']
        self.name_input = user_input['input_file']
        self.name_output = user_input['output_file']
        self.degree = list(map(lambda x:x+1,user_input['degrees']))
        self.weights = user_input['weights']
        self.polynomial_type = user_input['polynomial_type']
        self.lambdas = user_input['lambda']
        self.accuracy = 0.000001
        self.norm_error = 0.0
        self.error = 0.0

    def define_data(self):
        ##DONE
        self.datas = np.fromstring(self.name_input, sep='\t').reshape(-1, sum(self.dim))
        self.n = len(self.datas)
        self.degf = [sum(self.deg[:i + 1]) for i in range(len(self.dim))]

    def __minimize_equation__(self, A, b):
        ##DONE
        if np.abs(np.linalg.det(A)) < self.accuracy:
            return cg(A, b, tol=self.accuracy)[0].reshape(-1,1)

        grad = lambda x: A @ x - b
        x = np.random.randn(len(b))
        r, h = -grad(x), -grad(x)
        for _ in range(1, len(b)+1):
            alpha = np.linalg.norm(r)**2/np.dot(A @ h, h)
            x = x + alpha * h
            beta = np.linalg.norm(r - alpha * (A @ h))**2/np.linalg.norm(r)**2
            r = r - alpha * (A @ h)
            h = r + beta * h
            
        return x.reshape(-1, 1)

    def norm_define(self):
        #DONE
        self.data = np.array(normalize(self.datas))
        
        X1 = self.data[:, :self.degf[0]]
        X2 = self.data[:, self.degf[0]:self.degf[1]]
        X3 = self.data[:, self.degf[1]:self.degf[2]]
     
        self.X = [X1, X2, X3]

        self.mX = self.degf[2]

        self.Y = self.data[:, self.degf[2]:self.degf[3]]
        self.Y_ = self.datas[:, self.degf[2]:self.degf[3]]
        self.X_ = [self.datas[:, :self.degf[0]], self.datas[:,self.degf[0]:self.degf[1]],
                   self.datas[:, self.degf[1]:self.degf[2]]]
        
    def poly_func(self):
        ##DONE
        if self.polynomial_type =='Chebyshev':
            self.poly_function = special.eval_sh_chebyt
        elif self.polynomial_type == 'Legendre':
            self.poly_function = special.eval_sh_legendre
        elif self.polynomial_type == 'Laguerre':
            self.poly_function = special.eval_laguerre
        elif self.polynomial_type == 'Hermite':
            self.poly_function = special.eval_hermite
            
    def __implement_average_for_b__():         
        return np.tile((self.Y.max(axis=1) + self.Y.min(axis=1))/2, (self.deg[3], 1)).T

    def __implement_scale_for_b__():
        return copy.deepcopy(self.Y)
    
    def implement_b(self):
        ##DONE

        if self.weights == 'Mean':
            self.b = self.__implement_average_for_b__()
        elif self.weights =='Normalized':
            self.b = self.__implement_scale_for_b__()
        else:
            raise Exception("B is not defined")

    def poly_func(self):
        ##DONE
        if self.polynomial_type =='Chebyshev':
            self.poly_function = special.eval_sh_chebyt
        elif self.polynomial_type == 'Legendre':
            self.poly_function = special.eval_sh_legendre
        elif self.polynomial_type == 'Laguerre':
            self.poly_function = special.eval_laguerre
        elif self.polynomial_type == 'Hermite':
            self.poly_function = special.eval_hermite
            
    def __get_m_for_A__():
            m = 0
            for i in range(len(self.X)):
                m += self.X[i].shape[1]*(self.degree[i]+1)
            return m

    def __get_coord_for_A__(x,deg):
        n = self.data.shape[0]
        c = np.ndarray(shape=(n,1), dtype = float)
        for i in range(n):
            c[i,0] = self.poly_function(deg, x[i])
        return c

    def __res_for_A__(X, N):
        n_1, n_2 = X.shape
        a = np.ndarray(shape=(n_1,0),dtype = float)
        for j in range(n_2):
            for i in range(N):
                ch = get_coord(X[:,j],i)
                a = np.append(a,ch,1)
        return a
    
    def implement_A(self):
        #DONE
        A = np.ndarray(shape = (self.n,0),dtype =float)
        for i in range(len(self.X)):
            vec = self.__res_for_A__(self.X[i],self.p[i])
            A = np.append(A, vec,1)

        self.A = np.array(A)

    def lambdas(self):
        l = np.ndarray(shape = (self.A.shape[1],0), dtype = float)
        for i in range(self.dim[3]):
            if self.lambdas:
                b_1 = self.p[0] * self.dim[0]
                b_2 = self.p[1] * self.dim[1] + b_1
                l_1 = self.__minimize_equation__(self.A[:, :b_1], self.b[:, i])
                l_2 = self.__minimize_equation__(self.A[:, b_1:b_2], self.b[:, i])
                l_3 = self.__minimize_equation__(self.A[:, b_2:], self.b[:, i])
                l = np.append(l, np.concatenate((l_1, l_2, l_3)), axis=1)
            else:
                l = np.append(l, self.__minimize_equation__(self.A, self.b[:, i]), axis=1)
        
        self.L = np.array(l)

    def __get_first_level_function__(l):
        psi = np.ndarray(shape=(self.n, self.mX), dtype = float)
        i_1, i_2 = 0 
        for k in range(len(self.X)): 
            for s in range(self.X[k].shape[1]):
                for i in range(self.X[k].shape[0]):
                    lvl1[i,i_1] = self.A[i,i_2:i_2+self.degree[k]] @ l[i_2:i_2+self.degree[k]]
                i_2 += self.degree[k]
                i_1 += 1
        return np.array(lvl1)
    
    def process_lvl1(self):
        #DONE
        self.lvl1 = [] 
        for i in range(self.dim[3]):
            self.lvl1.append(self.__get_first_level_function__((self.L[:,i])))
                             
    def ays(self):
        ##DONE
        self.a = np.ndarray(shape=(self.mX,0), dtype=float)
                             
        for i in range(self.deg[3]):
            a_1 = self.__minimize_equation__(self.lvl1[i][:, :self.degf[0]], self.Y[:, i])
            a_2 = self.__minimize_equation__(self.lvl1[i][:, self.degf[0]:self.degf[1]], self.Y[:, i])
            a_3 = self.__minimize_equation__(self.lvl1[i][:, self.degf[1]:], self.Y[:, i])
            self.a = np.append(self.a, np.vstack((a_1, a_2, a_3)),axis = 1)

    def __get_second_level_function__(self, lvl1, coef):
        ##DONE
        N, k = len(self.X), 0
        lvl2 = np.ndarray(shape = (self.n,N),dtype = float)

        for j in range(m): 
            for i in range(self.n): 
                lvl2[i,j] = lvl2[i,k:self.degf[j]] @ coef[k:self.degf[j]]
            k = self.degf[j]

        return np.array(lvl2)

    def process_lvl2(self):
        ##DONE
        self.lvl2 = []
        for i in range(self.deg[3]):
            self.lvl2.append(self.__get_second_level_function__(self.lvl1[i],self.a[:,i]))

    def get_coeficients(self):
        ##DONE
        self.c = np.ndarray(shape = (len(self.X),0),dtype = float)
        for i in range(self.deg[3]):
            A = self.lvl2.T @ self.lvl2
            b = self.lvl2.T @ self.Y[:,i]
                             
            if np.abs(np.linalg.det(A)) < self.accuracy:
                self.c =  np.append(self.c, cg(A, b, tol=self.accuracy)[0].reshape(-1,1), axis = 1) 
                             
            grad = lambda x: A @ x - b
            x = np.random.randn(len(b))
            r, h = -grad(x), -grad(x)
            for _ in range(1, len(b)+1):
                alpha = np.linalg.norm(r)**2/np.dot(A @ h, h)
                x = x + alpha * h
                beta = np.linalg.norm(r - alpha * (A @ h))**2/np.linalg.norm(r)**2
                r = r - alpha * (A @ h)
                h = r + beta * h
            self.c = np.append(self.c, x.reshape(-1, 1), axis = 1) 

    def process_final(self):
        ##DONE
        final = np.ndarray(self.Y.shape, dtype = float)
        for j in range(final.shape[1]):
            for i in range(final.shape[0]):
                final[i,j] = self.lvl2[j][i,:] @ self.c[:,j]
        self.final = np.array(final)
        self.norm_error = np.abs(self.Y - self.F).max(axis=0).tolist()
        
        minY = self.Y_.min(axis=0)
        maxY = self.Y_.max(axis=0)
        self.F_ = np.multiply(self.final,maxY - minY) + minY
        self.error = np.abs(self.Y_ - self.F_).max(axis=0).tolist()


                             ###DELETE AFTER CHECKS
    def save_result(self):
        wb = Workbook()
        ws = wb.active

        l = [None]

        ws.append(['X: '])
        for i in range(self.n):
             ws.append(l+self.datas[i,:self.degf[3]].tolist())
        ws.append([])
                             
        ws.append(['Normalized X:'])
        for i in range(self.n):
             ws.append(l+self.data[i,:self.degf[2]].tolist())
        ws.append([])
                             
        ws.append(['Y: '])
        for i in range(self.n):
             ws.append(l+self.datas[i,self.degf[2]:self.degf[3]].tolist())
        ws.append([])
                             
        ws.append(['Normalized Y: '])
        for i in range(self.n):
             ws.append(l+self.data[i,self.degf[2]:self.degf[3]].tolist())
        ws.append([])
                             

        ws.append(['First level matrix: '])
        for i in range(self.L.shape[0]):
             ws.append(l+self.L[i].tolist())
        ws.append([])

        for j in range(len(self.Psi)):
             s = 'First level matrix Psi%i: ' %(j+1)
             ws.append([s])
             for i in range(self.n):
                  ws.append(l+self.lvl1[j][i].tolist())
             ws.append([])
        
        for j in range(len(self.lvl2)):
             s = 'Second level matrix %i: ' %(j+1)
             ws.append([s])
             for i in range(self.Fi[j].shape[0]):
                  ws.append(l+self.Fi[j][i].tolist())
             ws.append([])

        ws.append(['A : '])
        for i in range(self.mX):
             ws.append(l+self.a[i].tolist())
        ws.append([])

        ws.append(['c : '])
        for i in range(len(self.X)):
             ws.append(l+self.c[i].tolist())
        ws.append([])
        
                             
        ws.append(['Error: '])
        ws.append(l+self.error)
                             
        ws.append(['Normalized error: '])
        ws.append(l + self.norm_error)

        wb.save(self.name_output)

#     # @profile
#     def show_streamlit(self):
#         res = []
#         res.append(('Вхідні дані',
#             pd.DataFrame(self.datas, 
#             columns = [f'X{i+1}{j+1}' for i in range(3) for j in range(self.deg[i])] + [f'Y{i+1}' for i in range(self.deg[-1])],
#             index = np.arange(1, self.n+1))
#         ))
#         res.append(('Нормовані вхідні дані',
#             pd.DataFrame(self.data, 
#             columns = [f'X{i+1}{j+1}' for i in range(3) for j in range(self.deg[i])] + [f'Y{i+1}' for i in range(self.deg[-1])],
#             index = np.arange(1, self.n+1))
#         ))

#         res.append((r'Матриця $\|\lambda\|$',
#             pd.DataFrame(self.Lamb)
#         ))
#         res.append((r'Матриця $\|a\|$',
#             pd.DataFrame(self.a)
#         ))
#         res.append((r'Матриця $\|c\|$',
#             pd.DataFrame(self.c)
#         ))

#         for j in range(len(self.Psi)):
#             res.append((r'Матриця $\|\Psi_{}\|$'.format(j+1),
#             pd.DataFrame(self.Psi[j])
#         ))
#         for j in range(len(self.Fi)):
#             res.append((r'Матриця $\|\Phi_{}\|$'.format(j+1),
#             pd.DataFrame(self.Fi[j])
#         ))
    
#         df = pd.DataFrame(self.norm_error).T
#         df.columns = np.arange(1, len(self.norm_error)+1)
#         res.append((r'Нормалізована похибка',
#             df
#         ))
#         df = pd.DataFrame(self.error).T
#         df.columns = np.arange(1, len(self.error)+1)
#         res.append((r'Похибка',
#             df
#         ))
#         return res

    # @profile
    def prepare(self):
        self.define_data()
        self.norm_define()
        self.implement_b()
        self.poly_func()
        self.implement_A()
        self.lambdas()
        self.process_lvl1()
        self.ays()
        self.process_lvl2()
        self.get_coeficients()
        self.process_final()
        self.save_result()