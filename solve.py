#Python imports
import numpy as np
import pandas as pd
from datetime import datetime
from copy import deepcopy
from tabulate import tabulate as tb
from scipy import special
from openpyxl import Workbook
from scipy.sparse.linalg import cg

#Other imports
from basis import *
from functions_to_use import *


#Class to solve math problem, most of calculations are stored here
class Solve(object):
    ###Constructor
    def __init__(self, user_input, function_packed = []):
        #Defining accuracy, type of polynom etc. - all parameters entered
        self.accuracy = 0.000001
        self.name_input = user_input['input_file']
        self.name_output = user_input['output_file']
        self.degree = list(map(lambda x:x+1,user_input['degrees']))
        self.polynomial_type = user_input['polynomial_type']
        
        self.norm_error = 0.0
        self.error = 0.0
        self.dim = user_input['dimensions']
        self.fmode = user_input['mode']*1
        
        #functions for non-linear input
        try:
            if self.fmode == 1:
                self.f = function_packed[2]
                self.func =  function_packed[0]
                self.func_inv =  function_packed[1]
        except:
            print("Entered function is invalid")

    #Pre-process step, reading data :: public
    def define_data(self):
        self.datas = np.fromstring(self.name_input, sep='\t').reshape(-1, sum(self.dim))
        self.n = len(self.datas)
        self.degf = [sum(self.dim[:i + 1]) for i in range(len(self.dim))]
    
    #Minimizing equation using gradient descending method, fastest decline :: private
    def __minimize_equation__(self, A_input, b_input):
        A = A_input.T @ A_input
        b = A_input.T @ b_input
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
    
    #Defining X_1,X_2... (defragmentating it for future use) :: public
    def norm_define(self):
        n,m = self.datas.shape
        vec = np.ndarray(shape=(n,m),dtype=float)
        
        for j in range(m):
            minv = np.min(self.datas[:,j])
            maxv = np.max(self.datas[:,j])
            for i in range(n):
                vec[i,j] = (self.datas[i,j] - minv)/(maxv - minv)
                
        self.data = np.array(vec)

        
        X_1 = self.data[:, :self.degf[0]]
        X_2 = self.data[:, self.degf[0]:self.degf[1]]
        X_3 = self.data[:, self.degf[1]:self.degf[2]]
     
        self.X = [X_1, X_2, X_3]

        self.mX = self.degf[2]

        self.Y = self.data[:, self.degf[2]:self.degf[3]]
        
        self.Y_ = self.datas[:, self.degf[2]:self.degf[3]]
        self.X_ = [self.datas[:, :self.degf[0]], self.datas[:,self.degf[0]:self.degf[1]],
                   self.datas[:, self.degf[1]:self.degf[2]]]
    
    #Filling poly_function field based on polynom type :: public
    def poly_func(self):
        if self.polynomial_type == 'Chebyshev':
            self.poly_function = special.eval_chebyt
        elif self.polynomial_type == 'Chebyshev shifted':
            self.poly_function = lambda deg, x: special.eval_sh_chebyu(deg, x) / (deg + 1)
        
    #Initializing b :: public
    def implement_b(self):
        self.b = deepcopy(self.Y)
        if self.fmode == 1:
            self.Lb = np.log( self.f(self.b) + 1 + self.accuracy/10000)
        else:
            self.Lb = np.log( self.b + 1 + self.accuracy/10000)

    def __get_coord_for_A__(self, x, deg):
        n = self.data.shape[0]
        c = np.ndarray(shape=(n,1), dtype = float)
        for i in range(n):
            c[i,0] = self.poly_function(deg, x[i])
        return c

    def __res_for_A__(self,X, N):
        n_1, n_2 = X.shape
        a = np.ndarray(shape=(n_1,0),dtype = float)
        for j in range(n_2):
            for i in range(N):
                ch = self.__get_coord_for_A__(X[:,j],i)
                a = np.append(a,ch,1)
        return a
    
    #Initializing A :: public
    def implement_A(self):
        A = np.ndarray(shape = (self.n,0),dtype =float)
        
        for i in range(len(self.X)):
            vec = self.__res_for_A__(self.X[i],self.degree[i])
            A = np.append(A, vec,1)
            
        if self.fmode == 1:
            self.A = self.f(A) + 1 + self.accuracy/10000
        else:
            self.A = A + 1 + self.accuracy/10000
            
        self.LA = np.log(self.A)
        self.A = np.exp(self.LA)

        self.A = np.array(A)
    
    #Finding lambdas if user want (if not - then simple take from A) :: public
    def lambdas_fill(self):
        l = np.ndarray(shape = (self.A.shape[1],0), dtype = float)
        for i in range(self.dim[3]):
            l = np.append(l, self.__minimize_equation__(self.LA, self.Lb[:, i]), axis=1)
        
        self.L = np.array(l)
    
    #Getting first level functions as linear combination of x and lambdas :: public 
    def __get_first_level_function__(self, Lambda):
        lvl1 = np.ndarray(shape=(self.n, self.mX), dtype = float)
        i_1, i_2 = 0, 0 
        for k in range(len(self.X)): 
            for s in range(self.X[k].shape[1]):
                for i in range(self.X[k].shape[0]):
                    lvl1[i,i_1] = self.LA[i,i_2:i_2+self.degree[k]] @ Lambda[i_2:i_2+self.degree[k]]
                i_2 += self.degree[k]
                i_1 += 1
        return np.array(lvl1)
    
    #Processing first level for all dimensions of Y :: public
    def process_lvl1(self):
        self.Llvl1 = []
        for i in range(self.dim[3]):
            self.Llvl1.append(self.__get_first_level_function__(self.L[:,i]))
            
    
    #Defining next level coeficients :: public
    def ays(self):
        self.a = np.ndarray(shape=(self.mX,0), dtype=float)
                             
        for i in range(self.dim[3]):
            if self.fmode == 1:
                a_1 = self.__minimize_equation__(self.Llvl1[i][:, :self.degf[0]], np.log(self.f(self.Y[:, i]) + 1 + self.accuracy/10000))
                a_2 = self.__minimize_equation__(self.Llvl1[i][:, self.degf[0]:self.degf[1]],np.log(self.f(self.Y[:, i]) + 1 + self.accuracy/10000))
                a_3 = self.__minimize_equation__(self.Llvl1[i][:, self.degf[1]:], np.log(self.f(self.Y[:, i]) + 1 + self.accuracy/10000))
            else:
                a_1 = self.__minimize_equation__(self.Llvl1[i][:, :self.degf[0]], np.log(self.Y[:, i] + 1 + self.accuracy/10000))
                a_2 = self.__minimize_equation__(self.Llvl1[i][:, self.degf[0]:self.degf[1]],np.log(self.Y[:, i] + 1 + self.accuracy/10000))
                a_3 = self.__minimize_equation__(self.Llvl1[i][:, self.degf[1]:], np.log(self.Y[:, i] + 1 + self.accuracy/10000))
            self.a = np.append(self.a, np.vstack((a_1, a_2, a_3)),axis = 1)
    
    #Basicaly same as first step - just with previous functions :: private
    def __get_second_level_function__(self, lvl1, coef):
        N, k = len(self.X), 0
        lvl2 = np.ndarray(shape = (self.n,N),dtype = float)

        for j in range(N): 
            for i in range(self.n): 
                lvl2[i,j] = lvl1[i,k:self.degf[j]] @ coef[k:self.degf[j]]
            k = self.degf[j]

        return np.array(lvl2)
    
    #Same as for lvl1 - just next level :: public
    def process_lvl2(self):
        self.lvl2 = []
        self.Llvl2 = []
        for i in range(self.dim[3]):

            self.Llvl2.append(self.__get_second_level_function__(self.Llvl1[i],self.a[:,i]))
            self.lvl2.append(np.exp(self.Llvl2[-1]))
           
        self.lvl2 = np.array(self.lvl2)
        self.Llvl2 = np.array(self.Llvl2)
        
    #Getting coeficients for last level :: public
    def get_coeficients(self):
        
        self.c = np.ndarray(shape = (len(self.X),0),dtype = float)
        for i in range(self.dim[3]):
            
            if self.fmode ==1:
                J = self.Llvl2[i].T
                A = J@J.T
                b = J@ np.log(self.f(self.Y[:, i]) + 1 + self.accuracy/10000)
            else:
                J = self.Llvl2[i].T
                A = J@J.T
                b = J@ np.log(self.Y[:, i] + 1 + self.accuracy/10000)
            
            #Gradient descending
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

    #Last method to use - processing last layer :: public
    def process_final(self):
        final = np.ndarray(self.Y.shape, dtype = float)
        for j in range(final.shape[1]):
            for i in range(final.shape[0]):
                final[i,j] = self.Llvl2[j][i,:] @ self.c[:,j]
     
  
        self.final = np.exp(np.array(final)) - 1 - self.accuracy/10000
     
            
        self.norm_error = np.abs(self.Y - self.final).max(axis=0).tolist()
        
        #Defining error and scaling
        minY = self.Y_.min(axis=0)
        maxY = self.Y_.max(axis=0)
        self.final_d = np.multiply(self.final,maxY - minY) + minY
        self.error = np.abs(self.Y_ - self.final_d).max(axis=0).tolist()

    
    #Show result as a output for web page :: public
    def show(self):
        result = []
        
        #Show inputs (normed and not)
        result.append(('Inputs: ',
            pd.DataFrame(self.datas, 
            columns = [f'X{i+1}{j+1}' for i in range(3) for j in range(self.dim[i])] + [f'Y{i+1}' for i in range(self.dim[-1])],
            index = np.arange(1, self.n+1))
        ))
        
        result.append(('Normalized inputs: ',
            pd.DataFrame(self.data, 
            columns = [f'X{i+1}{j+1}' for i in range(3) for j in range(self.dim[i])] + [f'Y{i+1}' for i in range(self.dim[-1])],
            index = np.arange(1, self.n+1))
        ))
        
        #Show matrixes
        result.append((r'$\|\lambda\|$',
            pd.DataFrame(self.L)
        ))
        
        result.append((r'$\|a\|$',
            pd.DataFrame(self.a)
        ))
        
        result.append((r'$\|c\|$',
            pd.DataFrame(self.c)
        ))
        
        #Level-1 -2 
        for j in range(len(self.lvl1)):
            result.append((r' $\|\Psi_{}\|$'.format(j+1),
            pd.DataFrame(self.lvl1[j])
        ))
            
        for j in range(len(self.lvl2)):
            result.append((r' $\|\Phi_{}\|$'.format(j+1),
            pd.DataFrame(self.lvl2[j])
        ))
        
        #Show errors
        df = pd.DataFrame(self.norm_error).T
        df.columns = np.arange(1, len(self.norm_error)+1)
        result.append((r'Normalized errors',
            df
        ))
        df = pd.DataFrame(self.error).T
        df.columns = np.arange(1, len(self.error)+1)
        result.append((r'Errors',
            df
        ))
        
        return result
    
    
    #Saving result :: public
    def save_result(self):
        #Defining workbook
        wb = Workbook()
        ws = wb.active
        
        l = [None]
        
        #Write X:
        ws.append(['X: '])
        for i in range(self.n):
             ws.append(l+self.datas[i,:self.degf[2]].tolist())
        ws.append([])
        
        #Write normalized X:
        ws.append(['Normalized X:'])
        for i in range(self.n):
             ws.append(l+self.data[i,:self.degf[2]].tolist())
        ws.append([])
        
        #Write Y:
        ws.append(['Y: '])
        for i in range(self.n):
             ws.append(l+self.datas[i,self.degf[2]:self.degf[3]].tolist())
        ws.append([])
        
        #Write normed Y:
        ws.append(['Normalized Y: '])
        for i in range(self.n):
             ws.append(l+self.data[i,self.degf[2]:self.degf[3]].tolist())
        ws.append([])
                             
        #Write matrixes:
        
        #Level 1:
        for j in range(len(self.lvl1)):
            s = 'First level matrix Psi%i: ' %(j+1)
            ws.append([s])
            for i in range(self.n):
                ws.append(l+self.lvl1[j][i].tolist())
            ws.append([])
        
        #Level 2
        for j in range(len(self.lvl2)):
            s = 'Second level matrix %i: ' %(j+1)
            ws.append([s])
            for i in range(self.lvl2[j].shape[0]):
                ws.append(l+self.lvl2[j][i].tolist())
            ws.append([])
        
        #Other intermediate results used
        ws.append(['L: '])
        for i in range(self.L.shape[0]):
             ws.append(l+self.L[i].tolist())
        ws.append([])
        
        ws.append(['A : '])
        for i in range(self.mX):
             ws.append(l+self.a[i].tolist())
        ws.append([])

        ws.append(['c : '])
        for i in range(len(self.X)):
             ws.append(l+self.c[i].tolist())
        ws.append([])
        
        #Appending errors to worksheet             
        ws.append(['Error: '])
        ws.append(l+self.error)
                             
        ws.append(['Normalized error: '])
        ws.append(l + self.norm_error)
        
        #Saving
        wb.save(self.name_output)

    def run(self):
        #Full process:
        #Defining and preprocessing of data
        self.define_data()
        self.norm_define()
        self.poly_func()
        
        #Initializing A,b
        self.implement_b()
        self.implement_A()
        
        #First level
        self.lambdas_fill()
        self.process_lvl1()
        
        #Second level
        self.ays()
        self.process_lvl2()
        
        #Last calculations
        self.get_coeficients()
        self.process_final()
        
        #Saving results
        self.save_result()