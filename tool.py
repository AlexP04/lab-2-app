#Python imports
import itertools
from concurrent import futures
from time import time
from stqdm import stqdm

#Other imports, classes will be used
from solve import Solve
from functions_to_use import *

#Function to get errors
def get_err(params):
    #Using Solve class defined - get full info about errors
    params_new = params[-1].copy()
    params_new['degrees'] = [*(params[:-1])]
    solver = Solve(params_new, function_packed = tanh_list)
    solver.run()
    
    normed_error = min(solver.norm_error)
    return (params_new['degrees'], normed_error, 0 )

#Function to get solution
def get_solution(params, pbar_container, max_deg=15):
    #Check user input and asign values
    if params['degrees'][0] == 0:
        x1_range = list(range(1, max_deg+1))
    else:
        x1_range = [params['degrees'][0]]
    
    if params['degrees'][1] == 0:
        x2_range = list(range(1, max_deg+1))
    else:
        x2_range = [params['degrees'][1]]
    
    if params['degrees'][2] == 0:
        x3_range = list(range(1, max_deg+1))
    else:
        x3_range = [params['degrees'][2]]
    
    
    ranges = list(itertools.product(x1_range, x2_range, x3_range, [params]))

    if len(ranges) > 1:
        with futures.ThreadPoolExecutor() as pool:
            results = list(stqdm(
                pool.map(get_err, ranges), 
                total=len(ranges), 
                st_container=pbar_container,
                backend=True, frontend=True))

        results.sort(key=lambda t: t[1])
    else:
        results = [get_err(ranges[0])]
    # func_runtimes = {key: [] for key in results[-1][-1].keys()}
    # for key in func_runtimes:
    #     for res in results:
    #         func_runtimes[key] += res[-1][key]
    
    #Get solution and solver object
    final_params = params.copy()
    final_params['degrees'] = results[0][0]
    solver = Solve(final_params, function_packed = tanh_list)
    
    #Run solver
    solver.run()
    
    return solver, final_params['degrees']