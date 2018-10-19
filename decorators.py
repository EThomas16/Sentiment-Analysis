"""
This is a reusable library containing useful decorators including:
    -- Custom timing

History:

v1.0: Created by Erik Thomas -- September 2018
"""

import time

def timer(func_in):
    """Decorator used to print the length of time taken for a function to execute
    
    Keyword arguments:
    func_in -- the function object to be timed
    """
    def time_calc(*args, **kw):
        start_time = time.time()
        result = func_in(*args, **kw)
        end_time = time.time()
        print(f"Function: {func_in.__name__} took {end_time-start_time} seconds to execute")
        return result
    return time_calc