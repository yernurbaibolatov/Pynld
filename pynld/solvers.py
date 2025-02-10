import numpy as np

def rk45step(system_func, h, t, x, params):
    k1 = h * system_func(t, x, params)
    k2 = h * system_func(t + h/2, x + k1/2, params)
    k3 = h * system_func(t + h/2, x + k2/2, params)
    k4 = h * system_func(t +h, x + k3, params)
    x_next = x + k1/6 + k2/3 + k3/3 + k4/6
    return x_next
