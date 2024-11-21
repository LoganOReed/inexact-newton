
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.sparse.linalg import gmres
import plot as p

def f(x):
    """Transcendental nonlinear function F(x)."""
    return np.array([np.sin(x[0]) + x[1] + 1,
                     np.cos(x[1]) + x[0] - 1])

def fp(x):
    """Complicated nonlinear function F(x)."""
    return np.array([[np.cos(x[0]), 1],
                     [1, -np.sin(x[1])]])

def inexact_newton(x0, eta=None, tol=1e-25, max_iter=100, eta_max = 0.7):
    """Inexact Newton using GMRES for step size"""
    x = [x0]
    residuals = []
    s = []
    if eta==None:
        eta = [eta_max / (2**n) for n in range(max_iter)]
    # eta = [eta_max]
    # eta_k = eta_max

    for k in range(max_iter):
        # Approx solution to (1.2)
        # norm(b - A @ x) <= max(rtol*norm(b), atol)
        s_k = gmres(fp(x[-1]), -f(x[-1]), rtol=eta[k])
        residuals.append(np.linalg.norm(np.matmul(fp(x[-1]),s_k[0]) + f(x[-1])))
        # print(s_k[0])
        s.append(s_k[0])
        # Generate next x_k
        x.append(x[-1] + s_k[0])
        # Generate next eta using eta_{k-1} / 2
        if eta[k] < tol:
            eta = eta[:k]
            break
            
    print(f'F: {f(x)}')
    print(f'x: {x}')
    # print(f's: {s}')
    # print(f'r: {residuals}')
    print(f'eta: {eta}')
    return x, residuals, eta


    # x = [1.,-2.]
    # testing makes this seem reasonable
    # norm(b - A @ x) <= max(rtol*norm(b), atol)
    # s_k = gmres(fp(x), -f(x), rtol=eta[k])
    # print(s_k)
    # x_k = optimize.root(f, [0,0], method='krylov', jac=fp, tol=1e-4)
    # print(x_k)
         


    

if __name__ == "__main__":
    # size of eta for now
    tol = 1e-15
    max_iter = 25
    eta_max = 0.9
    eta = [eta_max / (2**n) for n in range(max_iter)]
    eta_slow = [eta_max/(n+1) for n in range(max_iter)]
    eta_fixed = [0.6 for n in range(max_iter)]
    x_one, residuals_one, eta_one = inexact_newton(np.array([1,1]), eta=eta, tol=tol, max_iter=max_iter, eta_max = eta_max)
    x_two, residuals_two, eta_two = inexact_newton(np.array([1,1]), eta=eta_fixed, tol=tol, max_iter=max_iter, eta_max = eta_max)
    p.visualize_lines(x_one, residuals_one, eta_one, filename='main_example_fixed.png')
    p.visualize_comparison(x_one, residuals_one, eta_one, x_two, residuals_two, eta_two, filename='compare.png')

