
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.sparse.linalg import gmres

def f(x):
    """Transcendental nonlinear function F(x)."""
    return np.array([np.sin(x[0]) + x[1] + 1,
                     np.cos(x[1]) + x[0] - 1])

def fp(x):
    """Complicated nonlinear function F(x)."""
    return np.array([[np.cos(x[0]), 1],
                     [1, -np.sin(x[1])]])

def visualize_lines(points, residuals, etas, filename='test.png'):
    """
    Visualizes a list of 2D points [x, y] by connecting successive points with dotted lines,
    coloring points orange except the last one, which is blue, and annotating each point with its order.
    
    Parameters:
        points (list of lists): A list where each element is a list of the form [x, y].
    """
    # Unpack x and y coordinates from the points
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    
    # Plot residuals, forcing terms, and trajectory
    plt.figure(figsize=(18, 6))

    # Residuals comparison
    plt.subplot(1, 3, 1)
    plt.semilogy(residuals, marker='o', label='Dynamic eta')
    # plt.semilogy(residuals_fixed, marker='x', label='Fixed eta (0.5)')
    plt.xlabel('Iteration')
    plt.ylabel('Residual norm (log scale)')
    plt.title('Residual Norms Comparison')
    plt.legend()
    plt.grid()

    # Forcing terms comparison
    plt.subplot(1, 3, 2)
    plt.plot(etas, marker='o', label='Dynamic eta')
    # plt.plot(range(len(etas_fixed)), etas_fixed, 'r--', label='Fixed eta (0.5)')
    plt.xlabel('Iteration')
    plt.ylabel('Forcing term (eta)')
    plt.title('Forcing Terms Comparison')
    plt.legend()
    plt.grid()

    # Trajectory of x_k points comparison
    plt.subplot(1, 3, 3)
    # x_vals_dynamic = np.array(x_vals_dynamic)
    # x_vals_fixed = np.array(x_vals_fixed)
    plt.plot(x_coords, y_coords, 'o-', label='Dynamic eta')
    # plt.plot(x_coords[-1], y_coords[-1], 'x--', label='Fixed eta')
    plt.scatter(x_coords[-1], y_coords[-1], color='orange', label='Dynamic Solution', zorder=5)
    # plt.scatter([expected_solution_fixed[0]], [expected_solution_fixed[1]], color='red', label='Fixed Solution', zorder=5)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Trajectory of Iterates $x_k$ Comparison')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(filename)

    # # Create the plot
    # plt.figure(figsize=(8, 6))
    # plt.plot(x_coords, y_coords, linestyle=':', color='black', label="Path")  # Dotted line
    #
    # # Plot each point with the specified colors
    # for i, (x, y) in enumerate(points):
    #     if i == len(points) - 1:  # Last point
    #         plt.scatter(x, y, color='blue', s=100, label="Last Point" if i == len(points) - 1 else None)
    #     else:
    #         plt.scatter(x, y, color='orange', s=100)
    #     # Annotate each point with its order
    #     plt.text(x, y, str(i), fontsize=12, color='red', ha='right', va='bottom')
    #
    # # Add labels, grid, and title
    # plt.title("Line Plot with Ordered Points")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.legend(loc="upper left")
    #
    # # Display the plot   
    # # Display the plot
    # plt.savefig(filename)

def inexact_newton(x0, tol=1e-25, max_iter=100, eta_max = 0.7):
    """Inexact Newton using GMRES for step size"""
    x = [x0]
    residuals = []
    s = []
    etas = [eta_max]
    # eta_k = eta_max

    for k in range(max_iter):
        # Approx solution to (1.2)
        # norm(b - A @ x) <= max(rtol*norm(b), atol)
        s_k = gmres(fp(x[-1]), -f(x[-1]), rtol=etas[-1])
        residuals.append(np.linalg.norm(np.matmul(fp(x[-1]),s_k[0]) + f(x[-1])))
        # print(s_k[0])
        s.append(s_k[0])
        # Generate next x_k
        x.append(x[-1] + s_k[0])
        # Generate next eta using eta_{k-1} / 2
        etas.append(etas[-1] / 2)
        if etas[-1] < tol:
            break
            
    print(f'F: {f(x)}')
    print(f'x: {x}')
    # print(f's: {s}')
    # print(f'r: {residuals}')
    print(f'etas: {etas}')
    return x, residuals, etas


    # x = [1.,-2.]
    # testing makes this seem reasonable
    # norm(b - A @ x) <= max(rtol*norm(b), atol)
    # s_k = gmres(fp(x), -f(x), rtol=eta[k])
    # print(s_k)
    # x_k = optimize.root(f, [0,0], method='krylov', jac=fp, tol=1e-4)
    # print(x_k)
         


    

if __name__ == "__main__":
    x, residuals, etas = inexact_newton(np.array([0,0]),tol=1e-15, max_iter=100, eta_max = 0.7)
    visualize_lines(x, residuals, etas, filename='test.png')

