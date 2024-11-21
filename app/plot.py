import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.sparse.linalg import gmres


def visualize_lines(points, residuals, eta, filename='test.png'):
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
    plt.plot(eta, marker='o', label='Dynamic eta')
    # plt.plot(range(len(eta_fixed)), eta_fixed, 'r--', label='Fixed eta (0.5)')
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


def visualize_comparison(x_one, residuals_one, eta_one, x_two, residuals_two, eta_two, filename='test_compare.png'):
    """docstring for compare_two_etas"""
    # Plot residuals, forcing terms, and trajectory
    plt.figure(figsize=(18, 6))

    # Residuals comparison
    plt.subplot(1, 3, 1)
    plt.semilogy(residuals_one, marker='o', label='First Residuals')
    plt.semilogy(residuals_two, 'ro--', label='Second Residuals')
    # plt.semilogy(residuals_fixed, marker='x', label='Fixed eta (0.5)')
    plt.xlabel('Iteration')
    plt.ylabel('Residual norm (log scale)')
    plt.title('Residual Norms Comparison')
    plt.legend()
    plt.grid()



    # Forcing terms comparison
    plt.subplot(1, 3, 2)
    plt.plot(eta_one, marker='o', label='First eta')
    plt.plot(eta_two, 'r--', label='Second eta')
    plt.xlabel('Iteration')
    plt.ylabel('Forcing term (eta)')
    plt.title('Forcing Terms Comparison')
    plt.legend()
    plt.grid()

    x_coords1 = [point[0] for point in x_one]
    y_coords1 = [point[1] for point in x_one]

    x_coords2 = [point[0] for point in x_two]
    y_coords2 = [point[1] for point in x_two]
    # Trajectory of x_k points comparison
    plt.subplot(1, 3, 3)
    # x_vals_dynamic = np.array(x_vals_dynamic)
    # x_vals_fixed = np.array(x_vals_fixed)
    # plt.plot(x_one, 'o-', label='First Points')
    plt.plot(x_coords1, y_coords1, 'o-', label='First Points')
    plt.plot(x_coords2, y_coords2, 'o-', color='orange', label='Second Points')
    # plt.plot(x_two, 'o-', color='orange', label='Second Points')
    # plt.plot(x_coords[-1], y_coords[-1], 'x--', label='Fixed eta')
    plt.scatter(x_coords1[-1], y_coords1[-1], color='green', label='First Points Solution', zorder=5)
    plt.scatter(x_coords1[-1], y_coords1[-1], color='red', label='Second Points Solution', zorder=5)
    # plt.scatter(x_one[0][-1], x_one[1][-1], color='green', label='First Points Solution', zorder=5)
    # plt.scatter(x_two[0][-1], x_two[1][-1], color='red', label='Second Points Solution', zorder=5)
    # plt.scatter([expected_solution_fixed[0]], [expected_solution_fixed[1]], color='red', label='Fixed Solution', zorder=5)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Trajectory of Iterates $x_k$ Comparison')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(filename)


    

