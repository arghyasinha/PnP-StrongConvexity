# Import necessary modules and functions
from denoisers.NLM import *  # Non-local means denoiser
from PIL import *  # Image handling (PIL stands for Python Imaging Library)
from iterative_algorithms import *  # Iterative algorithms for reconstruction
from forward_models import *  # Forward models such as inpainting, blurring, etc.
import matplotlib.pyplot as plt  # Plotting library for visualizing results
import pickle  # To save and load objects such as eigenvalues




"""
There are three command line arguments: forward_model, image_id, and sigma in [0,255].
Example usage: python Figure_1_Strong_Convexity_index.py inpainting 1 120
"""


# Take the forward model and image ID as command line input
import sys
forward_model = sys.argv[1]  # First argument is the forward model (e.g., inpainting, deblurring)
image_id = sys.argv[2]  # Second argument is the image identifier

# Display the chosen forward model and image
print("chosen forward model: ", forward_model)
print("chosen image: ", image_id)

# Read the image from the given file path
image = Image.open("./images/{}.png".format(image_id))
# Convert the image to a numpy array
image = np.array(image)
# Convert pixel values to floating-point format for processing
image = image.astype(np.float64)
# Normalize the pixel values to the range [0, 1]
image = image / 255.0

###############################################
######### Define the forward model for Inpainting ############
if forward_model == "inpainting":
    A_function = inpainting  # Set the forward model to inpainting
    inpainting_missing_fraction = 0.7  # Fraction of missing pixels
    inpainting_observed_fraction = 1 - inpainting_missing_fraction  # Observed fraction of pixels
    A_kwargs = {'prob_observe': inpainting_observed_fraction}  # Forward model arguments
    A_function_adjoint = inpainting  # Adjoint of the forward model

    # Apply the forward model to the image
    b0 = A_function(image, **A_kwargs)

    # Set parameters for the non-local means (NLM) denoiser
    patch = 3  # Patch radius for NLM
    window = 5  # Window radius for NLM
    sigma = np.float64(sys.argv[3]) / 255.0  # Standard deviation (sigma) for noise, provided via command line
    dn = {"guide_image": image, "patch_rad": patch, "window_rad": window, "sigma": sigma}  # Denoiser arguments

###############################################
######### Define the forward model for Deblurring ############
if forward_model == "deblurring":
    A_function = uniform_blurring  # Set the forward model to uniform blurring
    A_kwargs = {'kernal_size': 7}  # Blurring kernel size
    A_function_adjoint = uniform_blurring  # Adjoint of the forward model

    # Apply the forward model to the image
    b0 = A_function(image, **A_kwargs)

    # Set parameters for the non-local means (NLM) denoiser
    patch = 3  # Patch radius for NLM
    window = 5  # Window radius for NLM
    sigma = np.float64(sys.argv[3]) / 255.0  # Standard deviation (sigma) for noise, provided via command line
    dn = {"guide_image": image, "patch_rad": patch, "window_rad": window, "sigma": sigma}  # Denoiser arguments

# Define the conjugate gradient function to solve Wx = bb
def conjugate_gradient(bb, tol=1e-6, max_iter=100):
    # Initialize the solution vector x0 as zeros
    x0 = np.zeros_like(bb)
    x = x0.astype(float)  # Ensure x is float for further operations
    r = bb - DSG_NLM(x, **dn)  # Compute the residual (initial guess is zero)
    p = r.astype(float)  # Set the initial direction to the residual
    rsold = np.sum(r * r)  # Compute the squared norm of the residual

    # Iterative conjugate gradient method
    for i in range(max_iter):
        Ap = DSG_NLM(p, **dn)  # Apply denoising operator to p
        alpha = rsold / np.sum(p * Ap)  # Calculate step size
        x += alpha * p  # Update the solution vector
        r -= alpha * Ap  # Update the residual
        rsnew = np.sum(r * r)  # Compute new squared norm of the residual
        if np.sqrt(rsnew) < tol:  # Check convergence
            break
        p = r + (rsnew / rsold) * p  # Update direction
        rsold = rsnew  # Update old residual

    return x  # Return the solution



"""
Define the Q operator for use in the power method
Q = A^T A + W^dagger (I - W)
QQ computes (Q - bias * I)x. When bias = 0, it computes Qx.
"""


def QQ(x, bias=0.0):  # bias = 0 for highest eigenvalue, lambda_max for lowest eigenvalue
    x3 = x.copy()
    x1 = A_function_adjoint(A_function(x, **A_kwargs), **A_kwargs)
    x2 = conjugate_gradient(x)

    return x1 + x2 - x3 - bias * x3  # Return the result of the Q operator



"""
Implement the power method to find the most or least dominant eigenvalue
bias is important here. When bias = 0, it finds the most dominant eigenvalue of Q. 
When bias = lambda_max(QQ), it finds the least domoinant eigenvalue of Q.
So first we find the most dominant eigenvalue and then setting it as bias, 
we find the least dominant eigenvalue.
"""

def power(bias=0.0, max_iter=1000):
    # Initialize x0 with a normal distribution
    x0 = np.random.normal(0, 1, size=image.shape)

    print("Initial guess ...")
    print(x0)
    print("Starting the power method ...")
    print("bias = ", bias)
    print("max_iter = ", max_iter)

    old_eig_val = 0.0  # Initialize the old eigenvalue

    # Iteratively apply the power method
    for i in range(max_iter):
        x = QQ(x0, bias)  # Apply the Q operator
        eig_val = np.sum(x0 * x) / np.sum(x0 * x0)  # Compute the eigenvalue
        if np.abs(eig_val - old_eig_val) < 1e-8:  # Check convergence
            break
        old_eig_val = eig_val
        if i % 1 == 0:
            print(i, eig_val + bias)  # Print the current iteration and eigenvalue
        x0 = x / np.linalg.norm(x, 'fro')  # Normalize the vector

    eig_val = np.sum(x0 * QQ(x0, bias)) / np.sum(x0 * x0)  # Final eigenvalue computation
    return eig_val + bias, x0  # Return the eigenvalue and corresponding vector

# Display the chosen sigma value
print("Sigma = ", sigma)

# Find the highest eigenvalue using the power method with bias = 0.0
print("Finding the highest eigenvalue ...")
eigs = power(bias=0.0, max_iter=30)

# Adjust the maximum eigenvalue with a ceiling function
lam_max = eigs[0] + 10.0
lam_max = np.ceil(lam_max)
print("lam_max = ", lam_max - 10.0)

# Find the lowest eigenvalue using the power method with bias = lam_max
print("Finding the lowest eigenvalue ...")
eigs = power(bias=lam_max, max_iter=5000)

# Display the final sigma and eigenvalue results
print("sigma = {} and max_eig_val = {}".format(sigma, eigs[0]))
