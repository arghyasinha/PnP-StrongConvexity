from denoisers.NLM import *
from PIL import *
from iterative_algorithms import *
from forward_models import *
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pandas as pd



# -------- INPAINTING SETUP -------- #
# Set up the inpainting forward model
A_function = inpainting  # Define the forward model as inpainting
inpainting_missing_fraction = 0.7  # Fraction of pixels missing in the image
inpainting_observed_fraction = 1 - inpainting_missing_fraction  # Fraction of observed pixels
A_kwargs = {'prob_observe': inpainting_observed_fraction}  # Parameters for inpainting
A_function_adjoint = inpainting  # Define the adjoint operation (for linear problems it is often the same)
A_adjoint_kwargs = A_kwargs  # Parameters for the adjoint operation
# -------- END INPAINTING SETUP -------- #




# -------- SELECT DENOISING METHOD -------- #
Method = float(input("Enter the method: 1. DSG-NLM 2. NLM: "))  # Choose between DSG-NLM and NLM methods
if Method == 1:
    den = DSG_NLM  # Use symmetric DSG-NLM denoiser
    print("Method: DSG-NLM")
else:
    den = NLM  # Use standard NLM denoiser (non-symmetric)
    print("Method: NLM")

# -------- END DENOISING METHOD SELECTION -------- #


# FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) for grayscale images
def fista_gen(input_image, start_image, denoiser, denoiser_args, step_size=0.9, num_iterations=10):
    x_old = start_image.copy()  # Initialize x_old with the starting image
    y = start_image.copy()  # Initialize y as well
    N = []  # List to store cost values (not used in this version)
    all_iters = []  # List to store images at each iteration
    t = 1  # Momentum term

    # Calculate the D matrix for scaling (specific to NLM denoiser)
    _, DD = NLM(input_image, **denoiser_args, return_D_matrix=True)
    DD_inv = 1 / DD  # Invert D matrix

    if Method == 1:
        DD_inv = 1  # For DSG-NLM, set scaling to 1

    print("DD_inv:", DD_inv)  # Debugging information for D matrix

    # Main loop for FISTA
    for i in range(num_iterations):
        # Gradient step with scaling
        y = y - step_size * DD_inv * A_function_adjoint(A_function(y, **A_kwargs), **A_adjoint_kwargs) + step_size * DD_inv * A_function_adjoint(input_image, **A_adjoint_kwargs)
        y = y.astype(np.float32)  # Ensure data type is correct for computation

        # Apply the denoiser
        x = denoiser(y, **denoiser_args)

        # Update momentum parameter (PnP-FISTA)
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
        alpha = (t - 1) / t_new
        #alpha = 0  #### Uncomment For PnP-ISTA
        y = x + alpha * (x - x_old)  # Update y for next iteration

        # Calculate SSIM and PSNR metrics for performance evaluation
        ssim_index = ssim(x, image, data_range=1.0)
        psnr_value = psnr(x, image, data_range=1.0)

        print(i)  # Print current iteration
        print("PSNR:", psnr_value)  # Print PSNR value
        print("SSIM:", ssim_index)  # Print SSIM value

        # Store the result of the current iteration
        all_iters.append(x_old.astype(np.float32))
        x_old = x  # Update x_old for the next iteration
        t = t_new  # Update t for the next iteration

    return x, N, all_iters  # Return the final result, cost values, and all iterations

# FISTA for color images. Only difference is in calculating the SSIM and PSNR metrics.
def fista_gen_color(input_image, start_image, denoiser, denoiser_args, step_size=0.9, num_iterations=10, ground_truth=None):
    x_old = start_image.copy()  # Initialize x_old with the starting image
    y = start_image.copy()  # Initialize y as well
    N = []  # List to store cost values (not used in this version)
    all_iters = []  # List to store images at each iteration
    t = 1  # Momentum term

    # Main loop for FISTA (color version)
    for i in range(num_iterations):
        # Gradient step
        y = y - step_size * A_function_adjoint(A_function(y, **A_kwargs), **A_adjoint_kwargs) + step_size * A_function_adjoint(input_image, **A_adjoint_kwargs)
        y = y.astype(np.float32)  # Ensure data type is correct

        # Apply the denoiser
        x = denoiser(y, **denoiser_args)

        # Update momentum parameter
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
        alpha = (t - 1) / t_new
        #alpha = 0  #### Uncomment For PnP-ISTA
        y = x + alpha * (x - x_old)  # Update y for next iteration

        # Calculate SSIM and PSNR metrics for performance evaluation
        ssim_index = ssim(x, ground_truth, data_range=1.0)
        psnr_value = psnr(x, ground_truth, data_range=1.0)

        print("PSNR:", psnr_value)  # Print PSNR value
        print("SSIM:", ssim_index)  # Print SSIM value

        # Store the result of the current iteration
        all_iters.append(x_old)
        x_old = x  # Update x_old for the next iteration
        t = t_new  # Update t for the next iteration

    return x, N, all_iters  # Return the final result, cost values, and all iterations


# Load an image from file
image = Image.open("./images/11.png")

# Convert the image to a numpy array
image = np.array(image)

# Convert the pixel values to float (necessary for further processing)
image = image.astype(np.float32)

# Normalize pixel values to the range 0 to 1
image = image / 255.0

# Determine the number of channels (e.g., RGB image has 3 channels, grayscale has 1)
n_channels = image.shape[2] if len(image.shape) == 3 else 1
print("Number of channels:", n_channels)

if n_channels == 3:
    # Extract the red, green, and blue channels from the RGB image
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    # Apply the forward model A_function to each channel
    r0 = A_function(r, **A_kwargs)
    g0 = A_function(g, **A_kwargs)
    bb0 = A_function(b, **A_kwargs)

    # Merge the modified channels back into a single RGB image
    observed = np.stack([r0, g0, bb0], axis=-1)
else:
    # For grayscale images, apply the forward model directly
    observed = A_function(image, **A_kwargs)
    # Reshape the observed image to include the channel dimension
    observed = observed.reshape((observed.shape[0], observed.shape[1], 1))

# Function to calculate SSIM for color images
def ssim_index_color(observed, image):
    ssim_index = 0.0
    # Loop through each channel (R, G, B)
    for channel in range(3):
        # Extract each channel from the observed and original images
        channel_image = image[:, :, channel]
        observed_channel = observed[:, :, channel]
        
        # Calculate SSIM for each channel
        observed_ssim = ssim(observed_channel, channel_image, data_range=1.0)

        print(observed_ssim)
        ssim_index += observed_ssim

    # Average the SSIM values across all 3 channels
    ssim_index /= 3
    return ssim_index

# Calculate PSNR and SSIM values for the observed image
psnr_value = psnr(observed.reshape(image.shape), image, data_range=1.0)
print("PSNR:", psnr_value)

ssim_index = ssim_index_color(observed, image) if n_channels == 3 else ssim(observed.reshape(image.shape), image, data_range=1.0)
print("SSIM:", ssim_index)

# ----- MEDIAN FILTER FOR GUIDE IMAGE----- #
from scipy.ndimage import generic_filter

# Function to apply median filtering, excluding zero values
def zero_exclusive_median(arr):
    # Remove zero values from the array
    arr = arr[arr > 0.0]
    if len(arr) == 0:
        return 0  # If all values are zero, return zero
    else:
        return np.median(arr)  # Otherwise, return the median of non-zero values

# Apply the zero-exclusive median filter to each channel
filtered_image = observed.copy()
all_channels = []

# Loop through all channels (R, G, B or single channel)
for channels in range(n_channels):
    filtered_image = observed.copy()[:, :, channels]
    for i in range(3):  # Apply median filter 3 times
        filtered_image = generic_filter(filtered_image, zero_exclusive_median, size=3)

    # Append the filtered channel to the list
    all_channels.append(filtered_image)

# Merge the filtered channels back into a single image
filtered_image = np.stack(all_channels, axis=-1)

# Make a copy of the filtered image for further processing (GMM refers to guide image)
gmm = filtered_image.copy()
print(" --- gmm shape = {} --- ".format(gmm.shape))

# Calculate SSIM and PSNR for the GMM-filtered image
print("GMM")
ssim_index = ssim_index_color(gmm, image) if n_channels == 3 else ssim(gmm.reshape(image.shape), image, data_range=1.0)
psnr_value = psnr(gmm.reshape(image.shape), image, data_range=1.0)
print("PSNR:", psnr_value)
print("SSIM:", ssim_index)
print("-----------------")

# User input for the choice of starting image
 ###Reconstruction does not depend on this choice
choice_of_input = input("Enter choice of input image: 1. Observed 2. All-zeros and 3. Random: ")
print("Choice of Start Image (x0):", choice_of_input)

if choice_of_input == "1":
    start_image = observed.copy()  # Use observed image as the starting point
elif choice_of_input == "2":
    start_image = np.zeros_like(observed)  # Use an all-zero image as the starting point
elif choice_of_input == "3":
    np.random.seed(7)
    start_image = np.random.rand(*observed.shape)  # Use a random image as the starting point

# Create a folder to save results
folder_name = "INPAINTING_PnP-Strong-{}".format(Method)
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Save original, input, and median-filtered images
if n_channels == 3:
    cv2.imwrite("{}/original.png".format(folder_name), cv2.cvtColor(image * 255.0, cv2.COLOR_BGR2RGB))
    cv2.imwrite("{}/input.png".format(folder_name), cv2.cvtColor(observed * 255.0, cv2.COLOR_BGR2RGB))
    cv2.imwrite("{}/median.png".format(folder_name), cv2.cvtColor(gmm * 255.0, cv2.COLOR_BGR2RGB))
else:
    cv2.imwrite("{}/original.png".format(folder_name), image * 255.0)
    cv2.imwrite("{}/input.png".format(folder_name), observed * 255.0)
    cv2.imwrite("{}/median.png".format(folder_name), gmm[:, :, 0] * 255.0)
    cv2.imwrite("{}/start-image-{}.png".format(folder_name, choice_of_input), start_image * 255.0)

# Parameters for the reconstruction algorithm
"""
Parameters for the denoiser. Tuning these parameters can affect the reconstruction.
The guide image is stored as gmm.
All parameters are provided in the dictionary dn.
"""
patch = 1
window = 1
sigma = 40.0 / 255.0
ALL_ITERS = []

# Reshape observed image and perform reconstruction
observed = observed.reshape((observed.shape[0], observed.shape[1], n_channels))
reconstruction = gmm.copy()

# Loop through each channel and perform reconstruction
for channel in range(n_channels):
    print("channel:", channel)
    dn = {"guide_image": gmm[:, :, channel], "patch_rad": patch, "window_rad": window, "sigma": sigma}

    # Use FISTA algorithm to reconstruct each channel
    reconstruction[:, :, channel], _, ALL_ITERS_CHANNELS = fista_gen(observed[:, :, channel], start_image[:, :, channel], den, dn, step_size=0.9, num_iterations=70000)
    ALL_ITERS.append(ALL_ITERS_CHANNELS)
    print('----------------------')

# Reshape the final reconstructed image to the original shape
reconstruction = reconstruction.reshape(image.shape)
print("Reconstruction done")

# If the image has 3 channels, create iterations for each
if n_channels == 3:
    ITERS = []
    for i in range(len(ALL_ITERS[0])):
        temp = image.copy()
        for channel in range(n_channels):
            temp[:, :, channel] = ALL_ITERS[channel][i]
        ITERS.append(temp)
else:
    ITERS = ALL_ITERS[0]

# Calculate PSNR values for each iteration
ITERS_PSNR = [psnr(x, image, data_range=1.0) for x in ITERS]

# Save PSNR values into a DataFrame and export to CSV
df = pd.DataFrame(ITERS_PSNR, columns=['PSNR'])
df.to_csv('{}/ITERS_PSNR-{}.csv'.format(folder_name, choice_of_input), index=False)

# Print SSIM and PSNR values for reconstruction, observed, and start images
if n_channels == 3:
    print("SSIM OF RECONSTRUCTION =", ssim_index_color(reconstruction, image))
else:
    print("SSIM OF RECONSTRUCTION =", ssim(reconstruction, image, data_range=1.0))

print("PSNR OF RECONSTRUCTION =", psnr(reconstruction, image, data_range=1.0))


if n_channels == 3:
    print("SSIM OF OBSERVED =", ssim_index_color(observed, image))
else:
    print("SSIM OF OBSERVED =", ssim(observed.reshape(image.shape), image, data_range=1.0))

print("PSNR OF OBSERVED =", psnr(observed.reshape(image.shape), image, data_range=1.0))


if n_channels == 3:
    print("SSIM OF START IMAGE =", ssim_index_color(start_image, image))
else:
    print("SSIM OF START IMAGE =", ssim(start_image.reshape(image.shape), image, data_range=1.0))

print("PSNR OF START IMAGE =", psnr(start_image.reshape(image.shape), image, data_range=1.0))



# Save the reconstructed image in a pickle file
import pickle
with open('{}/reconstruction-{}.pkl'.format(folder_name, choice_of_input), 'wb') as f:
    pickle.dump(reconstruction, f)

# Save the reconstructed image as a PNG file
if n_channels == 3:
    cv2.imwrite("{}/reconstruction-{}.png".format(folder_name, choice_of_input), cv2.cvtColor(reconstruction * 255.0, cv2.COLOR_BGR2RGB))
else:
    cv2.imwrite("{}/reconstruction-{}.png".format(folder_name, choice_of_input), reconstruction * 255.0)
