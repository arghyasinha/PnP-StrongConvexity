#in this script we will call the functions from the other scripts and run the program
#this is the script to call from the command line with the arguments

#the arguments are:
# 1. the path to the folder with the images, default is the folder 'images'
# 2. the path to the folder where the results should be saved, default is the folder 'results'

#imports -- we will import the functions from the other scripts
import os
import numpy as np
import random
import importlib
import argparse
#1. utilities
import utils
importlib.reload(utils)
from utils import *
#2. denoiser
from denoisers import NLM
importlib.reload(NLM)
from denoisers.NLM import DSG_NLM , NLM
#3. Forward Models
import forward_models
importlib.reload(forward_models)    
from forward_models import *
#4. ISTA and ADMM
import iterative_algorithms
importlib.reload(iterative_algorithms)
from iterative_algorithms import *
#5. contractive factor
import contractive_factor
importlib.reload(contractive_factor)
from contractive_factor import *
#5. config
import config
importlib.reload(config)
from config import *
#seed the random number generator
seed =42
random.seed(seed)
np.random.seed(seed)


#define main function, pass the terminal arguments to main
def main():
    #get the terminal arguments, argument parser
    args = parse_args()
    #get image path, output path, image number, initialization, iterative algorithm and inverse problem
    image_path = args.image_path
    output_path = args.output_path
    image_num = args.image_num
    initialization = args.initialization
    ITERATIVE_ALGORITHM = args.ITERATIVE_ALGORITHM
    INVERSE_PROBLEM = args.INVERSE_PROBLEM
    
    #check if output path exists, else create it
    if not os.path.exists(output_path):
        os.makedirs(output_path)
 
    if INVERSE_PROBLEM == 'INPAINTING':
        A_function = inpainting
        inpainting_missing_fraction = 0.7
        inpainting_observed_fraction = 1 - inpainting_missing_fraction
        A_kwargs = {'prob_observe' : inpainting_observed_fraction}
        A_function_adjoint = inpainting
    elif INVERSE_PROBLEM == 'SUPERRESOLUTION':
        A_function = superresolution
        downscale = 0.5
        # A_kwargs = {'f_blur':uniform_blurring, 'blur_kwargs': {'kernal_size': 7}, 'downsample_fraction_x': downscale, \
        #             'downsample_fraction_y': downscale }
        # if downscale == 0.5:
        #     sigma = math.sqrt(3.37)
        # else:
        #     sigma = 1/ (math.pi * downscale)
        # sigma = 2*(1/ (math.pi * downscale))
        sigma = 1.0
        A_kwargs = {'f_blur':gaussian_blurring, 'blur_kwargs': {'kernal_size': 7, 'sigma': sigma}, 'downsample_fraction_x': \
                    downscale, 'downsample_fraction_y': downscale }
        A_function_adjoint = superresolution_adjoint
    elif INVERSE_PROBLEM == 'UNIFORM-DEBLURRING':
        A_function = uniform_blurring
        A_kwargs = {'kernal_size': 7}
        A_function_adjoint = uniform_blurring
    elif INVERSE_PROBLEM == 'GAUSSIAN-DEBLURRING':
        A_function = gaussian_blurring
        A_kwargs = {'kernal_size': 7, 'sigma': 1.0}
        A_function_adjoint = gaussian_blurring
    #else assert error
    else:
        assert False, 'Inverse problem not implemented'


    #----------------------------------      LOADING IMAGE       ----------------------------------
    #----------------------------------                          ----------------------------------
    #read the image
    image = Image.open(get_path(image_path = image_path, number = image_num))
    #convert the image to a numpy array
    image = np.array(image)
    #to double
    image = image.astype(np.float64)
    #to range 0 to 1P_operator
    image = image/255.0

    #we will augument the image if the appication is super resolution: to be a multiple of 2
    if A_function.__name__ == 'superresolution':
        image = augument_image_superresolution(image = image, downsample_fraction_x = A_kwargs['downsample_fraction_x'],\
             downsample_fraction_y = A_kwargs['downsample_fraction_y'])
    # #----------------------------------      LOADING IMAGE DONE      ----------------------------------
    # #----------------------------------                              ----------------------------------

    #----------------------------------      FORWARD MODEL       ----------------------------------
    #----------------------------------                          ----------------------------------
    #we would be calling the function forward_model
    #we will be changing the A_kwargs based on different forward models
    if A_function.__name__ == 'inpainting' :
        #add the keyword argument return_mask as True
        A_kwargs['return_matrix'] = True
        #we will apply the forward model function with the k_wargs
        b, application_matrix = A_function(image, **A_kwargs)
        #print the amounbt of memory allocated to the variable application_matrix
        # print('memory allocated to application_matrix: ', application_matrix.nbytes)
        #remove the keyword argument return_mask
        A_kwargs.pop('return_matrix')
        #add the keyword inpainting_mask with the value inpainting_mask
        A_kwargs[str(A_function.__name__)+'_matrix'] = application_matrix

    #else if name  == uniform_blurring or gaussian_blurring
    elif A_function.__name__ == 'uniform_blurring' or A_function.__name__ == 'gaussian_blurring' \
        or A_function.__name__ == 'superresolution':
        #we will apply the forward model function with the k_wargs
        b = A_function(image, **A_kwargs)
        #else, if anything else, error
    else:
        raise ValueError(A_function.__name__ , 'is not implemented')
    #here we have established teh the forward model is one of the following:
    #1. inpainting, 2. uniform_blurring, 3. gaussian_blurring, 4. superresolution

    if ADD_NOISE:
        #add gaussian noise with mean 0 and variance noise_variance to b
        noise = noise_variance*np.random.randn(image.shape[0], image.shape[1])
        start = time.time()
        noise = A_function(noise, **A_kwargs)
        end = time.time()
        #print time taken for forward modle on noise
        print('time taken for forward model on noise: ', end-start)
        b = b + noise
    b[b > 1] = 1
    b[b < 0] = 0
    #----------------------------------      FORWARD MODEL  ENDS    ----------------------------------
    #----------------------------------                             ----------------------------------
    
    #----------------------------------      ADJOINT MODEL       --------------------------------------
    #copy the A_kwargs dictionary into A_adjoint_kwargs
    A_adjoint_kwargs = A_kwargs.copy()
    #if A_function name is superresolution, then we will add the keyword arguments
    #original_image_rows and original_image_cols
    if A_function.__name__ == 'superresolution':
        A_adjoint_kwargs['original_image_rows'] = image.shape[0]
        A_adjoint_kwargs['original_image_cols'] = image.shape[1]
    #assert that the dictionary A_adjoint_kwargs has the key str(A_function.__name__)+'_matrix' if its inpainting
    #or superresolution
    if A_function.__name__ == 'inpainting' :
        assert str(A_function.__name__)+'_matrix' in A_adjoint_kwargs.keys()
    #----------------------------------      ADJOINT MODEL   ENDS    ----------------------------------
    
    #----------------------------------      INITIALIZATION       ----------------------------------
    #----------------------------------                           ----------------------------------
    #if initialization is median
    if initialization == 'median':
        init_kwargs = {}
        #we will be initializing the optimization variable x for reconstruction
        #we would do this in differently for different forward models
        #if A_function name is inpainting, then initialize_median_filter
        if A_function.__name__ == 'inpainting':
            x0 = initialize_median_filter(observed_image = b, application_matrix = A_kwargs[str(A_function.__name__)+'_matrix'], \
                        **init_kwargs) 
        #else if A_function name is superresolution, then init_ISTA but we will build teh parameters
        #observed_image would be A_function(b, **A_kwargs), application_matrix would be 
        #A_kwargs[str(A_function.__name__)+'_matrix']^T  A_kwargs[str(A_function.__name__)+'_matrix']
        elif A_function.__name__ == 'superresolution':
            #create the initial image
            initial_image = A_function_adjoint(b, **A_adjoint_kwargs)
            #create the superresolution mask matrix
            superresolution_mask_matrix = create_superresolution_mask(original_image_rows=\
                A_adjoint_kwargs['original_image_rows'], original_image_cols= A_adjoint_kwargs['original_image_cols'], \
                downsample_fraction_x = A_adjoint_kwargs['downsample_fraction_x'], downsample_fraction_y = \
                A_adjoint_kwargs['downsample_fraction_y'])
            x0 = initialize_median_filter(observed_image = initial_image, \
                        application_matrix = superresolution_mask_matrix, **init_kwargs)

        #elif A_function name is uniform_blurring or gaussian_blurring, then we will udo nothing
        elif A_function.__name__ == 'uniform_blurring' or A_function.__name__ == 'gaussian_blurring':
            x0 = b
        #else, report error that the forward model is not implemented
        else:
            raise ValueError(A_function.__name__ , 'is not implemented')
    #else if initialization is zero
    #we will be initializing the optimization variable x for reconstruction with all zeros
    elif initialization == 'zero':
        x0 = np.zeros_like(image)
    #else if initialization is ones
    #we will be initializing the optimization variable x for reconstruction with all ones
    elif initialization == 'ones':
        x0 = np.ones_like(image)
    #else if initialization is gaussian
    #we will be initializing the optimization variable x for reconstruction with gaussian noise between 0 and 1
    elif initialization == 'gaussian':
        x0 = np.random.normal(loc=0, scale=1, size=(image.shape[0], image.shape[1]))
        x0 = 0.5 + 0.5*x0
    else:
        #assert error, initialization is not implemented
        assert False, 'initialization is not implemented'
    #----------------------------------      INITIALIZATION ENDS       ----------------------------------
     #----------------------------------                                ----------------------------------
    #create the list to store the psnr, ssim and euclidean distance
    #create a dict with list of psnr, ssim and euclidean distance, mse
    quality_list_dict = {}
    quality_list_dict['psnr_list'] = []
    quality_list_dict['ssim_list'] = []
    quality_list_dict['log_l2_list'] = []
    quality_list_dict['log_euclidean_distance_list'] = []
    #we will have one more list to store all the iterates in a list
    iterate_list = []
    euclidean_distance=[]
    #create a variable to check convergence
    converged = False
    convergence_iter = -1
    #start the iterations
    b_h = A_function_adjoint(b, **A_adjoint_kwargs)
    x_k = x0
    z_k = np.zeros_like(x_k)
    y_k = np.zeros_like(x_k)
    #append the iterate to the list
    iterate_list.append(x_k)
    #save the initial image
    # save_image_to_file(x_k, log_path, 'initial_image')
    #print initial psnr, ssim and euclidean distance
    quality_metric_dict = image_quality_metrics(img1 = image, img2=x_k,data_range=1.0, verbose=True)

    print('initial psnr: ', quality_metric_dict['psnr'])
    print('initial ssim: ', quality_metric_dict['ssim'])
    print('initial log_l2: ', quality_metric_dict['log_l2'])
    print('(x0,ground truth) log euclidean distance: ', np.log(np.linalg.norm(x0 - image)))
    #add , A_function, A_kwargs, A_function_adjoint, A_adjoint_kwargs, step_size
    cg_leftside_arg_dict = {'A_function': A_function, 'A_kwargs': A_kwargs, 'A_function_adjoint': A_function_adjoint, \
                            'A_adjoint_kwargs': A_adjoint_kwargs, 'step_size': step_size}
    ## Contractive factor -----------------------------------------
    if ITERATIVE_ALGORITHM == 'ISTA':
        #we will calculate the contraction factor P = WG 
        #we will use power method to calculate the largest eigenvalue of P=WG
        #for power method we will call the function : power_method_for_images
        #we will construc the argument dict args_dict for calling the function
        #we will add input_image, denoiser, denoiser_kwargs, A_function, A_kwargs, A_function_adjoint, A_adjoint_kwargs, eta
        args_dict_P = {'denoiser': denoiser, 'denoiser_kwargs': denoiser_kwargs, \
            'A_function': A_function, 'A_kwargs': A_kwargs, 'A_function_adjoint': A_function_adjoint, \
            'A_adjoint_kwargs': A_adjoint_kwargs, 'eta': step_size}
        #if the denoiser is DSG_NLM, then we will use the function P_operator for f in the power method function
        #if its NLM, we will use f= D_half_P_D_half_inverse
        if denoiser.__name__ == 'DSG_NLM':
            #add keyword 'guide_image' = image to the denoiser_kwargs
            denoiser_kwargs['guide_image'] = image
            function_spectral_norm = P_operator
        elif denoiser.__name__ == 'NLM':
            #add keyword 'guide_image' = image to the denoiser_kwargs
            denoiser_kwargs['guide_image'] = image
            function_spectral_norm = D_half_P_D_half_inverse
            #we need to add the keyword argument 'D_matrix' to the args_dict_P
            #we can get teh D_matrix by calling teh denoiser with an extra keyword argument 'return_D_matrix' = True
            #add the keyword argument 'return_D_matrix' = True to the denoiser_kwargs
            denoiser_kwargs['return_D_matrix'] = True
            #call the denoiser
            _, D_matrix = denoiser(image, **denoiser_kwargs)
            #drop the keyword argument 'return_D_matrix' = True from the denoiser_kwargs dictionary
            del denoiser_kwargs['return_D_matrix']
            #add the D_matrix to the args_dict_P
            args_dict_P['D_matrix'] = D_matrix
        else:
            #assert error
            assert False, 'denoiser is not implemented'
        #call the power method function
        contractive_factor = power_method_for_images(f = function_spectral_norm, input_image = x0, \
            args_dict = args_dict_P, max_iterations=power_method_max_iterations )
        #print the contractive factor
        print('contractive factor: ', contractive_factor)
        #delete the keyword argument 'guide_image' from the denoiser_kwargs if its there
        if 'guide_image' in denoiser_kwargs.keys():
            del denoiser_kwargs['guide_image']
        #contractive factor ends -----------------------------------------
    
    #----------------------------------      ITERATIVE ALGORITHM       ----------------------------------
    #----------------------------------                                ----------------------------------
    for i in range(max_iterations):
        #check if the iterative algorithm is ADMM
        if ITERATIVE_ALGORITHM == 'ADMM' :
            # or iterative_algorithm.__name__ == 'ADMM_flip'\
            
            #call the ADMM update
            y_k, x_k_plus_1, z_k = ADMM(y_k, x_k, z_k, b_h, denoiser, denoiser_kwargs,\
                                                     cg_leftside_arg_dict, iterations_to_fix_W, i, image,\
                                                        max_iter=MAX_ITERS_CG, tol=CG_TOL)
        #else if the iterative algorithm is ISTA
        elif ITERATIVE_ALGORITHM == 'ISTA':
            #call the ISTA update
            a = ISTA(image, x_k, b,  denoiser, denoiser_kwargs, iterations_to_fix_W,\
                                              A_function, A_kwargs, A_function_adjoint, A_adjoint_kwargs, i, step_size,\
                                                power_method_max_iterations,  calculate_contractive_factor)
            x_k_plus_1 = a[0]
        #anything else, assert error, iterative algorithm not implemented
        else:
            raise ValueError(ITERATIVE_ALGORITHM, 'is not implemented')
        #analysis ---------------------------------------
        #iterates done, now we will do analysis of iterates
        #append the iterate to the list
        iterate_list.append(x_k_plus_1)
        
        #get euclidean distance
        euclidean_distance = np.linalg.norm(x_k_plus_1 - x_k)
        #update x_k
        x_k = x_k_plus_1
        #calculate the psnr, ssim and euclidean distance
        quality_metric_dict = image_quality_metrics(img1 = image, img2=x_k_plus_1,data_range=1.0, verbose=False)
        #append the values to the list
        quality_list_dict = append_and_print_quality_metrics(quality_metric_dict, np.log(euclidean_distance), quality_list_dict, i)

        #we will save each image in the log directory with iteration number as its name and then image extension
        #save the image
        # save_image_to_file(image = x_k_plus_1, directory_path=log_path, filename=str(i))
        #check convergence
        if np.isclose(euclidean_distance, tolerance, atol=1E-12):
            converged = True
            convergence_iter = i
            # downsample_fraction_x, downsample_fraction_y
            break
        if converged:
            print('converged in {} iterations'.format(convergence_iter))
        #plot the iterates and the reconstructed image
        plot_iterates(quality_list_dict = quality_list_dict, iterate_list = iterate_list,  x0 = x0, image = image, b = b, x_k = x_k, \
                                 output_directory = output_path, image_number = image_num)
        #----------------------------------      ITERATIVE ALGORITHM ENDS       ----------------------------------

#define main function
if __name__ == '__main__':
    main()
