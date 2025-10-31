import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import LinearNDInterpolator
from skimage import io

from package import Class_DIC as dic
from package.Class_Inputs import Inputs
from package.Class_Observations import Observations

device = torch.device('cpu')

# Physical parameters of the specimen 
length = 20 # in mm
heigth = 50 # in mm

# Image parameters
nb_px_row = 2048 # number of pixel along the rows
nb_px_col = 589 # number of pixel along the columns

# ratio to go from pixel to real world coordinate system
ratio_x = length / nb_px_col
ratio_y = heigth / nb_px_row

# Inputs
coloc_inputs  = Inputs(device, N_coloc = [40,100], N_coloc_bc=[100, 100, 500, 500], nbr_hlines=500,
                  variable_boundaries=[[0., length], [0., heigth]])

# Get the observations from the ground truth displacement solution 
obs_dci    = Observations('solution_reference_lineaire_2GPa_1000N.txt', N_obs = -1, stdu=0.0)
u_x_interp = LinearNDInterpolator(obs_dci.data[:,:2].numpy(), obs_dci.data[:,2].numpy(), fill_value = 0)
u_y_interp = LinearNDInterpolator(obs_dci.data[:,:2].numpy(), obs_dci.data[:,3].numpy(), fill_value = 0)

def u_func(inputs_px):
    x_rwc = inputs_px[1] * ratio_x
    y_rwc = heigth - inputs_px[0] * ratio_y
    inputs_rwc = np.array([x_rwc, y_rwc])
    u_x = u_x_interp(inputs_rwc)[0]
    u_y = u_y_interp(inputs_rwc)[0]
    
    return np.array([- u_y / ratio_y, u_x / ratio_x])

''' Generation of the image of reference '''
I_0 = np.asarray(io.imread('/Users/romainbonnet-eymard/Desktop/These/DIC/Sample14/Sample14_Reference.tif', as_gray=True)).T.astype(np.float32)

print('Image of reference generated')

''' Computation of the deformed image '''
I_t = dic.generate_deformed_image(I_0, u_func)

print('Deformed image generated')

''' Display the two images and the grey-level residual '''
fig, axs = plt.subplots(1,3,figsize = (15,5))

h = axs[0].imshow(I_0, cmap='gray')
plt.colorbar(h, ax=axs[0])
axs[0].set_title(r'$I_0(x)$')
axs[0].set_aspect(length/heigth)

h = axs[1].imshow(I_t, cmap='gray')
plt.colorbar(h, ax=axs[1])
axs[1].set_title(r'$I_t(x)$')
axs[1].set_aspect(length/heigth)

h = axs[2].imshow(I_0 - I_t, cmap='RdBu')
plt.colorbar(h, ax=axs[2])
axs[2].set_title(r'$I_0(x)-I_t(x)$')
axs[2].set_aspect(length/heigth)