# %% Library imports


import torch
import numpy as np
from package.Class_Mesh import Mesh


# %% Physical constants
F = torch.tensor([50.])  # traction effort in MPa
stdF = 0.  # standard deviation of the added gaussian noise on F
Fobs = F + stdF*torch.randn(1)  # F avec le bruit
nu = 0.3  # Poisson ratio


# %% Helper functions




def create_mesh_E(inputs, n_E):
    '''
    Creates a Q4-type mesh object for the interpolation of E.
    This function works specifically for a plate since all elements are the same.
    Nodes and elements are stored as 1d-vector (k-indexing) while nodes coordinated are given by 2d-vector ((i,j)-indexing).
    To get the correspondance between the two indexing notations we use the following convention:
        - k = 0 and (i,j)=(0,0) in the lower right corner 
        - k increases as i=0 and j increases
        - when j = j_max, then i += 1 and j=0 and we restart 
    '''
    def from_global_2_local(N, n_x, n_y): return (
        N // n_y, N % n_y)  # convert a k-index to (i,j)


    def from_local_2_global(i, j, n_x, n_y): return i * \
        n_y + j           # convert (i,j) to k-index


    nodes_x = np.linspace(inputs.x_variable_min,
                          inputs.x_variable_max, n_E[0]+1)
    nodes_y = np.linspace(inputs.y_variable_min,
                          inputs.y_variable_max, n_E[1]+1)


    # attribute of Mesh class which stores the coordinates of nodes
    nodes = np.zeros(((n_E[0]+1)*(n_E[1]+1), 2))
    for index_nodes in range(nodes.shape[0]):
        index_x, index_y = from_global_2_local(index_nodes, n_E[0]+1, n_E[1]+1)
        nodes[index_nodes, 0] = nodes_x[index_x]
        nodes[index_nodes, 1] = nodes_y[index_y]


    # attribute of Mesh class which connect each element to their corresponding
    connectivity_table = np.zeros((n_E[0]*n_E[1], 4), dtype=int)
    # nodes
    for index_elt in range(connectivity_table.shape[0]):
        index_row, index_col = from_global_2_local(index_elt, n_E[0], n_E[1])
        connectivity_table[index_elt, 0] = from_local_2_global(
            index_row, index_col, n_E[0]+1, n_E[1]+1)
        connectivity_table[index_elt, 1] = from_local_2_global(
            index_row, index_col+1, n_E[0]+1, n_E[1]+1)
        connectivity_table[index_elt, 2] = from_local_2_global(
            index_row+1, index_col+1, n_E[0]+1, n_E[1]+1)
        connectivity_table[index_elt, 3] = from_local_2_global(
            index_row+1, index_col, n_E[0]+1, n_E[1]+1)


    return Mesh(nb_nodes=nodes.shape[0], nb_elt=connectivity_table.shape[0], dim=2, elementType='Q4',
                nodes=nodes, connectivity_table=connectivity_table)




def E_function(x, k, metamodel, inputs):
    """
    Function used for the reconstruction of the spatially-distributed E parameter with P1 interpolation. 
    """
    if (metamodel.E_interpolation == 'P1'):


        # Shape functions for Q4 centered in (0,0) with dimension dx*dy
        def N0(x, dx=1, dy=1):
            return 1/(dx*dy)*(dx/2 - x[:, 0])*(dy/2 - x[:, 1])


        def N1(x, dx=1, dy=1):
            return 1/(dx*dy)*(dx/2 - x[:, 0])*(dy/2 + x[:, 1])


        def N2(x, dx=1, dy=1):
            return 1/(dx*dy)*(dx/2 + x[:, 0])*(dy/2 + x[:, 1])


        def N3(x, dx=1, dy=1):
            return 1/(dx*dy)*(dx/2 + x[:, 0])*(dy/2 - x[:, 1])


        def condition(x):
            return torch.sum(x < 0).item() == 0


        if (k.numel() == 1):
            return k*torch.tensor([1.])


        else:
            dx = (inputs.x_variable_max - inputs.x_variable_min) / \
                (metamodel.n_E[0])  # Same for all element
            dy = (inputs.y_variable_max - inputs.y_variable_min) / \
                (metamodel.n_E[1])  # Same for all element


            ''' As x[:,1] are not well ordered, the torch.bucketsize approach is not the most efficient. Here we use the fact
            all element are geometrically the same. Notice that we must clamp to max = metamodel.n_E[0]-1 because of the choice
            of indexing of nodes and element '''


            # x component index of the lower left corner node of the element in which lies x
            indice_x = ((x[:, 0] // dx).type(torch.int)
                        ).clamp(min=0, max=metamodel.n_E[0]-1)
            # y component index of the lower left corner node of the element in which lies x
            indice_y = ((x[:, 1] // dy).type(torch.int)
                        ).clamp(min=0, max=metamodel.n_E[1]-1)


            indice_elt = indice_x * metamodel.n_E[1] + indice_y
            indice_nodes = metamodel.mesh_E.connectivity_table[indice_elt, :]


            ''' Translate to the reference Q4 element to use the shape functions above-defined '''
            x_translated = x - \
                (torch.from_numpy(
                    metamodel.mesh_E.nodes[indice_nodes[:, 0]]) + torch.tensor([dx/2, dy/2]))


            #if not (condition(N0(x_translated, dx=dx, dy=dy)) and condition(N1(x_translated, dx=dx, dy=dy)) and condition(N2(x_translated, dx=dx, dy=dy)) and condition(N3(x_translated, dx=dx, dy=dy))):
            #   print(x_translated, dx, dy)


            assert condition(N0(x_translated, dx=dx, dy=dy)) and condition(N1(x_translated, dx=dx, dy=dy)) and condition(
                N2(x_translated, dx=dx, dy=dy)) and condition(N3(x_translated, dx=dx, dy=dy))
            return (k[indice_nodes[:, 0]] * N0(x_translated, dx=dx, dy=dy) +
                    k[indice_nodes[:, 1]] * N1(x_translated, dx=dx, dy=dy) +
                    k[indice_nodes[:, 2]] * N2(x_translated, dx=dx, dy=dy) +
                    k[indice_nodes[:, 3]] * N3(x_translated, dx=dx, dy=dy))


    else:
        return None




# %% Loss functions
def epsilon(metamodel, domain):
    """
    Compute the strain from the estimated displacement on and with respect to a given domain.
    """
    u = metamodel.model_u(domain)
    grad_u_x = torch.autograd.grad(u[:, 0], domain,
                                   grad_outputs=torch.ones_like(u[:, 0]),
                                   create_graph=True, retain_graph=True)[0]
    grad_u_y = torch.autograd.grad(u[:, 1], domain,
                                   grad_outputs=torch.ones_like(u[:, 1]),
                                   create_graph=True, retain_graph=True)[0]


    return torch.hstack((grad_u_x[:, 0].view(-1, 1),
                         grad_u_y[:, 1].view(-1, 1),
                         0.5*(grad_u_x[:, 1]+grad_u_y[:, 0]).view(-1, 1)))




def eps_2_sigma(metamodel, domain, inputs):
    """
    Compute the stress from the estimated strain with the 2D in plane stress constitutive law
    """
    epsilon_tilde = epsilon(metamodel, domain)


    E = metamodel.model_E(domain) #metamodel.E_ref * E_function(domain, metamodel.E, metamodel, inputs)


    sigma_xx = E/(1-nu**2) * (epsilon_tilde[:, 0] + nu*epsilon_tilde[:, 1])
    sigma_yy = E/(1-nu**2) * (nu*epsilon_tilde[:, 0] + epsilon_tilde[:, 1])
    sigma_xy = E/(1+nu) * epsilon_tilde[:, 2]


    return torch.hstack((sigma_xx.view(-1, 1),
                         sigma_yy.view(-1, 1),
                         sigma_xy.view(-1, 1)))




def J_res(metamodel, domain, is_sigma_trained):
    """
    Loss function for physical model residual calculated on a given domain. The computation of the stress field comes from the dedicated PINN 
    """


    sigma_tilde = Fobs*is_sigma_trained * \
        metamodel.model_sigma(domain) + (1-is_sigma_trained) * \
        metamodel.model_sigma(domain)
    grad_sigma_xx = torch.autograd.grad(sigma_tilde[:, 0], domain,
                                        grad_outputs=torch.ones_like(
                                            sigma_tilde[:, 0]),
                                        create_graph=True, retain_graph=True)[0]
    grad_sigma_yy = torch.autograd.grad(sigma_tilde[:, 1], domain,
                                        grad_outputs=torch.ones_like(
                                            sigma_tilde[:, 1]),
                                        create_graph=True, retain_graph=True)[0]
    grad_sigma_xy = torch.autograd.grad(sigma_tilde[:, 2], domain,
                                        grad_outputs=torch.ones_like(
                                            sigma_tilde[:, 2]),
                                        create_graph=True, retain_graph=True)[0]
    div_sigma = torch.hstack(((grad_sigma_xx[:, 0]+grad_sigma_xy[:, 1]).view(-1, 1),
                              (grad_sigma_xy[:, 0]+grad_sigma_yy[:, 1]).view(-1, 1)))


    return 1/domain.shape[0]*torch.norm(div_sigma, p=2)**2






def J_obs_F_u(metamodel, inputs):
    """
    Loss function for the boundary conditions on the stress field, computed from the estimated strain.
    """
    nbr_colloc_points, nbr_hlines = inputs.hlines.size()
    length = inputs.x_variable_max - inputs.x_variable_min
    dx = length / (nbr_colloc_points - 1)


    eval_integral = torch.zeros(nbr_hlines-1)
    for index_hline in range(1, nbr_hlines):
        sigma_tilde = eps_2_sigma(
            metamodel, inputs.hlines[:, [0, index_hline]], inputs)
        eval_integral[index_hline-1] = dx*torch.sum(sigma_tilde[:, 1])


    return 1/(nbr_hlines-1) * torch.norm(eval_integral - Fobs*length, 2)**2




def J_obs(metamodel, dic_model):
    """
    Loss function for DIC discrepancy on u. 
    """
    return dic_model.loss_DIC(dic_model.train_set, dic_model.I_0, dic_model.I_t, metamodel.model_u)




def J_BC(metamodel, inputs, is_sigma_trained):
    """
    Loss function for the boundary conditions on the stress field, estimated by the dedicated PINN.
    """
    Fobs_unit = torch.tensor([1.])
    Fobs_bd = is_sigma_trained*Fobs + (1-is_sigma_trained)*Fobs_unit
    sigma_top = Fobs_bd*metamodel.model_sigma(inputs.top_BC)
    sigma_bottom = Fobs_bd*metamodel.model_sigma(inputs.bottom_BC)
    sigma_left = Fobs_bd*metamodel.model_sigma(inputs.left_BC)
    sigma_right = Fobs_bd*metamodel.model_sigma(inputs.right_BC)


    loss_top = torch.norm(sigma_top[:, 1:] - torch.tensor([Fobs_bd, 0.]), 2)**2
    loss_bottom = torch.norm(
        sigma_bottom[:, 1:] - torch.tensor([Fobs_bd, 0.]), 2)**2
    loss_left = torch.norm(sigma_left[:, ::2] - torch.tensor([0., 0.]), 2)**2
    loss_right = torch.norm(sigma_right[:, ::2] - torch.tensor([0., 0.]), 2)**2


    result = (1/inputs.left_BC.shape[0]*loss_left +
              1/inputs.right_BC.shape[0]*loss_right +
              1/inputs.top_BC.shape[0]*loss_top +
              1/inputs.bottom_BC.shape[0]*loss_bottom)


    return result



def J_obs_F_sigma(metamodel, inputs, is_sigma_trained):
    """
    Loss function for the boundary conditions on the stress field, estimated by the dedicated PINN.
    """
    Fobs_unit = torch.tensor([1.])
    Fobs_bd = is_sigma_trained*Fobs + (1-is_sigma_trained)*Fobs_unit


    nbr_colloc_points, nbr_hlines = inputs.hlines.size()
    length = inputs.x_variable_max - inputs.x_variable_min
    dx = length / (nbr_colloc_points - 1)


    eval_integral = torch.zeros(nbr_hlines-1)
    for index_hline in range(1, nbr_hlines):
        sigma_tilde = Fobs_bd * \
            metamodel.model_sigma(inputs.hlines[:, [0, index_hline]])
        eval_integral[index_hline-1] = dx*torch.sum(sigma_tilde[:, 1])
    loss_hlines = torch.norm(eval_integral - Fobs_bd*length, 2)**2
    result = (1/(nbr_hlines-1) * loss_hlines)


    # result  = torch.var(eval_integral)


    return result




"""
#ajout perte sur E pas trop de sens physique car on ne peut pas vraiment calculer E en fonction de l'effort
def J_obs_F_E(metamodel, inputs):
    
    #Loss function for the boundary conditions on the stress field, estimated by the dedicated PINN.
   

    Fobs_bd = Fobs


    nbr_colloc_points, nbr_hlines = inputs.hlines.size()
    length = inputs.x_variable_max - inputs.x_variable_min
    dx = length / (nbr_colloc_points - 1)


    eval_integral = torch.zeros(nbr_hlines-1)
    for index_hline in range(1, nbr_hlines):
        E = Fobs_bd * \
            metamodel.model_E(inputs.hlines[:, [0, index_hline]])
        eval_integral[index_hline-1] = dx*torch.sum(E[:, 1])
    loss_hlines = torch.norm(eval_integral - Fobs_bd*length, 2)**2
    result = (1/(nbr_hlines-1) * loss_hlines)


    # result  = torch.var(eval_integral)


    return result

"""



def J_constitutive(metamodel, domain, inputs, is_sigma_trained,weigths={'eps_xx': 1, 'eps_yy': 1, 'eps_xy': 1}):
    """
    Loss function for the constitutive relation between the strain and the stress
    """
    epsilon_tilde = epsilon(metamodel, domain)
    sigma_tilde = Fobs*is_sigma_trained * \
        metamodel.model_sigma(domain) + (1-is_sigma_trained) * \
        metamodel.model_sigma(domain)
    
    #CHANGEMENT POUR E

    E=metamodel.model_E(domain)

    #E = metamodel.E_ref * E_function(domain, metamodel.E, metamodel, inputs)


    relation_1 = sigma_tilde[:, 0] - E / \
        (1-nu**2) * (epsilon_tilde[:, 0] + nu*epsilon_tilde[:, 1])
    relation_2 = sigma_tilde[:, 1] - E / \
        (1-nu**2) * (nu*epsilon_tilde[:, 0] + epsilon_tilde[:, 1])
    relation_3 = sigma_tilde[:, 2] - E/(1+nu) * epsilon_tilde[:, 2]


    return 1/domain.shape[0] * (weigths['eps_xx']*torch.norm(relation_1, p=2)**2 + weigths['eps_yy']*torch.norm(relation_2, p=2)**2 + weigths['eps_xy']*torch.norm(relation_3, p=2)**2)

import torch

def get_gl_from_rwc(inputs_rwc, dic_instance, image_type='I_0'):
    """
    Interpolates the grayscale values of an image (I_0 or I_t)
    at points given in real-world coordinates (inputs_rwc).

    inputs_rwc : (N,2) tensor
                 columns = [x_rwc, y_rwc] in mm
    dic_instance : instance of the DIC class
    image_type : 'I_0' (initial image) or 'I_t' (deformed image)

    Returns ---> (N,) interpolated grayscale values
    """

    # Selection of the image 
    if image_type == 'I_0':
        image = dic_instance.I_0
    elif image_type == 'I_t':
        image = dic_instance.I_t
    else:
        raise ValueError("image_type must be 'I_0' or 'I_t'")

    # Make sure the image is a torch tensor on the correct device
    device = inputs_rwc.device
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image, dtype=torch.float32, device=device)
    else:
        image = image.to(device)

    # Conversion real world coordinates to pixel index
    pts_px = dic_instance.from_rwc_2_px(inputs_rwc)   # (N,2)
    # Convention: pts_px[:,0] = row index, pts_px[:,1] = col index

    H, W = dic_instance.nb_px_row, dic_instance.nb_px_col

    # Clamp to stay inside image bounds
    x = torch.clamp(pts_px[:,1], 0, W-1)  # column
    y = torch.clamp(pts_px[:,0], 0, H-1)  # row

    # INTERPOLATION
    x0 = torch.floor(x).long()
    x1 = torch.clamp(x0 + 1, max=W-1)
    y0 = torch.floor(y).long()
    y1 = torch.clamp(y0 + 1, max=H-1)
    # interpolation weights
    wx = x - x0.float()
    wy = y - y0.float()
    # intensities of the 4 neighboring pixels
    I00 = image[y0, x0]
    I10 = image[y0, x1]
    I01 = image[y1, x0]
    I11 = image[y1, x1]
    # interpolate horizontally then vertically
    I_top = I00 * (1 - wx) + I10 * wx
    I_bottom = I01 * (1 - wx) + I11 * wx
    I_interp = I_top * (1 - wy) + I_bottom * wy
    return I_interp   # (N,)

