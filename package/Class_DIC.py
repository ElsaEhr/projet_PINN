import numpy as np
import torch 
from scipy.optimize import fsolve
from scipy.interpolate import LinearNDInterpolator, RectBivariateSpline

""" Utilities functions """
#%%
def N0(x, dx, dy):
    res = 1/(dx*dy)*(dx/2 - x[:, 0])*(dy/2 - x[:, 1])
    return res


def N1(x, dx, dy):
    res = 1/(dx*dy)*(dx/2 + x[:, 0])*(dy/2 - x[:, 1])
    return res


def N2(x, dx, dy):
    res = 1/(dx*dy)*(dx/2 + x[:, 0])*(dy/2 + x[:, 1])
    return res


def N3(x, dx, dy):
    res = 1/(dx*dy)*(dx/2 - x[:, 0])*(dy/2 + x[:, 1])
    return res



#%% 
def generate_speckle(Nx,Ny):
    """
    Inputs: Nx, Ny (int): number of pixel along the lines and the columns of a picture
    """
    density = np.random.randint(1000, 3000)
        
    P = np.hstack((np.random.randint(0,Nx,density).reshape((density,1)), np.random.randint(0,Ny,density).reshape((density,1))))
    B = 1.2 + (6.8-1.2)*np.random.rand(density, 2) 
    G = 255*np.random.rand(density)
        
    I = np.zeros((Nx,Ny))
    x = np.arange(Nx)
    y = np.arange(Ny)
    [X, Y] = np.meshgrid(x,y)
    X = X.T
    Y = Y.T
        
    for j in range(density):
        ellips = (X-P[j,0])**2/B[j,0]**2 + (Y-P[j,1])**2/B[j,1]**2 
        index  = np.where(ellips < 1)
        I[index] = G[j]
    return I

def generate_deformed_image(I_0, u_func):
    """
    Inputs: - I_0 (numpy ndarray): reference image 
            - u_func: ground truth displacement 
    Outputs: I_t: deformed image of I_0 with respect to the displacement modeled by model_u
    """
    I_t = np.zeros_like(I_0)
    
    x = np.arange(0, I_0.shape[0])
    y = np.arange(0, I_0.shape[1])
    X, Y = np.meshgrid(x, y, indexing = 'ij')
    inputs = np.vstack((X.flatten(), Y.flatten())).T
    inputs_t = np.zeros_like(inputs)
    index_inputs = 0

    interpolator = LinearNDInterpolator(inputs, I_0.flatten(), fill_value = 0)
    
    def phi(x, x0): # define the non linear function to solve for each pixel
        return x + u_func(x) - x0

    for i in range(I_0.shape[0]):
        for j in range(I_0.shape[1]):
            x = np.array([i,j])
            x_t = fsolve(phi, x0 = x, args = (x))
            inputs_t[index_inputs, :] = x_t
            grey_level = interpolator(np.array([x_t[0], x_t[1]]))
            I_t[i,j] = grey_level
            index_inputs += 1
    
    return I_t


class DIC():
    """ 
    Build a DIC framework based on a image of reference I_0, a deformed image I_t. The DIC class also provides a 
    function to compute the residual in grey level, provided a displacement model.
    
    Few key points: 1) The images are expressed in pixel coordinates, so a transformation must established between 
    real-world coordinate system and pixel coordinate system. We assume simply consists of a dilatation operation 
    along x-axis (ratio_x) and y-axis (ratio_y).  
    
    2) The origin of the two systems are different (cf the scheme). 
                                              o ----> (j)
    (0,0) in pixel coordinates -->            | +-----------------------+
                                              | |                       |
                                              v |                       |
                                            (i) |                       |
                                                |                       |
                                                |                       |
                                                |                       |
                                            (y) |                       |
                                              ^ |                       |
                                              | |                       |
    (0,0) in real world coordinates -->       | +-----------------------+
                                              o ----> (x)
                                              
    3) For a displacement (u_px, v_px) computed in pixel coordinates, the latter is transformed in real work coordinates 
    (u_rwc, v_rwc) as u_rwc = v_px / ratio_x and v_rwc = - u_px / ratio_y
    """
    def __init__(self, I_0, I_t, train_set, coord_rwc):
        self.I_0 = I_0
        self.I_t = I_t
        self.train_set = train_set
        
        self.nb_px_row = I_0.shape[0]
        self.nb_px_col = I_0.shape[1]
        self.L = coord_rwc[0]
        self.H = coord_rwc[1]
        self.ratio_x = self.L/self.nb_px_col # [l]/[px]
        self.ratio_y = self.H/self.nb_px_row # [l]/[px]
        
    def from_px_2_rwc(self,inputs_px):
        """ Convert the pixel coordinates to real-world coordinates 
        Inputs: inputs_px : coordinates in pixel
        Outputs: inputs_rwc: converted coordinated into real-world coordinates
        """
        return torch.hstack((inputs_px[:,1].view(-1,1) * self.ratio_x, 
                                   self.H - inputs_px[:,0].view(-1,1) * self.ratio_y))
    def from_rwc_2_px(self,inputs_rwc):
        """ Convert the real-world coordinates to pixel coordinates 
        Inputs: inputs_rwc : coordinates in real-world
        Outputs: inputs_px: converted coordinated into pixel
        """
        return torch.hstack(((inputs_rwc[:,1].view(-1,1) - self.H) / self.ratio_y, 
                                   inputs_rwc[:,0].view(-1,1) / self.ratio_x))
    
    def disp_from_rwc_2_px(self, disp_rwc):
        """ Convert a displacement in real-world coordinates to pixel coordinates 
        Inputs: disp_rwc : displacements in real-world coordinate system
        Outputs: disp_px : converted coordinated into pixel coordinates
        """
        return torch.hstack((-disp_rwc[:,1].view(-1,1) / self.ratio_y, 
                                   disp_rwc[:,0].view(-1,1) / self.ratio_x))    
        
                    
    def loss_DIC(self, inputs, I_0_sub, I_t_sub, model_u):
        """ Compute the error in grey-level 
        Inputs: - inputs (torch tensor of size [nb_inputs, 2]): coordinates (in pixel) of the pixels involved in I_O_sub 
        (and I_t_sub) within I_0 and I_t
                - I_0_sub (torch tensor): sub-image of I_0
                - I_t_sub (torch tensor): sub_image of I_t
                - model_u (PINN): displacement modeled by the PINN model_u
        Outputs: loss (float) = || I_0(x) - I_t(x+u(x)) ||
        """
        N_i = I_0_sub.shape[0]
        N_j = I_0_sub.shape[1]

        inputs_centered = inputs - \
            torch.tensor([torch.min(inputs[:, 0]), torch.min(inputs[:, 1])])

        I_0_vec = I_0_sub.flatten()
        I_t_vec = I_t_sub.flatten()

        disp = model_u(self.from_px_2_rwc(inputs))
        inputs_translate = inputs_centered + self.disp_from_rwc_2_px(disp)

        i_def = torch.floor(inputs_translate[:, 0]).type(torch.int)
        j_def = torch.floor(inputs_translate[:, 1]).type(torch.int)

        index_top_left = i_def*N_j + j_def
        index_bot_left = (i_def+1)*N_j + j_def
        index_bot_right = (i_def+1)*N_j + j_def + 1
        index_top_right = i_def*N_j + j_def + 1

        good_index_i = np.intersect1d(torch.where(
            0 <= i_def), torch.where(i_def <= N_i-2))
        good_index_j = np.intersect1d(torch.where(
            0 <= j_def), torch.where(j_def <= N_j-2))
        good_index = np.intersect1d(good_index_i, good_index_j)

        I_0_predict = torch.zeros_like(I_0_vec)

        coord_locales = inputs_translate[good_index] - \
            inputs_centered[index_top_left[good_index], :] - \
            torch.tensor([1/2, 1/2])

        I_0_predict[good_index] += N0(coord_locales, dx=1, dy=1)*I_t_vec[index_top_left[good_index]] + \
            N1(coord_locales, dx=1, dy=1)*I_t_vec[index_bot_left[good_index]] + \
            N2(coord_locales, dx=1, dy=1)*I_t_vec[index_bot_right[good_index]] + \
            N3(coord_locales, dx=1, dy=1) * \
            I_t_vec[index_top_right[good_index]]

        return 1/inputs.shape[0]*torch.norm(I_0_vec - I_0_predict, 2)**2
    
    def descente_gradient(self, inputs, I_0_sub, I_t_sub, optimizer, model_u, loss_monitoring):
        def closure():
            optimizer.zero_grad()

            loss_eval = self.loss_DIC(inputs, I_0_sub, I_t_sub, model_u)
            loss_eval.backward()
            loss_monitoring.append(loss_eval.item())

            return loss_eval

        optimizer.step(closure)
        
    def reconstruction(self, inputs, I_0_sub, I_t_sub, model_u):
        """ Compute the error in grey-level 
        Inputs: - inputs (torch tensor of size [nb_inputs, 2]): coordinates (in pixel) of the pixels involved in I_O_sub 
        (and I_t_sub) within I_0 and I_t
                - I_0_sub (torch tensor): sub-image of I_0
                - I_t_sub (torch tensor): sub_image of I_t
                - model_u (PINN): displacement modeled by the PINN model_u
        Outputs: loss (float) = || I_0(x) - I_t(x+u(x)) ||
        """
        N_i = I_0_sub.shape[0]
        N_j = I_0_sub.shape[1]

        inputs_centered = inputs - \
            torch.tensor([torch.min(inputs[:, 0]), torch.min(inputs[:, 1])])

        I_0_vec = I_0_sub.flatten()
        I_t_vec = I_t_sub.flatten()

        disp = model_u(self.from_px_2_rwc(inputs))
        inputs_translate = inputs_centered + self.disp_from_rwc_2_px(disp)

        i_def = torch.floor(inputs_translate[:, 0]).type(torch.int)
        j_def = torch.floor(inputs_translate[:, 1]).type(torch.int)

        index_top_left = i_def*N_j + j_def
        index_bot_left = (i_def+1)*N_j + j_def
        index_bot_right = (i_def+1)*N_j + j_def + 1
        index_top_right = i_def*N_j + j_def + 1

        good_index_i = np.intersect1d(torch.where(
            0 <= i_def), torch.where(i_def <= N_i-2))
        good_index_j = np.intersect1d(torch.where(
            0 <= j_def), torch.where(j_def <= N_j-2))
        good_index = np.intersect1d(good_index_i, good_index_j)

        I_0_predict = torch.zeros_like(I_0_vec)

        coord_locales = inputs_translate[good_index] - \
            inputs_centered[index_top_left[good_index], :] - \
            torch.tensor([1/2, 1/2])

        I_0_predict[good_index] += N0(coord_locales, dx=1, dy=1)*I_t_vec[index_top_left[good_index]] + \
            N1(coord_locales, dx=1, dy=1)*I_t_vec[index_bot_left[good_index]] + \
            N2(coord_locales, dx=1, dy=1)*I_t_vec[index_bot_right[good_index]] + \
            N3(coord_locales, dx=1, dy=1) * \
            I_t_vec[index_top_right[good_index]]
            
        return I_0_predict.detach().numpy()