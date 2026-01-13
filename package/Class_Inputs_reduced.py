import torch
import numpy as np
import cv2

class Inputs:
    """
    Inputs class, used to define the collocation points sampled in the physical domain. 
    """

    def __init__(self, device, N_coloc, N_coloc_bc, variable_boundaries, I0,It, nbr_hlines=10,
                 test_size=0, seed=None):
        """
        -device: refering to the device used for torch computations
        -N_coloc: list of 2 elements corresponding to the number of colocation points along each axis
        -N_coloc_bc: list of 4 elements corresponding to the number of colocation points on each boundary
        -variable_boundaries: coordinates of the corner of the plate
        -nbr_hlines: number of evenly spaced horizontal lines on which are inforced the global equilibrium equation (J_{obs}^{F,v})
        -test_size: ratio (between 0 and 1) of the total points used for testing
        -seed: impose the seed for reproductibility
        """
        # Seed for random colocation points and training/testing sets reproductibility
        if (seed != None):
            torch.manual_seed(seed)

        self.device = device
        self.test_size = test_size

        self.N_coloc = N_coloc
        self.x_variable_min = variable_boundaries[0][0]
        self.x_variable_max = variable_boundaries[0][1]
        self.y_variable_min = variable_boundaries[1][0]
        self.y_variable_max = variable_boundaries[1][1]
        
        mask_diff = np.abs(It-I0) > 0.0

        marge = 10
        # Le noyau doit avoir un rayon de 10, donc une taille d'environ 2*10+1
        kernel_size = 2 * marge + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        # On dilate le masque des différences
        mask_roi = cv2.dilate(mask_diff.astype(np.uint8), kernel).astype(bool)

        # Get the collocation points in the domain
        x_grid = torch.linspace(self.x_variable_min,
                                self.x_variable_max,
                                N_coloc[0])

        y_grid = torch.linspace(self.y_variable_min,
                                self.y_variable_max,
                                N_coloc[1])

        [X, Y] = torch.meshgrid(x_grid, y_grid)

        background_grid = torch.hstack((X.reshape(X.numel(), 1), Y.reshape(Y.numel(), 1)))
        
        roi_y_idx, roi_x_idx = np.where(mask_roi)

        H, W = mask_diff.shape 

        x_ratio = roi_x_idx / (W - 1)
        y_ratio = roi_y_idx / (H - 1) 

        roi_x_phy = self.x_variable_min + x_ratio * (self.x_variable_max - self.x_variable_min)
        roi_y_phy = self.y_variable_min + y_ratio * (self.y_variable_max - self.y_variable_min)

        roi_points = torch.tensor(np.column_stack((roi_x_phy, roi_y_phy)), dtype=torch.float32)



        self.grid = torch.vstack((background_grid, roi_points))

        self.all = self.grid.detach().clone()

        self.all.requires_grad = True

        self.train = self.grid
        self.train.requires_grad = True

        # Get the collocation points on the boundaries
        self.top_BC = torch.hstack((torch.linspace(self.x_variable_min, self.x_variable_max,
                                   N_coloc_bc[0]).view(-1, 1), self.y_variable_max*torch.ones(N_coloc_bc[0]).view(-1, 1)))
        self.bottom_BC = torch.hstack((torch.linspace(self.x_variable_min, self.x_variable_max,
                                      N_coloc_bc[1]).view(-1, 1), self.y_variable_min*torch.ones(N_coloc_bc[1]).view(-1, 1)))
        self.left_BC = torch.hstack((self.x_variable_min*torch.ones(N_coloc_bc[2]).view(-1, 1), torch.linspace(
            self.y_variable_min, self.y_variable_max, N_coloc_bc[2]).view(-1, 1)))
        self.right_BC = torch.hstack((self.x_variable_max*torch.ones(N_coloc_bc[3]).view(-1, 1), torch.linspace(
            self.y_variable_min, self.y_variable_max, N_coloc_bc[3]).view(-1, 1)))
        self.top_BC.requires_grad = True
        self.bottom_BC.requires_grad = True
        self.right_BC.requires_grad = True
        self.left_BC.requires_grad = True


        y_points = torch.linspace(
            self.y_variable_min, self.y_variable_max, nbr_hlines)

        self.hlines = torch.hstack((torch.linspace(self.x_variable_min, self.x_variable_max,
                                   N_coloc_bc[0]).view(-1, 1), y_points[0]*torch.ones(N_coloc_bc[0]).view(-1, 1)))
        for y_index in range(1, nbr_hlines):
            self.hlines = torch.hstack(
                (self.hlines, y_points[y_index]*torch.ones(N_coloc_bc[0]).view(-1, 1)))

        self.hlines.requires_grad = True

        """

        # Get the collocation points on horizontal lines

        
        y_points = torch.linspace(
            self.y_variable_min, self.y_variable_max, nbr_hlines)
        x_points = torch.linspace(
            self.x_variable_min, self.x_variable_max, N_coloc[0])
        
        #Récupération des points de la grille de base
        [Xh, Yh] = torch.meshgrid(x_points, y_points)
        fond_hlines_points = torch.hstack((Xh.reshape(-1, 1), Yh.reshape(-1, 1)))

        #Ajout des points de la région d'intérêt sur ces lignes
        H_ROI, W_ROI = mask_roi.shape

        y_ratio = (y_points - self.y_variable_min) / (self.y_variable_max - self.y_variable_min)
        i = (y_ratio * (H_ROI - 1)).round().long()
        i = torch.clamp(i, 0, H_ROI - 1).numpy()

        #On crée un masque vide et on active seulement les lignes qu'on veut scanner
        line_selection_mask = np.zeros_like(mask_roi, dtype=bool)
        line_selection_mask[i, :] = True

        intersection_mask = mask_roi & line_selection_mask

        i_idx, j_idx = np.where(intersection_mask)

        if len(j_idx) > 0:
            # convertion en rwc
            ratio_j = j_idx / (W_ROI - 1)
            ratio_i = i_idx / (H_ROI - 1)
            
            x_roi = self.x_variable_min + ratio_j * (self.x_variable_max - self.x_variable_min)
            y_roi = self.y_variable_min + ratio_i * (self.y_variable_max - self.y_variable_min)
            
            roi_hlines_points = torch.tensor(
                np.column_stack((x_roi, y_roi)), 
                dtype=torch.float32
            )
            
            #fusion des points de la grille de fond et de la roi
            self.hlines = torch.vstack((fond_hlines_points, fond_hlines_points))
        else:
            self.hlines = fond_hlines_points


    

        self.hlines.requires_grad = True
        """