import torch


class Inputs:
    """
    Inputs class, used to define the collocation points sampled in the physical domain. 
    """

    def __init__(self, device, N_coloc, N_coloc_bc, variable_boundaries, nbr_hlines=10,
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

        # Get the collocation points in the domain
        x_grid = torch.linspace(self.x_variable_min,
                                self.x_variable_max,
                                N_coloc[0])

        y_grid = torch.linspace(self.y_variable_min,
                                self.y_variable_max,
                                N_coloc[1])

        [X, Y] = torch.meshgrid(x_grid, y_grid)

        self.grid = torch.hstack(
            (X.reshape(X.numel(), 1), Y.reshape(Y.numel(), 1)))

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

        # Get the collocation points on horizontal lines
        y_points = torch.linspace(
            self.y_variable_min, self.y_variable_max, nbr_hlines)

        self.hlines = torch.hstack((torch.linspace(self.x_variable_min, self.x_variable_max,
                                   N_coloc_bc[0]).view(-1, 1), y_points[0]*torch.ones(N_coloc_bc[0]).view(-1, 1)))
        for y_index in range(1, nbr_hlines):
            self.hlines = torch.hstack(
                (self.hlines, y_points[y_index]*torch.ones(N_coloc_bc[0]).view(-1, 1)))

        self.hlines.requires_grad = True
