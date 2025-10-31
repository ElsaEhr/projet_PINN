import torch
import numpy as np


class Observations:
    """
    Observations class, used to generate the observations given the reference solution. 
    """

    def __init__(self, filename, N_obs=-1, stdu=0.,
                 seed=None):
        """
        -filename: file .txt which contains the u_x and u_y displacements at points (x,y) (obtained with FEM simulation)
        -N_obs: number of sample of measures. Take all sample available if N_obs = -1
        -stdu: parameter to add gaussian noise with standard deviation equal to stdu
        -seed: impose the seed for reproductibility
        """
        # Seed for random obvservations reproductibility
        if (seed != None):
            torch.manual_seed(seed)

        self.N_obs = N_obs
        self.stdu = stdu

        data = np.loadtxt(filename, skiprows=1)
        x = data[:, 0]
        y = data[:, 1]
        u_x = data[:, 2]
        u_y = data[:, 3]

        coords = np.vstack((x, y)).T
        if N_obs == -1:
            obs_index = np.arange(0, np.size(u_x))
        else:
            obs_index = np.arange(0, np.size(u_x), np.size(u_x)//N_obs)

        obs_x = u_x[obs_index]+np.multiply(np.random.randn(
            np.shape(u_x[obs_index])[0],), np.abs(u_x[obs_index])) * self.stdu
        obs_y = u_y[obs_index]+np.multiply(np.random.randn(
            np.shape(u_y[obs_index])[0],), np.abs(u_y[obs_index])) * self.stdu

        self.data = torch.hstack((torch.from_numpy(coords[obs_index, 0]).type(torch.float32).view(-1, 1),
                                  torch.from_numpy(coords[obs_index, 1]).type(
                                      torch.float32).view(-1, 1),
                                  torch.from_numpy(obs_x).type(
                                      torch.float32).view(-1, 1),
                                  torch.from_numpy(obs_y).type(torch.float32).view(-1, 1)))

        self.weights = torch.ones(np.size(obs_index)).view(-1, 1)

    def Adaptive_Weights(self, missfit):

        missfit = missfit.detach().clone()

        self.weights = torch.abs(missfit)/torch.min(torch.abs(missfit))

        # self.weights = missfit**2/torch.min(missfit**2)
