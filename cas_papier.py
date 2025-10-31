import torch
import numpy as np
import matplotlib.pyplot as plt
from Class_Observations import Observations
from Class_Mesh import Mesh
from Class_Inputs import Inputs
from Class_Metamodel import MetaModel
from Mechanics_model import create_mesh_E, E_function
import display
from copy import deepcopy

''' Parameter '''
device = torch.device('cpu')
plot_data = True

length = 20
heigth = 50

n_E = [20, 40]

''' Observations and inputs '''
obs_dci = Observations('solution_reference_papier.txt', 1000, stdu=0.0)
inputs = Inputs(device, N_coloc=[40, 100], N_coloc_bc=[100, 100, 500, 500], nbr_hlines=500,
                variable_boundaries=[[0., length], [0., heigth]])

if plot_data:
    mesh = Mesh()
    mesh.readmesh('plate.mesh')

    fig, axs = plt.subplots(2, 1, figsize=(5, 10))
    plt.subplots_adjust(hspace=0.4)

    ax = axs[0]
    ax = mesh.plot(ax)
    ax.scatter(obs_dci.data[:, 0], obs_dci.data[:, 1],
               s=2**5, color='r', label='Observation points')
    ax.legend(bbox_to_anchor=(0, 1.02, 1., 1.02),
              loc='lower left', borderaxespad=0.)
    ax.set_xlim(-10, 30)
    ax.set_xlabel(r'$x$ \ [mm]')
    ax.set_ylabel(r'$y$ \ [mm]')

    ax = axs[1]
    ax.scatter(inputs.all[:, 0].detach().numpy(), inputs.all[:, 1].detach(
    ).numpy(), s=2**5, color='g', label='Colocation points')
    ax.scatter(inputs.top_BC[:, 0].detach().numpy(
    ), inputs.top_BC[:, 1].detach().numpy(), s=2**5, label='Top boundary')
    ax.scatter(inputs.bottom_BC[:, 0].detach().numpy(
    ), inputs.bottom_BC[:, 1].detach().numpy(), s=2**5, label='Bottom boundary')
    ax.scatter(inputs.right_BC[:, 0].detach().numpy(
    ), inputs.right_BC[:, 1].detach().numpy(), s=2**5, label='Right boundary')
    ax.scatter(inputs.left_BC[:, 0].detach().numpy(
    ), inputs.left_BC[:, 1].detach().numpy(), s=2**5, label='Left boundary')
    ax.set_xlabel(r'$x$ \ [mm]')
    ax.set_ylabel(r'$y$ \ [mm]')
    ax.set_xlim(-10, 30)
    ax.legend(ncols=2, bbox_to_anchor=(0, 1.02, 1., 1.02),
              loc='lower left', borderaxespad=0.)

    fig, axs = plt.subplots(1, 2)
    ax = axs[0]
    h = ax.scatter(obs_dci.data[:, 0], obs_dci.data[:, 1],
                   c=obs_dci.data[:, 2], cmap='plasma')
    plt.colorbar(h, ax=ax)
    ax.axis("equal")
    ax.set_xlabel(r'$x$ \ [mm]')
    ax.set_ylabel(r'$y$ \ [mm]')
    ax.set_title(r'$u_{x}$ \ [mm]')

    ax = axs[1]
    h = ax.scatter(obs_dci.data[:, 0], obs_dci.data[:, 1],
                   c=obs_dci.data[:, 3], cmap='plasma')
    plt.colorbar(h, ax=ax)
    ax.axis("equal")
    ax.set_xlabel(r'$x$ \ [mm]')
    ax.set_ylabel(r'$y$ \ [mm]')
    ax.set_title(r'$u_{y}$ \ [mm]')

    plt.show()
# %%


def Efunc(x):
    return (20000 - 10000*(1/(1+np.exp(-(x[:, 1]+heigth/(2*length)*x[:, 0]-heigth/3)*5))
                           - 1/(1+np.exp(-(x[:, 1]+heigth/(2*length)*x[:, 0]-heigth/3-heigth/8)*5)))) - 15000*np.exp(-(((x[:, 0]-2*length/3)/3)**2+((x[:, 1]-2*heigth/3)/5)**2))


''' Initialization of the metamodel '''
metamodel = MetaModel(device, inputs, layers=[2, 50, 50, 50],
                      E_0=20000.*torch.ones(((n_E[0]+1)*(n_E[1]+1))), E_ref=20000, E_interpolation='P1', n_E=n_E,
                      Fourier_features=True, sigma_FF_u=1, sigma_FF_sigma=1,
                      verbose=1,
                      Efunc=Efunc,
                      seed=3112001)

if plot_data:
    fig, ax = plt.subplots()
    ax = metamodel.mesh_E.plot(ax)
    ax.set_title('E mesh')
    ax.set_xlabel(r'$x$ \ [mm]')
    ax.set_ylabel(r'$y$ \ [mm]')
    ax.axis('equal')
    plt.plot()

# %%
is_pretrained = True

if not (is_pretrained):
    ''' Pretraining '''
    metamodel.pretrain_u(inputs, obs_dci, pre_train_iter=100000)

    metamodel.is_sigma_trained = True
    metamodel.pretrain_sigma(inputs, pre_train_iter=1000, lambdas={
                             'res': 0.01, 'obs': 0, 'obs_F': 0, 'BC': 1., 'lines': 1., 'constitutive': 1})

else:
    metamodel.model_u = torch.load('model_u_papier_pretrained_figure.pth')
    metamodel.model_sigma = torch.load(
        'model_sigma_papier_pretrained_figure.pth')

display.display_sigma(metamodel, inputs, 5.)
display.display_epsilon(metamodel, inputs)
display.display_u(metamodel, inputs, display_error=True, obs=obs_dci)
display.display_E(metamodel, inputs)

# %%
is_trained = False


if not (is_trained):
    metamodel.train_model(inputs, obs_dci, alter_steps=10,
                          alter_freq=(60, 0, 100, 100),
                          lambdas_identif_E={
                              'res': 0, 'obs': 0, 'obs_F': 10, 'BC': 0, 'lines': 0, 'constitutive': 10, 'tikhonov': 0},
                          lambdas_update_sigma={
                              'res': 0.01, 'obs': 0, 'obs_F': 0, 'BC': 0.01, 'lines': 0.01, 'constitutive': 10, 'tikhonov': 0},
                          lambdas_update_u={'res': 0, 'obs': 0.01, 'obs_F': 10, 'BC': 0, 'lines': 0, 'constitutive': 10, 'tikhonov': 0})
else:
    metamodel.model_u = torch.load('model_u_trained_figure.pth')
    metamodel.model_sigma = torch.load('model_sigma_trained_figure.pth')
    metamodel.E = torch.from_numpy(np.loadtxt('E_papier_figure.txt'))
    metamodel.E.requires_grad = True

display.display_sigma(metamodel, inputs, 5.)
display.display_epsilon(metamodel, inputs)
display.display_u(metamodel, inputs, display_error=True, obs=obs_dci)
display.display_E(metamodel, inputs)
