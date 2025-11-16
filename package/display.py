import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colors

from package.Mechanics_model import create_mesh_E, epsilon ,E_function

def display_data(model, inputs, ref_solution, obs): 
    """
    Function used to display the data for the inverse problem. 
    """
        
    if (model.E_interpolation == 'P0'):
        subdomains = torch.linspace(min(ref_solution['domain']).item(), 
                                    max(ref_solution['domain']).item(),
                                    model.dim_k + 1)

    if (model.E_interpolation == 'P1'):
        subdomains = torch.linspace(min(ref_solution['domain']).item(), 
                                    max(ref_solution['domain']).item(),
                                    model.dim_k)
    fig, ax = plt.subplots()
    if (obs.N_obs < 5):
        if (obs.N_obs != 0):
            ax.set_title('Reference solution and data, $E = {} \ Pa$'.format(list(np.around(ref_solution['parameter'].detach().clone().cpu().numpy(),2))))
        else:
            ax.set_title('Reference solution, $E = {} \ Pa$'.format(list(np.around(ref_solution['parameter'].detach().clone().cpu().numpy(),2))))
    else:
        ax.set_title('Reference solution and observations')
    ax.plot(ref_solution['domain'].detach().clone().cpu().numpy(), 
            ref_solution['solution'].detach().clone().cpu().numpy(), 
            '--k', label = '$u_{ref}(x)$')
    if (obs.N_obs != 0):
        ax.plot(inputs.all.detach().clone().cpu().numpy(), 
                   inputs.all.detach().clone().cpu().numpy()*0.-0.05,'.k', 
                   label = '$x_{col}$')
        ax.plot(obs.data[:, 0].detach().clone().cpu().numpy(), 
                   obs.data[:, 1].detach().clone().cpu().numpy()*0.-0.05,'o',color='#1f77b4', 
                   label = '$x_{obs}$')
        ax.scatter(obs.data[:, 0].detach().clone().cpu().numpy(), 
                   obs.data[:, 1].detach().clone().cpu().numpy(), 
                   label = '$u_{obs}$', color='#1f77b4')
        ax.plot(inputs.variable_max,0-0.05,'sr', 
                   label = '$x_{obs}^F$')
        ax.scatter(subdomains.detach().clone().cpu().numpy(), 
               np.zeros(subdomains.shape[0])-0.05, marker = '|', c = 'k', s = 100)
    
    
    ax.set_ylim(-0.20,max(ref_solution['solution'].detach().clone().cpu().numpy())+0.03)
    ax.set_xlabel(r'$x \ [mm]$')
    ax.set_ylabel(r'$u \ [mm]$')
    ax.grid()
    
    ax2 = ax.twinx()

    ax2.set_ylabel(r'$E \ [GPa]$', y = 0.10)  
    ax2.plot(ref_solution['domain'].flatten().detach().clone().cpu().numpy(),
             ref_solution['parameter_function'].flatten().detach().clone().cpu().numpy()/1000,
             '-.', label = r'$E_{ref}(x)$', c = 'grey')
    ax2.set_ylim(0, 110)
    custom_ticks = np.arange(0, 30, 5)  
    ax2.set_yticks(custom_ticks) 
    
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left', 
               ncol = 3, prop={'size': 9})
        
    plt.show()
    

def display_sigma(metamodel, inputs, Fobs):
    ''' Function to display the stress tensor components 
    Inputs: - metamodel
    - inputs
    - Fobs: scaling factor 
    '''

    fig, axs = plt.subplots(1,3,figsize=(15,5))
    plt.subplots_adjust(hspace=0.4)

    x, y        = torch.linspace(inputs.x_variable_min, inputs.x_variable_max, 100), torch.linspace(inputs.y_variable_min, inputs.y_variable_max, 100)
    [X, Y]      = torch.meshgrid(x,y)
    domain      = torch.hstack((X.reshape((X.numel(), 1)), Y.reshape((Y.numel(), 1))))
    sigma_tilde = Fobs*metamodel.model_sigma(domain)
    sigma_xx    = sigma_tilde[:,0].detach().numpy()
    sigma_yy    = sigma_tilde[:,1].detach().numpy()
    sigma_xy    = sigma_tilde[:,2].detach().numpy()

    bounds = np.linspace(np.min(sigma_xx), np.max(sigma_xx), 21)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    h = axs[0].scatter(domain[:,0].detach().numpy(), domain[:,1].detach().numpy(), c=sigma_xx, s=2**4, cmap='plasma', norm = norm)
    plt.colorbar(h, ax=axs[0])
    axs[0].set_xlabel(r'$x$ \ [mm]')
    axs[0].set_ylabel(r'$y$ \ [mm]')
    axs[0].set_title(r'$\sigma_{xx}$ \ [MPa]')
    axs[0].axis('equal')

    bounds = np.linspace(np.min(sigma_yy), np.max(sigma_yy), 21)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    h = axs[1].scatter(domain[:,0].detach().numpy(), domain[:,1].detach().numpy(), c=sigma_yy, s=2**4, cmap='plasma', norm = norm)
    plt.colorbar(h, ax=axs[1])
    axs[1].set_xlabel(r'$x$ \ [mm]')
    axs[1].set_ylabel(r'$y$ \ [mm]')
    axs[1].set_title(r'$\sigma_{yy}$ \ [MPa]')
    axs[1].axis('equal')

    bounds = np.linspace(np.min(sigma_xy), np.max(sigma_xy), 21)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    h = axs[2].scatter(domain[:,0].detach().numpy(), domain[:,1].detach().numpy(), c=sigma_xy, s=2**4, cmap='plasma', norm = norm)
    plt.colorbar(h, ax=axs[2])
    axs[2].set_xlabel(r'$x$ \ [mm]')
    axs[2].set_ylabel(r'$y$ \ [mm]')
    axs[2].set_title(r'$\sigma_{xy}$ \ [MPa]')
    axs[2].axis('equal')


    plt.show()

def display_u(metamodel, inputs, obs, display_error = True):
    ''' Function to display the displacement components
    Inputs: - metamodel
    - inputs
    - obs
    - display_error: True to display the pointwise absolute error with the reference solution '''

    if display_error:
        fig, axs = plt.subplots(2,2,figsize=(10,10))
        plt.subplots_adjust(hspace=0.4, wspace=0.4)

        x, y        = torch.linspace(inputs.x_variable_min, inputs.x_variable_max, 100), torch.linspace(inputs.y_variable_min, inputs.y_variable_max, 100)
        [X, Y]      = torch.meshgrid(x,y)
        domain      = torch.hstack((X.reshape((X.numel(), 1)), Y.reshape((Y.numel(), 1))))
        u_tilde     = metamodel.model_u(domain)
        u_x         = u_tilde[:,0].detach().numpy()
        u_y         = u_tilde[:,1].detach().numpy()

        bounds = np.linspace(np.min(u_x), np.max(u_x), 21)
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
        h = axs[0,0].scatter(domain[:,0].detach().numpy(), domain[:,1].detach().numpy(), c=u_x, s=2**4, cmap='plasma', norm =norm)
        plt.colorbar(h, ax=axs[0,0])
        axs[0,0].set_xlabel(r'$x$ \ [mm]')
        axs[0,0].set_ylabel(r'$y$ \ [mm]')
        axs[0,0].set_title(r'$u_{x}$ \ [mm]')
        axs[0,0].axis('equal')

        bounds = np.linspace(np.min(u_y), np.max(u_y), 21)
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
        h = axs[0,1].scatter(domain[:,0].detach().numpy(), domain[:,1].detach().numpy(), c=u_y, s=2**4, cmap='plasma', norm =norm)
        plt.colorbar(h, ax=axs[0,1])
        axs[0,1].set_xlabel(r'$x$ \ [mm]')
        axs[0,1].set_ylabel(r'$y$ \ [mm]')
        axs[0,1].set_title(r'$u_{y}$ \ [mm]')
        axs[0,1].axis('equal')

        predictions  = metamodel.model_u(obs.data[:,:2]).detach().numpy()
        observations = obs.data[:, 2:].detach().numpy()
        
        h = axs[1,0].scatter(obs.data[:,0].detach().numpy(), 
                             obs.data[:,1].detach().numpy(), 
                             c = np.abs(predictions[:,0] - observations[:,0]), s=2**4, cmap='plasma')
        plt.colorbar(h, ax=axs[1,0])
        axs[1,0].set_xlabel(r'$x$ \ [mm]')
        axs[1,0].set_ylabel(r'$y$ \ [mm]')
        axs[1,0].set_title(r'Erreur $u_{x}$ \ [mm]')
        axs[1,0].axis('equal')

        h = axs[1,1].scatter(obs.data[:,0].detach().numpy(), 
                             obs.data[:,1].detach().numpy(), 
                             c = np.abs(predictions[:,1] - observations[:,1]), s=2**4, cmap='plasma')
        plt.colorbar(h, ax=axs[1,1])
        axs[1,1].set_xlabel(r'$x$ \ [mm]')
        axs[1,1].set_ylabel(r'$y$ \ [mm]')
        axs[1,1].set_title(r'Erreur $u_{x}$ \ [mm]')
        axs[1,1].axis('equal')

        plt.show()
    
    else:
        fig, axs = plt.subplots(1,2,figsize=(10,5))
        plt.subplots_adjust(wspace=0.4)

        x, y        = torch.linspace(inputs.x_variable_min, inputs.x_variable_max, 100), torch.linspace(inputs.y_variable_min, inputs.y_variable_max, 100)
        [X, Y]      = torch.meshgrid(x,y)
        domain      = torch.hstack((X.reshape((X.numel(), 1)), Y.reshape((Y.numel(), 1))))
        u_tilde     = metamodel.model_u(domain)
        u_x         = u_tilde[:,0].detach().numpy()
        u_y         = u_tilde[:,1].detach().numpy()

        h = axs[0].scatter(domain[:,0].detach().numpy(), domain[:,1].detach().numpy(), c=u_x, s=2**4, cmap='plasma')
        plt.colorbar(h, ax=axs[0])
        axs[0].set_xlabel(r'$x$ \ [mm]')
        axs[0].set_ylabel(r'$y$ \ [mm]')
        axs[0].set_title(r'$u_{x}$ \ [mm]')
        axs[0].axis('equal')

        h = axs[1].scatter(domain[:,0].detach().numpy(), domain[:,1].detach().numpy(), c=u_y, s=2**4, cmap='plasma')
        plt.colorbar(h, ax=axs[1])
        axs[1].set_xlabel(r'$x$ \ [mm]')
        axs[1].set_ylabel(r'$y$ \ [mm]')
        axs[1].set_title(r'$u_{y}$ \ [mm]')
        axs[1].axis('equal')

        plt.show()

def display_epsilon(metamodel, inputs):
    '''Function to display the strain tensor components
    Inputs: - metamodel
    - inputs
    '''
    fig, axs = plt.subplots(1,3,figsize=(15,5))
    plt.subplots_adjust(hspace=0.4)

    x, y        = torch.linspace(inputs.x_variable_min, inputs.x_variable_max, 100, requires_grad=True), torch.linspace(inputs.y_variable_min, inputs.y_variable_max, 100, requires_grad=True)
    [X, Y]      = torch.meshgrid(x,y)
    domain      = torch.hstack((X.reshape((X.numel(), 1)), Y.reshape((Y.numel(), 1))))
    eps         = epsilon(metamodel, domain)
    eps_xx      = eps[:,0].detach().numpy()
    eps_yy      = eps[:,1].detach().numpy()
    eps_xy      = eps[:,2].detach().numpy()

    h = axs[0].scatter(domain[:,0].detach().numpy(), domain[:,1].detach().numpy(), c=eps_xx, s=2**4, cmap='plasma')
    plt.colorbar(h, ax=axs[0])
    axs[0].set_xlabel(r'$x$ \ [mm]')
    axs[0].set_ylabel(r'$y$ \ [mm]')
    axs[0].set_title(r'$\varepsilon_{xx}$')
    axs[0].axis('equal')

    h = axs[1].scatter(domain[:,0].detach().numpy(), domain[:,1].detach().numpy(), c=eps_yy, s=2**4, cmap='plasma')
    plt.colorbar(h, ax=axs[1])
    axs[1].set_xlabel(r'$x$ \ [mm]')
    axs[1].set_ylabel(r'$y$ \ [mm]')
    axs[1].set_title(r'$\varepsilon_{yy}$')
    axs[1].axis('equal')

    h = axs[2].scatter(domain[:,0].detach().numpy(), domain[:,1].detach().numpy(), c=eps_xy, s=2**4, cmap='plasma')
    plt.colorbar(h, ax=axs[2])
    axs[2].set_xlabel(r'$x$ \ [mm]')
    axs[2].set_ylabel(r'$y$ \ [mm]')
    axs[2].set_title(r'$\varepsilon$')
    axs[2].axis('equal')

    plt.show()


def display_E(metamodel, inputs):
    '''Function to display the strain tensor components
    Inputs: - metamodel
    - inputs
    '''
    fig, ax = plt.subplots()
    plt.subplots_adjust(hspace=0.4)

    x, y        = torch.linspace(inputs.x_variable_min, inputs.x_variable_max, 100, requires_grad=True), torch.linspace(inputs.y_variable_min, inputs.y_variable_max, 100, requires_grad=True)
    [X, Y]      = torch.meshgrid(x,y)
    domain      = torch.hstack((X.reshape((X.numel(), 1)), Y.reshape((Y.numel(), 1))))
    E_est       = metamodel.model_E(domain).detach().numpy()


    h = ax.scatter(domain[:,0].detach().numpy(), domain[:,1].detach().numpy(), c=E_est, s=2**4, cmap = 'jet')
    plt.colorbar(h, ax=ax)
    ax.set_xlabel(r'$x$ \ [mm]')
    ax.set_ylabel(r'$y$ \ [mm]')
    ax.set_title(r'$E_{est}(x,y)$ \ [GPa]')
    ax.axis('equal')


    plt.show()


"""
def display_E(metamodel, inputs, display_mesh = True, alpha_mesh = 0.5):
    ''' Function to display a colormap of the Young's modulus
    Inputs: - metamodel
    - inputs
    - display_mesh: True to display the mesh of E on top of the colormap 
    - alpha_mesh: set the alpha value for displaying the mesh of E
    '''
    x, y        = torch.linspace(inputs.x_variable_min, inputs.x_variable_max, 100, requires_grad=True), torch.linspace(inputs.y_variable_min, inputs.y_variable_max, 100, requires_grad=True)
    [X, Y]      = torch.meshgrid(x,y)
    domain      = torch.hstack((X.reshape((X.numel(), 1)), Y.reshape((Y.numel(), 1))))
    coarse_mesh = create_mesh_E(inputs, n_E = [100, 100])
    E_est       = metamodel.model_E(domain)
    E_solution  =  None #E_function(torch.from_numpy(coarse_mesh.nodes), metamodel.E_solution, metamodel, inputs)

    if metamodel.Efunc is None:
        fig, ax = plt.subplots()
        h = ax.scatter(coarse_mesh.nodes[:,0], coarse_mesh.nodes[:,1],
                   c = E_est.detach().numpy()*metamodel.E_ref / 1000, s = 2**4, cmap = 'plasma')
        ax.colorbar(h, ax=ax)
        ax.set_xlabel(r'$x$ \ [mm]')
        ax.set_ylabel(r'$y$ \ [mm]')
        ax.set_title(r'$E_{est}(x,y)$ \ [GPa]')
        ax.axis('equal')
    
    else:
        bounds = np.linspace(np.min(E_solution.detach().numpy()/ 1000), np.max(E_solution.detach().numpy()/ 1000), 21)
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
        fig, axs = plt.subplots(1,3,figsize=(15,5))
        ax = axs[0]
        h = ax.scatter(coarse_mesh.nodes[:,0], coarse_mesh.nodes[:,1],
                   c = E_est.detach().numpy()*metamodel.E_ref / 1000, s = 2**4, cmap = 'jet', norm=norm)
        plt.colorbar(h, ax=ax)
        ax.set_xlabel(r'$x$ \ [mm]')
        ax.set_ylabel(r'$y$ \ [mm]')
        ax.set_title(r'$E_{est}(x,y)$ \ [GPa]')
        ax.axis('equal')

        ax = axs[1]
        h = ax.scatter(coarse_mesh.nodes[:,0], coarse_mesh.nodes[:,1],
                   c = E_solution / 1000, s = 2**4, cmap = 'jet', norm=norm)
        plt.colorbar(h, ax=ax)
        ax.set_xlabel(r'$x$ \ [mm]')
        ax.set_ylabel(r'$y$ \ [mm]')
        ax.set_title(r'$E_{solution}(x,y)$ \ [GPa]')
        ax.axis('equal')

        ax = axs[2]
        disp = np.abs(np.divide(E_est.detach().numpy()*metamodel.E_ref-E_solution.detach().numpy(), E_solution.detach().numpy()))
        disp = np.clip(disp, a_min = 0, a_max = 0.3)
        h = ax.scatter(coarse_mesh.nodes[:,0], coarse_mesh.nodes[:,1],
                   c = disp, s = 2**4, cmap = 'plasma')
        plt.colorbar(h, ax=ax)
        ax.set_xlabel(r'$x$ \ [mm]')
        ax.set_ylabel(r'$y$ \ [mm]')
        ax.set_title(r'Erreur')
        ax.axis('equal')

        if display_mesh:
            ax = axs[0]
            ax = metamodel.mesh_E.plot(ax, alpha = alpha_mesh)     

    plt.show()
"""
def display_training(metamodel, loss_to_display = {'res':1,'obs':1,'BC_sigma':1,'constitutive':1}):
    J_train = np.asarray(metamodel.list_J_train)[metamodel.list_iter_flag[1:]]

    fig, ax = plt.subplots()

    ax.set_title = 'Training'
    ax.set_xlabel('Iteration')
    ax.set_yscale('log')
    ax.grid()

    plt.plot(J_train[:,0], linewidth = 2., label = 'Total loss')

    for (key, value), index in zip(loss_to_display.items(), range(6)):
        if value == 1:
            ax.plot(J_train[:,index+1], linewidth = 2., label = key)
    ax.legend()
    plt.show()