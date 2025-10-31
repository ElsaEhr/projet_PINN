import torch
import torch.nn as nn
import numpy as np
import package.display as display
import package.Mechanics_model as Mechanics_model
from package.Class_PINN import PINN

from copy import deepcopy


class MetaModel():
    """
    This is the MetaModel class, the PINNs representing the displacement and the stress will be defined as a instance of that class.
    This class allows the proper initialisation and training of the PINNs as well as identifying the unknown parameters 
    """

    def __init__(self, device, inputs, layers, E_0, E_ref, E_interpolation, n_E=10, activation=nn.Tanh(), optim="Adam",
                 Fourier_features=False, initial_freqs=torch.tensor([]),
                 seed=None, verbose=0, obs='rien',
                 N_FF=5,
                 sigma_FF_u=1, sigma_FF_sigma=1, optim_freq=0, iter_LFBGS_altern=None, Efunc=None):
        """
        Initialization of the PINN, with the number of layers and the first guess for the physical parameter. 
        """

        # Seed for initialization reproductibility
        if seed is not None:
            torch.manual_seed(seed)

        self.device = device  # Device specification
        self.input_save = inputs
        self.obs = obs
        
        # Initialize the PINNs for u and sigma
        layers_u = deepcopy(layers)
        layers_u.append(2)  # the output of u is u_x and u_y

        layers_sigma = deepcopy(layers)
        # the output of sigma is sigma_xx, sigma_yy, sigma_xy
        layers_sigma.append(3)

        self.model_u = PINN(device, inputs, layers_u, activation, optim,
                            Fourier_features,
                            seed, verbose, N_FF,
                            sigma_FF_u, optim_freq)

        self.model_sigma = PINN(device, inputs, layers_sigma, activation, optim,
                                Fourier_features,
                                seed, verbose, N_FF,
                                sigma_FF_sigma, optim_freq=optim_freq)

        self.E_ref = E_ref
        self.E = E_0.float().to(self.device)/self.E_ref     # Transférer E_0 au GPU
        self.E_0 = E_0.detach().clone().to(self.device)       # Transférer E_0 au GPU
        self.E.requires_grad = True

        # Verify the consistency of dimension
        assert (n_E[0]+1)*(n_E[1]+1) == self.E.shape[0]
        # Create the mesh for the interpolation for E
        self.mesh_E = Mechanics_model.create_mesh_E(inputs, n_E)
        self.n_E = n_E
        self.E_interpolation = E_interpolation
        self.Efunc = Efunc
        if not (Efunc is None):
            self.E_solution = Efunc(torch.from_numpy(self.mesh_E.nodes))

        self.iter_LFBGS_altern = iter_LFBGS_altern

        # Lists initialization
        self.list_J_train = []
        self.list_J_test = []

        self.list_y = []

        self.list_grad = []
        self.list_J_gradients = []

        self.list_params = []

        self.list_LBFGS_n_iter = []
        self.list_iter_flag = []

        self.list_theta_optim = []
        self.list_E_optim = []
        self.list_res_optim = []

        self.list_lr = []

        self.list_k = []
        self.list_grad_k = []
        self.list_E_matrix = []

        self.iter = 0
        self.iter_eval = 0
        self.alter_iter = 0

        self.list_subdomains = []

        self.end_training = False

        self.normalized_losses = {'res': torch.tensor(np.inf).to(self.device),
                                  'obs': torch.tensor(np.inf).to(self.device),
                                  'obs_F': torch.tensor(np.inf).to(self.device),
                                  'BC': torch.tensor(np.inf).to(self.device),
                                  'lines': torch.tensor(np.inf).to(self.device),
                                  'constitutive': torch.tensor(np.inf).to(self.device),
                                  'tikhonov': torch.tensor(1.).to(self.device)}

        self.optim = optim
        self.verbose = verbose  # 1 all, 0 nothing
        self.is_sigma_trained = False

        # You can change the values of the lambdas for pre-training here!
        self.lambdas = {'res': 1, 'obs': 1, 'obs_F': 1,
                        'BC': 1, 'lines': 1, 'constitutive': 1, 'tikhonov': 1}

        self.stagnation = []

        def L_operator():

            dx = (inputs.x_variable_max - inputs.x_variable_min) / (n_E[0])
            dy = (inputs.y_variable_max - inputs.y_variable_min) / (n_E[1])

            L = np.zeros(
                (self.mesh_E.nb_nodes, self.mesh_E.nb_nodes), dtype=np.float32)

            for index_elt in range(self.mesh_E.nb_elt):
                nodes = self.mesh_E.connectivity_table[index_elt, :]
                L[nodes[0], nodes[0]] += dx*dy**3/3 + dy*dx**3/3
                L[nodes[1], nodes[1]] += dx*dy**3/3 + dy*dx**3/3
                L[nodes[2], nodes[2]] += dx*dy**3/3 + dy*dx**3/3
                L[nodes[3], nodes[3]] += dx*dy**3/3 + dy*dx**3/3

                L[nodes[0], nodes[1]] += dx*dy**3/6 - dy*dx**3/3
                L[nodes[0], nodes[2]] += -dx*dy**3/6 - dy*dx**3/6
                L[nodes[0], nodes[3]] += -dx*dy**3/3 + dy*dx**3/6

                L[nodes[1], nodes[0]] += dx*dy**3/6 - dy*dx**3/3
                L[nodes[1], nodes[2]] += -dx*dy**3/3 + dy*dx**3/6
                L[nodes[1], nodes[3]] += -dx*dy**3/6 - dy*dx**3/6

                L[nodes[2], nodes[0]] += -dx*dy**3/6 - dy*dx**3/6
                L[nodes[2], nodes[1]] += -dx*dy**3/3 + dy*dx**3/6
                L[nodes[2], nodes[3]] += dx*dy**3/6 - dy*dx**3/3

                L[nodes[3], nodes[0]] += -dx*dy**3/3 + dy*dx**3/6
                L[nodes[3], nodes[1]] += -dx*dy**3/6 - dy*dx**3/6
                L[nodes[3], nodes[2]] += dx*dy**3/6 - dy*dx**3/3
            return torch.from_numpy(L)

        self.L = L_operator()

    def save_model(self, model, path):
        if model == 'sigma':
            torch.save(self.model_sigma, path)
        else:
            torch.save(self.model_u, path)

    def load_model(self, model, path):
        if model == 'sigma':
            self.model_sigma = torch.load(path)
        else:
            self.model_u = torch.load(path)

    def pretrain_sigma(self, inputs, pre_train_iter=100, lambdas={'res': 1, 'obs': 0, 'obs_F': 0, 'BC': 1, 'lines': 1, 'constitutive': 1, 'tikhonov': 0}):
        self.lambdas = lambdas

        # Normalization of losses if not done already
        if self.normalized_losses['res'] == np.inf:
            self.normalized_losses['res'] = Mechanics_model.J_res_sigma(
                self, inputs.train, is_sigma_trained=self.is_sigma_trained).detach().clone()
        if self.normalized_losses['BC'] == np.inf:
            self.normalized_losses['BC'] = Mechanics_model.J_BC(
                self, inputs, is_sigma_trained=self.is_sigma_trained).detach().clone()
        if self.normalized_losses['lines'] == np.inf:
            self.normalized_losses['lines'] = Mechanics_model.J_lines(
                self, inputs, is_sigma_trained=self.is_sigma_trained).detach().clone()
        if self.normalized_losses['constitutive'] == np.inf:
            self.normalized_losses['constitutive'] = Mechanics_model.J_constitutive(
                self, inputs.train, inputs, is_sigma_trained=self.is_sigma_trained).detach().clone()

        def J(metamodel, obs, domain, inputs):
            return (metamodel.lambdas['res'] * 1/metamodel.normalized_losses['res'] * Mechanics_model.J_res_sigma(metamodel, domain, is_sigma_trained=metamodel.is_sigma_trained),
                    torch.tensor(0),
                    torch.tensor(0),
                    metamodel.lambdas['BC'] * 1/metamodel.normalized_losses['BC'] * Mechanics_model.J_BC(
                        metamodel, inputs, is_sigma_trained=metamodel.is_sigma_trained),
                    metamodel.lambdas['lines'] * 1/metamodel.normalized_losses['lines'] *
                    Mechanics_model.J_lines(
                        self, inputs, is_sigma_trained=self.is_sigma_trained),
                    metamodel.lambdas['constitutive'] * 1/metamodel.normalized_losses['constitutive'] *
                    Mechanics_model.J_constitutive(
                        self, inputs.train, inputs, is_sigma_trained=self.is_sigma_trained),
                    torch.tensor(0.))

        optimizer = torch.optim.Adam(self.model_sigma.parameters())
        self.optim = 'Adam'

        for epoch in range(pre_train_iter):
            self.gradient_descent(J, optimizer, inputs, obs=self.obs)
            if self.verbose == 1:
                print("Epoch: ", epoch+1, "/", pre_train_iter,
                      " Loss: ", self.J_train.item())

        self.is_sigma_trained = True

    def pretrain_u(self, inputs, dic_model, pre_train_iter=100):
        ''' 
        Supervised learning for u
        '''
        self.lambdas = {'res': 0, 'obs': 1, 'obs_F': 0,
                        'BC': 0, 'lines': 0, 'constitutive': 0, 'tikhonov': 0}

        # Normalization of losses if not already done
        if self.normalized_losses['obs'] == np.inf:
            self.normalized_losses['obs'] = Mechanics_model.J_obs(
                self, dic_model).detach().clone()

        def J(metamodel, domain, inputs, dic_model):
            return (torch.tensor(0),
                    metamodel.lambdas['obs'] * 1/metamodel.normalized_losses['obs'] *
                    Mechanics_model.J_obs(metamodel, dic_model),
                    torch.tensor(0),
                    torch.tensor(0),
                    torch.tensor(0),
                    torch.tensor(0),
                    torch.tensor(0.))

        optimizer = torch.optim.Adam(self.model_u.parameters(), lr=1e-3)
        self.optim = 'Adam'

        for epoch in range(pre_train_iter):
            self.gradient_descent_u(J, optimizer, inputs, dic_model)
            if self.verbose == 1:
                print("Epoch: ", epoch+1, "/", pre_train_iter,
                      " Loss: ", self.J_train.item())

    def regularize_E(self, inputs, obs, pre_train_iter, lambdas={'res_u': 1, 'obs_F': 1}):
        ''' Regulazition of the estimation of E based on Physics laws '''
        optimizer = torch.optim.LBFGS([self.E],
                                      lr=1,
                                      max_iter=pre_train_iter,
                                      max_eval=10*pre_train_iter,
                                      line_search_fn="strong_wolfe",
                                      tolerance_grad=-1,
                                      tolerance_change=-1)
        self.optim = 'LBFGS'

        def closure():
            optimizer.zero_grad()

            J_train = lambdas['obs_F'] * Mechanics_model.J_obs_F(self, inputs)
            J_train.backward(retain_graph=True)

            # Simple constraint to keep the physical parameter box-constrained
            with torch.no_grad():
                self.E.clamp_(min=1e3/self.E_ref, max=5e4/self.E_ref)
                # print('E clamped')
            return J_train
        optimizer.step(closure)

    def train_model(self, inputs, dic_model, alter_steps=10,
                    alter_freq=(50, 50, 50),
                    lambdas_identif_E={'res': 0, 'obs': 0, 'obs_F': 1,
                                       'BC': 0, 'lines': 0, 'constitutive': 1, 'tikhonov': 1},
                    lambdas_update_sigma={
                        'res': 0.1, 'obs': 0, 'obs_F': 0, 'BC': 0.1, 'lines': 1, 'constitutive': 10},
                    lambdas_update_u={'res': 0, 'obs': 1, 'obs_F': 0, 'BC': 0, 'lines': 0, 'constitutive': 10}):
        """
        Method used for training the model.
        """

        # Normalization of losses if not done already
        if self.normalized_losses['res'] == np.inf:
            self.normalized_losses['res'] = Mechanics_model.J_res_sigma(
                self, inputs.train, is_sigma_trained=self.is_sigma_trained).detach().clone()
        if self.normalized_losses['BC'] == np.inf:
            self.normalized_losses['BC'] = Mechanics_model.J_BC(
                self, inputs, is_sigma_trained=self.is_sigma_trained).detach().clone()
        if self.normalized_losses['constitutive'] == np.inf:
            self.normalized_losses['constitutive'] = Mechanics_model.J_constitutive(
                self, inputs.train, inputs, is_sigma_trained=self.is_sigma_trained).detach().clone()
        if self.normalized_losses['obs'] == np.inf:
            self.normalized_losses['obs'] = Mechanics_model.J_obs(
                self, dic_model).detach().clone()
        if self.normalized_losses['obs_F'] == np.inf:
            self.normalized_losses['obs_F'] = Mechanics_model.J_obs_F(
                self, inputs).detach().clone()
        if self.normalized_losses['lines'] == np.inf:
            self.normalized_losses['lines'] = Mechanics_model.J_lines(
                self, inputs, is_sigma_trained=self.is_sigma_trained).detach().clone()
        if self.normalized_losses['tikhonov'] == np.inf:
            self.normalized_losses['tikhonov'] = Mechanics_model.Tikhonov(
                self).detach().clone()

        def J_update_u(metamodel, domain, inputs, dic_model):
            return (torch.tensor(0.),
                    metamodel.lambdas['obs'] * 1/metamodel.normalized_losses['obs'] *
                    Mechanics_model.J_obs(metamodel, dic_model),
                    metamodel.lambdas['obs_F'] * 1/metamodel.normalized_losses['obs_F'] *
                    Mechanics_model.J_obs_F(self, inputs),
                    torch.tensor(0.),
                    torch.tensor(0.),
                    metamodel.lambdas['constitutive'] * 1/metamodel.normalized_losses['constitutive'] *
                    Mechanics_model.J_constitutive(
                        metamodel, domain, inputs, is_sigma_trained=True),
                    torch.tensor(0.))

        def J_update_sigma(metamodel, domain, inputs):
            return (metamodel.lambdas['res'] * 1/metamodel.normalized_losses['res'] * Mechanics_model.J_res_sigma(metamodel, inputs.train, is_sigma_trained=True),
                    torch.tensor(0.),
                    torch.tensor(0.),
                    metamodel.lambdas['BC'] * 1/metamodel.normalized_losses['BC'] *
                    Mechanics_model.J_BC(
                        metamodel, inputs, is_sigma_trained=True),
                    metamodel.lambdas['lines'] * 1/metamodel.normalized_losses['lines'] *
                    Mechanics_model.J_lines(
                        self, inputs, is_sigma_trained=self.is_sigma_trained),
                    metamodel.lambdas['constitutive'] * 1/metamodel.normalized_losses['constitutive'] *
                    Mechanics_model.J_constitutive(
                        metamodel, domain, inputs, is_sigma_trained=True),
                    torch.tensor(0.))

        def J_identif_E(metamodel, domain, inputs):
            return (torch.tensor(0),
                    torch.tensor(0),
                    metamodel.lambdas['obs_F'] * 1/metamodel.normalized_losses['obs_F'] *
                    Mechanics_model.J_obs_F(self, inputs),
                    torch.tensor(0),
                    torch.tensor(0),
                    metamodel.lambdas['constitutive'] * 1/metamodel.normalized_losses['constitutive'] *
                    Mechanics_model.J_constitutive(
                        metamodel, domain, inputs, is_sigma_trained=True),
                    metamodel.lambdas['tikhonov'] * 1/metamodel.normalized_losses['tikhonov'] * Mechanics_model.Tikhonov(metamodel))

        # Alternating minimization
        self.best_rmse = float('inf')
        self.best_ite = 0
        self.iter_early_stopping = 0

        self.list_E_matrix.append(self.E.detach().clone())
        old_J = float("inf")

        for self.alter_step in range(alter_steps):
            print('ITERATION n°%d' % self.alter_step)
            # Minimizing on E
            self.lambdas = lambdas_identif_E

            optimizer = torch.optim.Adam([self.E])
            self.optim = 'Adam'

            for epoch in range(alter_freq[0]//2):
                self.gradient_descent(J_identif_E, optimizer, inputs)
                print("Epoch: ", epoch+1, "/",
                      alter_freq[0]//2, " Loss: ", self.J_train.item())

            optimizer = torch.optim.LBFGS([self.E],
                                          lr=1,
                                          max_iter=alter_freq[0]//2 - 1,
                                          max_eval=10*alter_freq[0]//2,
                                          line_search_fn="strong_wolfe",
                                          tolerance_grad=-1,
                                          tolerance_change=-1)
            self.optim = 'LBFGS'
            self.gradient_descent(J_identif_E, optimizer, inputs)

            # display.display_E(self, inputs)
            print("Saving E ")
            self.list_E_matrix.append(self.E.detach().clone())
            
            display.display_E(self, inputs, display_mesh=False)

            # Minimizing on sigma
            self.lambdas = lambdas_update_sigma
            iter_theta = 0
            optimizer = torch.optim.LBFGS(self.model_sigma.parameters(),
                                          # list(self.model_u.parameters()) + list(self.model_sigma.parameters())
                                          lr=1,
                                          max_iter=alter_freq[2],
                                          max_eval=10*alter_freq[2],
                                          line_search_fn="strong_wolfe",
                                          tolerance_grad=-1,
                                          tolerance_change=-1)
            self.optim = 'LBFGS'
            self.gradient_descent(J_update_sigma, optimizer, inputs)
            iter_theta += 1

            self.alter_iter += 1
            self.iter_early_stopping += 1

            # Minimizing on u
            self.lambdas = lambdas_update_u
            iter_theta = 0
            optimizer = torch.optim.Adam(self.model_u.parameters(), lr=1e-4)
            self.optim = 'Adam'
            self.gradient_descent_u(J_update_u, optimizer, inputs, dic_model)
            
            # optimizer = torch.optim.LBFGS(self.model_u.parameters(),
            #                               # list(self.model_u.parameters()) + list(self.model_sigma.parameters())
            #                               lr=1,
            #                               max_iter=alter_freq[3],
            #                               max_eval=10*alter_freq[3],
            #                               line_search_fn="strong_wolfe",
            #                               tolerance_grad=-1,
            #                               tolerance_change=-1)
            # self.optim = 'LBFGS'
            # self.gradient_descent_u(J_update_u, optimizer, inputs, dic_model)
            iter_theta += 1

            self.alter_iter += 1
            self.iter_early_stopping += 1

            # Early stopping
            if self.iter_early_stopping == 100:
                print("Early stopping at altern step", self.alter_step + 1)
                break

            # display.display_sigma(self, inputs, 50.)
            # display.display_epsilon(self, inputs)
            # display.display_u(self, inputs, obs, display_error=True)

        self.end_training = True

        if self.optim == "LBFGS":
            self.list_iter_flag.append(True)

    def gradient_descent_u(self, J, optimizer, inputs, dic_model):
        """
        Gradient descent method used during the training for updating parameters. 
        """

        def closure():
            optimizer.zero_grad()

            self.J_train = J(self, inputs.train, inputs, dic_model)
            self.J_res_train = torch.tensor([0])
            self.J_obs_train = self.J_train[1]
            self.J_obs_F_train = self.J_train[2]
            self.J_BC_train = torch.tensor([0])
            self.J_lines_train = torch.tensor([0])
            self.J_constitutive_train = self.J_train[5]
            self.J_tikhonov_train = torch.tensor([0])
            self.J_train = sum(self.J_train)
            self.J_train.backward(retain_graph=True)

            # Clipping the gradient to avoid diverging during the training
            nn.utils.clip_grad_norm_(
                self.model_u.parameters(), max_norm=1e3, norm_type=2.0)
            nn.utils.clip_grad_norm_(
                self.model_sigma.parameters(), max_norm=1e3, norm_type=2.0)

            # Simple constraint to keep the physical parameter box-constrained
            with torch.no_grad():
                self.E.clamp_(min=1e3/self.E_ref, max=5e4/self.E_ref)
                # print('E clamped')

            self.iter_eval += 1

            with torch.no_grad():
                self.update_lists(inputs, optimizer)
                self.iter += 1
            return self.J_train

        optimizer.step(closure)
        
    def gradient_descent(self, J, optimizer, inputs):
        """
        Gradient descent method used during the training for updating parameters. 
        """

        def closure():
            optimizer.zero_grad()

            self.J_train = J(self, inputs.train, inputs)
            self.J_res_train = self.J_train[0]
            self.J_obs_train = torch.tensor([0])
            self.J_obs_F_train = self.J_train[2]
            self.J_BC_train = self.J_train[3]
            self.J_lines_train = self.J_train[4]
            self.J_constitutive_train = self.J_train[5]
            self.J_tikhonov_train = self.J_train[6]
            self.J_train = sum(self.J_train)
            self.J_train.backward(retain_graph=True)

            # Clipping the gradient to avoid diverging during the training
            nn.utils.clip_grad_norm_(
                self.model_u.parameters(), max_norm=1e3, norm_type=2.0)
            nn.utils.clip_grad_norm_(
                self.model_sigma.parameters(), max_norm=1e3, norm_type=2.0)

            # Simple constraint to keep the physical parameter box-constrained
            with torch.no_grad():
                self.E.clamp_(min=1e3/self.E_ref, max=5e4/self.E_ref)
                # print('E clamped')

            self.iter_eval += 1

            with torch.no_grad():
                self.update_lists(inputs, optimizer)
                self.iter += 1
            return self.J_train

        optimizer.step(closure)

    def update_lists(self, inputs, optimizer):
        """
        Method used for updating the model lists, to keep track of the values of interest during the training.
        """

        self.list_J_train.append([self.J_train.item(), self.J_res_train.item(),
                                  self.J_obs_train.item(), self.J_obs_F_train.item(),
                                  self.J_BC_train.item(), self.J_lines_train.item(),
                                  self.J_constitutive_train.item(), self.J_tikhonov_train.item()])

        self.J_train = self.J_train.detach().clone()
        self.J_res_train = self.J_res_train.detach().clone()
        self.J_obs_train = self.J_obs_train.detach().clone()
        self.J_obs_F_train = self.J_obs_F_train.detach().clone()
        self.J_BC_train = self.J_BC_train.detach().clone()
        self.J_lines_train = self.J_lines_train.detach().clone()
        self.J_constitutive_train = self.J_constitutive_train.detach().clone()
        self.J_tikhonov_train = self.J_tikhonov_train.detach().clone()

        if self.optim == "LBFGS":
            self.list_LBFGS_n_iter.append(
                optimizer.state_dict()['state'][0]['n_iter'])

            if self.end_training:
                self.list_iter_flag.pop()
                self.end_training = False

            if (len(self.list_LBFGS_n_iter) > 1):
                if (self.list_LBFGS_n_iter[-1] == self.list_LBFGS_n_iter[-2]):
                    self.list_iter_flag.append(False)
                else:
                    self.list_iter_flag.append(True)
                    self.iter += 1
            else:
                self.iter += 1

            if (self.lambdas['res'] == 0):
                self.list_res_optim.append(False)
            else:
                self.list_res_optim.append(True)

        elif self.optim == 'Adam':
            self.list_LBFGS_n_iter.append([1])
            self.list_iter_flag.append(True)
            self.iter += 1
            if self.end_training:
                print('!!!!! ATTENTION !!!!!!')
                self.list_iter_flag.pop()
                self.end_training = False

        else:
            self.list_LBFGS_n_iter.append(
                optimizer.state_dict()['state'][0]['n_iter'])

            if self.end_training:
                print('!!!!! ATTENTION !!!!!!')
                self.list_iter_flag.pop()
                self.end_training = False

            if (len(self.list_LBFGS_n_iter) > 1):
                if (self.list_LBFGS_n_iter[-1] == self.list_LBFGS_n_iter[-2]):
                    self.list_iter_flag.append(False)
                else:
                    self.list_iter_flag.append(True)
                    self.iter += 1
            else:
                self.iter += 1

            if (self.lambdas['res'] == 0):
                self.list_res_optim.append(False)
            else:
                self.list_res_optim.append(True)
