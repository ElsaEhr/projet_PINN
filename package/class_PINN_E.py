import torch
import torch.nn as nn


class PINN(nn.Module):
    """
    This is the PINN class, the Neural Network will be defined as a instance of that class. 
    """

    def __init__(self, device, inputs, layers, activation=nn.Tanh(), optim="Adam",
                 Fourier_features=False,
                 seed=None, verbose=0, N_FF=5,
                 sigma_FF=1, optim_freq=0):
        """
        Initialization of the PINN, with the number of layers and the first guess for the physical parameter. 
        """

        super(PINN, self).__init__()

        # Seed for initialization reproductibility
        if seed is not None:
            torch.manual_seed(seed)

        self.device = device  # Device specification
        self.input_save = inputs

        self.variable_min = torch.tensor(
            [inputs.x_variable_min, inputs.y_variable_min])
        self.variable_max = torch.tensor(
            [inputs.x_variable_max, inputs.y_variable_max])

        self.hidden = nn.ModuleList().to(self.device)
        self.layers = layers
        self.activation = activation

        self.Fourier_features = Fourier_features
        if self.Fourier_features:
            self.N_FF = N_FF
            self.sigma_FF = sigma_FF
            self.freqs = torch.randn(
                self.N_FF * self.layers[0], self.layers[0]).to(self.device) * self.sigma_FF + optim_freq

        # Input Layer
        if self.Fourier_features:
            input_layer = nn.Linear(
                self.layers[0] * (2 * self.N_FF + 1), self.layers[1], bias=True)
        else:
            input_layer = nn.Linear(self.layers[0], self.layers[1], bias=True)
        nn.init.xavier_normal_(input_layer.weight.data, gain=1.0)
        nn.init.zeros_(input_layer.bias.data)
        self.hidden.append(input_layer)

        # Hidden layers
        for i, (input_size, output_size) in enumerate(zip(self.layers[1:-1], self.layers[2:-1])):
            linear = nn.Linear(input_size, output_size, bias=True)
            nn.init.xavier_normal_(linear.weight.data, gain=1.0)
            nn.init.zeros_(linear.bias.data)
            self.hidden.append(linear)

        # Output layer
        output_layer = nn.Linear(self.layers[-2], self.layers[-1], bias=False)
        nn.init.xavier_normal_(output_layer.weight.data, gain=1.0)
        self.hidden.append(output_layer)

    def forward(self, input_tensor):

        # Normalization layer
        input_tensor = input_tensor.to(self.device)
        input_tensor = torch.div(
            input_tensor - self.variable_min, self.variable_max - self.variable_min)

        # Fourier features
        if self.Fourier_features:
            FF = torch.zeros(
                input_tensor.shape[0], input_tensor.shape[1] * (2 * self.N_FF + 1)).to(self.device)
            FF[:, 0:input_tensor.shape[1]] = input_tensor
            FF[:, input_tensor.shape[1]:input_tensor.shape[1] * (self.N_FF + 1)] = torch.sin(
                2 * torch.pi * torch.matmul(self.freqs, input_tensor.T).T)
            FF[:, input_tensor.shape[1] * (self.N_FF + 1):input_tensor.shape[1] * (
                2 * self.N_FF + 1)] = torch.cos(2 * torch.pi * torch.matmul(self.freqs, input_tensor.T).T)
            input_tensor = FF
            input_tensor.to(self.device)

        # Forward
        for (l, linear_transform) in zip(range(len(self.hidden)), self.hidden):
            # For input and hidden layers, apply activation function after linear transformation
            if l < len(self.hidden) - 1:
                input_tensor = self.activation(linear_transform(input_tensor))

            # For output layer, apply only linear transformation
            else:
                output = linear_transform(input_tensor)

        return output
