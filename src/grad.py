import random
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function

import qiskit
#from qiskit.aqua.operators import X, Y, Z, I, StateFn, CircuitStateFn
from qiskit.opflow import X, Y, Z, I, StateFn, CircuitStateFn
#from qiskit.aqua.operators.gradients import Gradient
from qiskit.opflow import Gradient


class ParameterShiftGradFunction(Function):
    """Hybrid quantum - classical function definition."""
    
    @staticmethod
    def forward(ctx, thetas, input, quantum_circuit):
        """ Forward pass computation """
        ctx.quantum_circuit = quantum_circuit
        ctx.save_for_backward(input, thetas)
        input_list = input.tolist()
        expectations_z = ctx.quantum_circuit.exp_val([*input_list], thetas.tolist())
        return input.new(expectations_z).reshape(1,-1)
        
    @staticmethod
    def backward(ctx, grad_output):
        input, thetas = ctx.saved_tensors

        if len(thetas) == 0:
            return grad_output, grad_output, None

        input_list = input.tolist()
        input = input_list
        thetas = thetas.tolist()

        n_params = len(thetas)
        
        params_extended = np.array([thetas] * n_params).reshape((n_params, n_params))
        
        shift_right = params_extended + np.eye(n_params) * np.pi / 2
        shift_left  = params_extended - np.eye(n_params) * np.pi / 2
        
        gradients = []
        for i in range(len(shift_right)):
            expectation_right = ctx.quantum_circuit.exp_val(input, shift_right[i])
            expectation_left  = ctx.quantum_circuit.exp_val(input, shift_left[i])
            
            gradient = .5 * (torch.tensor(np.array([expectation_right])) - torch.tensor(np.array([expectation_left])))
            gradients.append(gradient)

        gradients_thetas = grad_output @ torch.cat(gradients, dim=0).float().T

        #scale_factor_input = int(len(input) / n_qubits)
        #gradients_inputs = torch.cat([grad_output] * 2, dim=1)[:,:-1]
        return gradients_thetas, grad_output, None


class AdversarialGradFunction(Function):
    """Hybrid quantum - classical function definition."""
    
    @staticmethod
    def forward(ctx, thetas, input, quantum_circuit, qvae, classifier, x, shots=1):
        """ Forward pass computation """
        ctx.quantum_circuit = quantum_circuit
        ctx.qvae = qvae
        ctx.classifier = classifier
        ctx.save_for_backward(x, thetas)
        
        sample = ctx.quantum_circuit.sample(input.tolist(), thetas.tolist(), shots)
        return input.new(sample)

    @staticmethod
    def backward(ctx, grad_output):
        x, thetas = ctx.saved_tensors

        if len(thetas) == 0:
            return grad_output, grad_output, None

        thetas = thetas.tolist()

        n_params = len(thetas)
        cross_entopy = nn.BCELoss()

        params_extended = np.array([thetas] * n_params).reshape((n_params, n_params))
        
        shift_right = params_extended + np.eye(n_params) * np.pi / 2
        shift_left  = params_extended - np.eye(n_params) * np.pi / 2
        
        gradients = []
            
        for i in range(len(shift_right)):
            #  --- Sample right ---
            ctx.qvae.hybrid.weight = nn.Parameter(torch.tensor(shift_right[i]))
            loss_right = []
            for _ in range(4):
                z = ctx.qvae.encoder(x)
                x_hat = ctx.qvae.decoder(z).float()
                loss_right.append(torch.tensor([1e-4 * torch.mean(torch.logit(1 - ctx.classifier(x_hat))) - cross_entopy(x_hat, x)]))
            
            #  --- Sample left ---
            ctx.qvae.hybrid.weight = nn.Parameter(torch.tensor(shift_left[i]))
            loss_left = []
            for _ in range(4):
                z = ctx.qvae.encoder(x)
                x_hat = ctx.qvae.decoder(z).float()
                loss_left.append(torch.tensor([1e-4 * torch.mean(torch.logit(1 - ctx.classifier(x_hat))) - cross_entopy(x_hat, x)]))
            
            gradients.append(torch.tensor([.5 * (torch.mean(torch.cat(loss_right)) - torch.mean(torch.cat(loss_left)))]))
        
        # Revert the weights
        ctx.qvae.hybrid.weight = nn.Parameter(torch.tensor(thetas))
        gradients = torch.cat([grad_output] * 1, axis=1) * torch.cat(gradients).float()
        #print(grad_output.reshape(-1, 4))
        return gradients, grad_output.reshape(-1, 4), None, None, None, None, None
