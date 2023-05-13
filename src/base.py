import random
import copy
import numpy as np

import qiskit
#from qiskit.aqua.operators import X, Y, Z, I
from qiskit.opflow import X, Y, Z, I
from qiskit.compiler import transpile
from qiskit.circuit import Parameter



class QuantumBase:
    """Qiskit wrapper class.
    """
    
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        
        self.params = []
        self.params_no_gradient = []

        self.measurements = []

        self.same_init = False
        self.circuits = []

        self.layer_index = 0
        self.embedding_layer_index = 0

        self.rotation_gates = {
            'X': self.circuit.rx,
            'Y': self.circuit.ry, 
            'Z': self.circuit.rz}
        
        self.crotation_gates = {
            'X': self.circuit.crx,
            'Y': self.circuit.cry, 
            'Z': self.circuit.crz}
        
        self.pauli_operators = {
            'X': X,
            'Y': Y,
            'Z': Z}

        self.statevectors = [[0]*2**4]
        self.inputs = [[0]*4]
        self.outputs = [[0]*4]

    def _add_base_embedding_layer(self):
        self.embedding_layer_index += 1

        layer_params = [
            qiskit.circuit.Parameter('x_{}{}'.format(self.embedding_layer_index, i)) 
            for i in range(self.n_qubits)]
        self.params_no_gradient += layer_params

        return layer_params
    
    def add_same_embedding_layer(self):
        self.same_init = True
        self.circuits.append(self.circuit.copy())
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
    
    def add_cangle_embedding_layer(self, axis='X'):
        """Adds data embedding layer with angle encoding."""
        layer_params = self._add_base_embedding_layer()
        
        for i in range(self.n_qubits):
            self.crotation_gates[axis](layer_params[i], i, i - 1)
    
    def add_angle_embedding_layer(self, axis='X'):
        """Adds data embedding layer with angle encoding."""
        layer_params = self._add_base_embedding_layer()
        
        for i in range(self.n_qubits):
            self.rotation_gates[axis](layer_params[i], i)

    def add_kitchen_sink_embedding_layer(self, axis='X'):
        layer_params = self._add_base_embedding_layer()

        all_qubits = np.arange(self.n_qubits)
        self.circuit.h(all_qubits)
        self.circuit.swap(all_qubits[1:], all_qubits[:-1])
        for i in range(self.n_qubits):
            self.rotation_gates[axis](layer_params[i], i)

    def add_IQP_embedding_layer(self):
        """Adds data embedding layer with angle encoding."""
        layer_params = self._add_base_embedding_layer()

        self.circuit.h(range(self.n_qubits))
        for i in range(self.n_qubits):
            self.circuit.rz(layer_params[i], i)
        
    def add_double_IQP_embedding_layer(self):
        """Adds data embedding from https://arxiv.org/pdf/2011.00027.pdf
        """
        self.params_no_gradient = [qiskit.circuit.Parameter('x_{}'.format(0))]
        for i in range(1, self.n_qubits):
            self.params_no_gradient.append(
                qiskit.circuit.Parameter('x_{}'.format(i)))
            self.params_no_gradient.append(
                qiskit.circuit.Parameter('x_{}x_{}'.format(i-1, i)))
        
        self.circuit.h(range(self.n_qubits))
        self.circuit.rz(self.params_no_gradient[0], 0)
        for i in range(1, self.n_qubits):
            self.circuit.rz(self.params_no_gradient[2 * i - 1], i)
            self.circuit.cx(i - 1, i)
            self.circuit.rz(self.params_no_gradient[2 * i], i)
            self.circuit.cx(i - 1, i)
    
    def add_rotation_random_layer(self):
        """Adds a layers with randomly chosen rotation axis.
        
        Args:
            layer_index: layer index that will be appended to the parameter name
        """
        self.layer_index += 1

        layer_params = [
            qiskit.circuit.Parameter('theta_{}{}'.format(self.layer_index, i)) 
            for i in range(self.n_qubits)]
        self.params += layer_params
        
        for i in range(self.n_qubits):
            random.choice([self.circuit.rx, 
                           self.circuit.ry, 
                           self.circuit.rz])(layer_params[i], i)
    
    def add_rotation_layer(self, axis='X'):
        """Adds a layer with rotation gates around X-axis.
        
        Args:
            layer_index: layer index that will be appended to the parameter name
        """
        self.rotation_gates = {
            'X': self.circuit.rx,
            'Y': self.circuit.ry, 
            'Z': self.circuit.rz}
        
        self.layer_index += 1
        
        layer_params = [
            qiskit.circuit.Parameter('theta_{}{}'.format(self.layer_index, i)) 
            for i in range(self.n_qubits)]
        self.params += layer_params

        for i in range(self.n_qubits):
            self.rotation_gates[axis](layer_params[i], i)
        
    def add_crotation_layer(self, axis='X', order=None):
        self.layer_index += 1
        
        layer_params = [
            qiskit.circuit.Parameter('theta_{}{}'.format(self.layer_index, i)) 
            for i in range(self.n_qubits)]
        self.params += layer_params

        if order == 'reverse':
            self.crotation_gates[axis](layer_params[0], 0, self.n_qubits - 1)
            for i in range(1, self.n_qubits):
                self.crotation_gates[axis](layer_params[i], i, i - 1)
        else:
            self.crotation_gates[axis](layer_params[0], self.n_qubits - 1, 0)
            for i in range(1, self.n_qubits):
                self.crotation_gates[axis](layer_params[i], i - 1, i)
    
    def add_cnot_layer(self, order=None):
        if order == 'reverse':
            self.circuit.cx(0, self.n_qubits - 1)
            for i in range(1, self.n_qubits):
                self.circuit.cx(i, i - 1)
        else:
            self.circuit.cx(self.n_qubits - 1, 0)
            for i in range(1, self.n_qubits):
                self.circuit.cx(i - 1, i)

    def add_single_measurements(self, pauli='Z'):
        operators = [self.pauli_operators[pauli] for i in range(self.n_qubits)]
        self.measurements += [(I ^ self.n_qubits - 1 - i) ^ operators[i] ^ (I ^ i) for i in range(self.n_qubits)]
        
    def bind_parameters(self, inputs, thetas):
        """Bind parameter values to parameters of the circuit

        Args:
            inputs (list): paramerunter values for embedding 
            thetas (list): parameter values for 
        """
        if self.same_init:
            return {self.params[i]: theta for i, theta in enumerate(thetas)}
        input_binds = {self.params_no_gradient[i]: input for i, input in enumerate(inputs)}
        layer_binds = {self.params[i]: theta for i, theta in enumerate(thetas)}

        return {**input_binds, **layer_binds}
    
    def _same(circuit, inputs):
        X = np.sqrt((np.array(inputs) + 1) / 2)
        Y = np.sqrt(1 - X**2)
        for i in range(len(inputs)):
            circuit.initialize([X[i], Y[i]], i)
        return circuit

    def get_statevector(self, inputs, thetas):
        if self.same_init:
            self.circuits.append(self.circuit.copy())
            self.circuit = qiskit.QuantumCircuit(self.n_qubits)
            for i in range(len(self.circuits) - 1):
                self.circuit += QuantumBase._same(self.circuits[i], [0] * 4)
            self.circuit += self.circuits[-1]
        
#         job = qiskit.execute(
#             self.circuit,
#             qiskit.Aer.get_backend('statevector_simulator'),
#             parameter_binds = [self.bind_parameters(inputs, thetas)])
        circuits = self.circuit.bind_parameters(self.bind_parameters(inputs, thetas))
        backend_sv=qiskit.Aer.get_backend('statevector_simulator')
        job = backend_sv.run(transpile(circuits, backend_sv))
        statevector = job.result().get_statevector()
        return statevector

    def exp_val(self, inputs, thetas):
        """exp_val the circuit

        Args:
            inputs (list): list of input values
            thetas (list): list of parameter values

        Returns:
            list: Pauli Z expectiation for all qubits
        """
        statevector = self.get_statevector(inputs, thetas)
        self.statevectors.append(statevector)
        self.inputs.append(inputs)
        
        expectations = []
        for measurement in self.measurements:
            #print(measurement.to_matrix())
            #print(statevector)
            expectations.append(statevector.expectation_value(measurement))
            #expectations.append(np.conj(statevector).T @ measurement.to_matrix() @ statevector)

        self.outputs.append(np.real(expectations))
        #print(np.real(expectations))
        return np.real(expectations)
    
    def sample(self, inputs, thetas, shots=1):
        circuit = self.circuit.copy()
        circuit.measure_all()

#         job = qiskit.execute(
#             circuit,
#             qiskit.Aer.get_backend('qasm_simulator'),
#             shots=shots,
#             parameter_binds = [self.bind_parameters(inputs, thetas)])
        
        circuits = circuit.bind_parameters(self.bind_parameters(inputs, thetas))
        backend_qasm=qiskit.Aer.get_backend('qasm_simulator')
        job = backend_qasm.run(transpile(circuits, backend_qasm))
        result = job.result().get_counts()
        
        sampled_z = []
        for state, count in result.items():
            for _ in range(count):
                sampled_z.append(state)
        sampled_z = np.array(list(list(sample) for sample in sampled_z)).astype('float')

        circuit = self.circuit.copy()
        circuit.h(np.arange(self.n_qubits))
        circuit.measure_all()
        
        circuits = circuit.bind_parameters(self.bind_parameters(inputs, thetas))
        
        job = backend_qasm.run(transpile(circuits, backend_qasm))
        result = job.result().get_counts()

#         job = qiskit.execute(
#             circuit,
#             qiskit.Aer.get_backend('qasm_simulator'),
#             shots=shots,
#             parameter_binds = [self.bind_parameters(inputs, thetas)])

#        result = job.result().get_counts(circuit)
        #print('..')
        sampled_x = []
        for state, count in result.items():
            for _ in range(count):
                sampled_x.append(state)
        sampled_x = np.array(list(list(sample) for sample in sampled_x)).astype('float')
        return np.append(sampled_z, sampled_x, axis=1)