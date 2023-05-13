import copy
import numpy as np
import qiskit
import itertools


def apply_fft(quantum_circuit, iterations=5):
    """Apply FFT to quantum circuit output

    Args:
        quantum_circuit (base.QuantumBase): 
        iterations (int, optional): defines how many expreminent iteration to perform. Defaults to 5.

    Returns:
        numpy.array: spectrum
    """
    spectrum = []
    X = np.linspace(-2 * np.pi, 2 * np.pi, 50, endpoint=False)
    for _ in range(iterations):
        weights = np.random.uniform(-np.pi, np.pi, len(quantum_circuit.params))
        Y = np.array([
            quantum_circuit.exp_val([x_1, x_2], weights) 
            for x_1 in X for x_2 in X])
        fourier = np.fft.rfftn(Y)
        spectrum.append(fourier)
    return np.array(spectrum)

def get_amplitudes(X, N, type='abs'):
    """Compute amplitude spectrum from a FFT transformed data

    Args:
        X (numpy.array): FFT spectrum
        N (int): length of the input to FFT
        type (str, optional): type of amplitudes to extract. Defaults to 'abs'.

    Returns:
        numpy.array: amplitude spectrum
    """
    types = {'abs': np.abs, 'imag': np.imag, 'real': np.real}
    amplitudes = 2. / N * types[type](X)[:,:N//2]
    return amplitudes.reshape(N//2, -1)

def get_phases(X, N):
    """Compute phase spectrum from FFT transformed data

    Args:
        X (numpy.array): FFT spectrum
        N (int): length of the input to FFT
        
    Returns:
        numpy.array: phase spectrum
    """
    threshold = np.max(np.abs(X)) * 5e-3
    X_denoised = copy.deepcopy(X)
    X_denoised[np.abs(X_denoised) < threshold] = 0j
    
    phases = np.angle(X_denoised / np.max(X_denoised))[:,:N//2]
    return phases.reshape(N//2, -1)

def extract_components(spectrum):
    prominent_amplitudes = np.max(spectrum, axis=1)
    difference = np.diff(prominent_amplitudes)
    difference[difference < 0] = 0
    return sum(np.histogram(difference)[0][1:])

def extract_phases_amount(phases):
    return sum(np.histogram(np.diff(np.unique(phases)))[0][1:]) + 1

def extract_variance(spectrum):
    variance = np.var(spectrum, axis=1)
    cut_off = np.histogram(spectrum)[1][1]
    return np.round(np.mean(variance[variance > cut_off]), 4)

def kl_divergence(p, q):
    divergence = []
    for i in range(len(p)):
        if p[i] < 1e-5:
            divergence.append(0)
            continue
        if q[i] < 1e-5:
            q[i] += 1e-5
        else:
            divergence.append(p[i] * np.log(p[i]/q[i]))
    return sum(divergence)

def get_haar(quantum_circuit, n_qubit):
    n_params = len(quantum_circuit.params)
    n_params_no_gradient = len(quantum_circuit.params_no_gradient)

    fidelities = []
    entanglement = []

    for _ in range(1000):
        state_1 = quantum_circuit.get_statevector(np.random.uniform(-1, 1, n_params_no_gradient), np.random.uniform(0, 2*np.pi, n_params))
        state_2 = quantum_circuit.get_statevector(np.random.uniform(-1, 1, n_params_no_gradient), np.random.uniform(0, 2*np.pi, n_params))
        
        partial_trace = 0
        qargs = itertools.combinations(list(range(n_qubit)),r=n_qubit-1)
        for i in qargs:
            rho = qiskit.quantum_info.partial_trace(state_1, i).data
            partial_trace += np.trace(np.dot(rho, rho))
        
        entanglement.append(np.round(2 * (1 - 1 / n_qubit * partial_trace), 4).real)
        fidelities.append(qiskit.quantum_info.state_fidelity(state_1, state_2))
    
    N = 2 ** 4
    haar_density = np.array([(N - 1) * (1 - i) ** (N - 2) for i in np.linspace(0, 1, 100)])
    fidelities_density = np.histogram(fidelities, bins=100, density=True)[0]
    return kl_divergence(fidelities_density / sum(fidelities_density), haar_density / sum(haar_density)), sum(entanglement) / 1000

