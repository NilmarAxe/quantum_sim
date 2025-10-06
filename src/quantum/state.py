import numpy as np
from typing import Tuple, Dict
from qiskit.quantum_info import state_fidelity, entropy, partial_trace, DensityMatrix


class QuantumStateAnalyzer:
    """Analyzes quantum states and provides various metrics."""
    
    @staticmethod
    def fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
        """
        Calculates fidelity between two quantum states.
        
        Args:
            state1: First quantum state
            state2: Second quantum state
            
        Returns:
            Fidelity value between 0 and 1
        """
        state1 = state1 / np.linalg.norm(state1)
        state2 = state2 / np.linalg.norm(state2)
        return np.abs(np.dot(np.conj(state1), state2)) ** 2
    
    @staticmethod
    def trace_distance(state1: np.ndarray, state2: np.ndarray) -> float:
        """
        Calculates trace distance between two states.
        
        Args:
            state1: First quantum state
            state2: Second quantum state
            
        Returns:
            Trace distance
        """
        rho1 = np.outer(state1, np.conj(state1))
        rho2 = np.outer(state2, np.conj(state2))
        diff = rho1 - rho2
        eigenvalues = np.linalg.eigvalsh(diff)
        return 0.5 * np.sum(np.abs(eigenvalues))
    
    @staticmethod
    def purity(state: np.ndarray) -> float:
        """
        Calculates purity of a quantum state.
        
        Args:
            state: Quantum state vector
            
        Returns:
            Purity value
        """
        rho = np.outer(state, np.conj(state))
        return np.real(np.trace(rho @ rho))
    
    @staticmethod
    def von_neumann_entropy(state: np.ndarray) -> float:
        """
        Calculates von Neumann entropy.
        
        Args:
            state: Quantum state vector
            
        Returns:
            Entropy value
        """
        rho = DensityMatrix(state)
        return entropy(rho)
    
    @staticmethod
    def bloch_coordinates(state: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculates Bloch sphere coordinates for a single qubit state.
        
        Args:
            state: Single qubit state vector
            
        Returns:
            Tuple (x, y, z) of Bloch coordinates
        """
        state = state / np.linalg.norm(state)
        
        # Density matrix
        rho = np.outer(state, np.conj(state))
        
        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_y = np.array([[0, -1j], [1j, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])
        
        x = np.real(np.trace(rho @ sigma_x))
        y = np.real(np.trace(rho @ sigma_y))
        z = np.real(np.trace(rho @ sigma_z))
        
        return x, y, z
    
    @staticmethod
    def probability_distribution(state: np.ndarray) -> Dict[str, float]:
        """
        Calculates probability distribution for computational basis.
        
        Args:
            state: Quantum state vector
            
        Returns:
            Dictionary mapping basis states to probabilities
        """
        state = state / np.linalg.norm(state)
        n_qubits = int(np.log2(len(state)))
        
        probs = {}
        for i, amplitude in enumerate(state):
            basis_state = format(i, f'0{n_qubits}b')
            prob = np.abs(amplitude) ** 2
            if prob > 1e-10:  # Filter negligible probabilities
                probs[basis_state] = prob
                
        return probs
    
    @staticmethod
    def expectation_value(state: np.ndarray, observable: np.ndarray) -> complex:
        """
        Calculates expectation value of an observable.
        
        Args:
            state: Quantum state vector
            observable: Observable operator matrix
            
        Returns:
            Expectation value
        """
        state = state / np.linalg.norm(state)
        return np.dot(np.conj(state), observable @ state)
    
    @staticmethod
    def overlap(state1: np.ndarray, state2: np.ndarray) -> complex:
        """
        Calculates overlap (inner product) between states.
        
        Args:
            state1: First quantum state
            state2: Second quantum state
            
        Returns:
            Complex overlap value
        """
        state1 = state1 / np.linalg.norm(state1)
        state2 = state2 / np.linalg.norm(state2)
        return np.dot(np.conj(state1), state2)
    
    @staticmethod
    def concurrence(state: np.ndarray) -> float:
        """
        Calculates concurrence for a two-qubit state (entanglement measure).
        
        Args:
            state: Two-qubit state vector
            
        Returns:
            Concurrence value
        """
        if len(state) != 4:
            raise ValueError("Concurrence is defined for two-qubit states only")
        
        state = state / np.linalg.norm(state)
        rho = np.outer(state, np.conj(state))
        
        # Spin-flip operator
        sigma_y = np.array([[0, -1j], [1j, 0]])
        spin_flip = np.kron(sigma_y, sigma_y)
        
        # Compute R matrix
        rho_tilde = spin_flip @ np.conj(rho) @ spin_flip
        R = rho @ rho_tilde
        
        eigenvalues = np.linalg.eigvalsh(R)
        eigenvalues = np.sqrt(np.maximum(eigenvalues, 0))
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        concurrence = max(0, eigenvalues[0] - eigenvalues[1] - 
                         eigenvalues[2] - eigenvalues[3])
        
        return concurrence
    
    @staticmethod
    def schmidt_decomposition(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs Schmidt decomposition for a bipartite state.
        
        Args:
            state: Bipartite quantum state
            
        Returns:
            Tuple of (schmidt_coefficients, left_basis, right_basis)
        """
        n = len(state)
        dim = int(np.sqrt(n))
        
        if dim * dim != n:
            raise ValueError("State dimension must be a perfect square")
        
        # Reshape into matrix
        psi_matrix = state.reshape(dim, dim)
        
        # SVD
        U, s, Vh = np.linalg.svd(psi_matrix)
        
        return s, U, Vh.T.conj()
    
    @staticmethod
    def state_tomography_data(state: np.ndarray, n_measurements: int = 1000) -> Dict[str, np.ndarray]:
        """
        Simulates state tomography measurements.
        
        Args:
            state: Quantum state to measure
            n_measurements: Number of measurements per basis
            
        Returns:
            Dictionary of measurement results
        """
        state = state / np.linalg.norm(state)
        n_qubits = int(np.log2(len(state)))
        
        if n_qubits != 1:
            raise NotImplementedError("Currently only supports single qubit tomography")
        
        # Measurement bases
        bases = {
            'Z': np.eye(2),
            'X': np.array([[1, 1], [1, -1]]) / np.sqrt(2),
            'Y': np.array([[1, -1j], [1, 1j]]) / np.sqrt(2)
        }
        
        results = {}
        for basis_name, basis_matrix in bases.items():
            # Transform to measurement basis
            transformed_state = basis_matrix @ state
            probs = np.abs(transformed_state) ** 2
            
            # Simulate measurements
            measurements = np.random.choice([0, 1], size=n_measurements, p=probs)
            results[basis_name] = measurements
            
        return results