import numpy as np
from typing import List, Tuple, Optional
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter


class QuantumGates:
    """Collection of quantum gates and gate operations."""
    
    @staticmethod
    def hadamard() -> np.ndarray:
        """Returns Hadamard gate matrix."""
        return np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    
    @staticmethod
    def pauli_x() -> np.ndarray:
        """Returns Pauli-X (NOT) gate matrix."""
        return np.array([[0, 1], [1, 0]])
    
    @staticmethod
    def pauli_y() -> np.ndarray:
        """Returns Pauli-Y gate matrix."""
        return np.array([[0, -1j], [1j, 0]])
    
    @staticmethod
    def pauli_z() -> np.ndarray:
        """Returns Pauli-Z gate matrix."""
        return np.array([[1, 0], [0, -1]])
    
    @staticmethod
    def rotation_x(theta: float) -> np.ndarray:
        """Returns RX rotation gate matrix."""
        return np.array([
            [np.cos(theta/2), -1j*np.sin(theta/2)],
            [-1j*np.sin(theta/2), np.cos(theta/2)]
        ])
    
    @staticmethod
    def rotation_y(theta: float) -> np.ndarray:
        """Returns RY rotation gate matrix."""
        return np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ])
    
    @staticmethod
    def rotation_z(phi: float) -> np.ndarray:
        """Returns RZ rotation gate matrix."""
        return np.array([
            [np.exp(-1j*phi/2), 0],
            [0, np.exp(1j*phi/2)]
        ])
    
    @staticmethod
    def phase_gate(phi: float) -> np.ndarray:
        """Returns phase gate matrix."""
        return np.array([[1, 0], [0, np.exp(1j*phi)]])
    
    @staticmethod
    def u3_gate(theta: float, phi: float, lambda_: float) -> np.ndarray:
        """Returns U3 universal single-qubit gate matrix."""
        return np.array([
            [np.cos(theta/2), -np.exp(1j*lambda_)*np.sin(theta/2)],
            [np.exp(1j*phi)*np.sin(theta/2), 
             np.exp(1j*(phi+lambda_))*np.cos(theta/2)]
        ])
    
    @staticmethod
    def cnot() -> np.ndarray:
        """Returns CNOT gate matrix."""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
    
    @staticmethod
    def apply_gate(state: np.ndarray, gate: np.ndarray) -> np.ndarray:
        """Applies a gate to a quantum state."""
        return gate @ state
    
    @staticmethod
    def create_parameterized_circuit(n_qubits: int, 
                                     n_layers: int) -> Tuple[QuantumCircuit, List[Parameter]]:
        """
        Creates a parameterized quantum circuit for optimization.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of variational layers
            
        Returns:
            Tuple of (circuit, parameters)
        """
        qr = QuantumRegister(n_qubits, 'q')
        qc = QuantumCircuit(qr)
        
        parameters = []
        
        for layer in range(n_layers):
            # Rotation layer
            for i in range(n_qubits):
                theta = Parameter(f'θ_{layer}_{i}')
                phi = Parameter(f'φ_{layer}_{i}')
                lambda_ = Parameter(f'λ_{layer}_{i}')
                
                qc.u(theta, phi, lambda_, i)
                parameters.extend([theta, phi, lambda_])
            
            # Entangling layer
            if n_qubits > 1:
                for i in range(n_qubits - 1):
                    qc.cx(i, i + 1)
                
                if n_qubits > 2:
                    qc.cx(n_qubits - 1, 0)
        
        return qc, parameters
    
    @staticmethod
    def decompose_unitary(target_state: np.ndarray, 
                         initial_state: np.ndarray) -> Tuple[float, float, float]:
        """
        Decomposes the unitary transformation needed to reach target from initial.
        Returns approximate U3 parameters.
        """
        # Normalize states
        target = target_state / np.linalg.norm(target_state)
        initial = initial_state / np.linalg.norm(initial_state)
        
        # Calculate required rotation
        alpha = np.angle(target[0])
        beta = np.angle(target[1])
        
        theta = 2 * np.arccos(np.abs(target[0]))
        phi = beta - alpha
        lambda_ = alpha
        
        return theta, phi, lambda_