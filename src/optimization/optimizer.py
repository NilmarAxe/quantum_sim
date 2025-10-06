import numpy as np
from typing import Callable, Dict, List, Tuple, Optional
from scipy.optimize import minimize
from qiskit import QuantumCircuit

from src.quantum.gates import QuantumGates
from src.quantum.circuit import QuantumCircuitBuilder
from src.quantum.state import QuantumStateAnalyzer
from src.optimization.cost_functions import CostFunctions
from config.settings import QuantumConfig


class StateOptimizer:
    """Optimizes quantum circuits to achieve desired states."""
    
    def __init__(self, target_state: np.ndarray, n_qubits: int = 1, n_layers: int = 3):
        """
        Initialize optimizer.
        
        Args:
            target_state: Desired quantum state
            n_qubits: Number of qubits
            n_layers: Number of variational layers
        """
        self.target_state = target_state / np.linalg.norm(target_state)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Create parameterized circuit
        self.param_circuit, self.parameters = QuantumGates.create_parameterized_circuit(
            n_qubits, n_layers
        )
        
        # Cost function
        self.cost_function = CostFunctions.infidelity(self.target_state)
        
        # Optimization history
        self.history = {
            'cost': [],
            'fidelity': [],
            'parameters': [],
            'gradients': []
        }
        
        self.best_parameters = None
        self.best_fidelity = 0.0
        
    def _evaluate_circuit(self, parameters: np.ndarray) -> np.ndarray:
        """
        Evaluates circuit with given parameters.
        
        Args:
            parameters: Circuit parameters
            
        Returns:
            Resulting quantum state
        """
        builder = QuantumCircuitBuilder(self.n_qubits)
        builder.from_parameters(parameters.tolist(), self.param_circuit)
        return builder.get_statevector()
    
    def _cost_wrapper(self, parameters: np.ndarray) -> float:
        """
        Wrapper for cost function evaluation.
        
        Args:
            parameters: Circuit parameters
            
        Returns:
            Cost value
        """
        state = self._evaluate_circuit(parameters)
        cost = self.cost_function(state)
        
        # Track metrics
        fidelity = QuantumStateAnalyzer.fidelity(state, self.target_state)
        self.history['cost'].append(cost)
        self.history['fidelity'].append(fidelity)
        self.history['parameters'].append(parameters.copy())
        
        # Update best
        if fidelity > self.best_fidelity:
            self.best_fidelity = fidelity
            self.best_parameters = parameters.copy()
        
        if QuantumConfig.VERBOSE and len(self.history['cost']) % QuantumConfig.LOG_INTERVAL == 0:
            print(f"Iteration {len(self.history['cost'])}: Cost = {cost:.6f}, Fidelity = {fidelity:.6f}")
        
        return cost
    
    def _gradient(self, parameters: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        """
        Computes numerical gradient using parameter shift rule.
        
        Args:
            parameters: Current parameters
            epsilon: Finite difference step
            
        Returns:
            Gradient vector
        """
        grad = np.zeros_like(parameters)
        
        for i in range(len(parameters)):
            params_plus = parameters.copy()
            params_minus = parameters.copy()
            
            params_plus[i] += epsilon
            params_minus[i] -= epsilon
            
            cost_plus = self._cost_wrapper(params_plus)
            cost_minus = self._cost_wrapper(params_minus)
            
            grad[i] = (cost_plus - cost_minus) / (2 * epsilon)
        
        self.history['gradients'].append(grad.copy())
        return grad
    
    def optimize_gradient_descent(self, 
                                  initial_params: Optional[np.ndarray] = None,
                                  max_iterations: int = QuantumConfig.MAX_ITERATIONS) -> Dict:
        """
        Optimizes using gradient descent.
        
        Args:
            initial_params: Initial parameter values
            max_iterations: Maximum number of iterations
            
        Returns:
            Optimization results dictionary
        """
        if initial_params is None:
            initial_params = np.random.uniform(-np.pi, np.pi, len(self.parameters))
        
        params = initial_params.copy()
        learning_rate = QuantumConfig.LEARNING_RATE
        
        for iteration in range(max_iterations):
            grad = self._gradient(params)
            params -= learning_rate * grad
            
            # Check convergence
            if np.linalg.norm(grad) < QuantumConfig.CONVERGENCE_THRESHOLD:
                print(f"Converged at iteration {iteration}")
                break
        
        return self._prepare_results(params)
    
    def optimize_adam(self,
                     initial_params: Optional[np.ndarray] = None,
                     max_iterations: int = QuantumConfig.MAX_ITERATIONS,
                     beta1: float = 0.9,
                     beta2: float = 0.999,
                     epsilon: float = 1e-8) -> Dict:
        """
        Optimizes using Adam optimizer.
        
        Args:
            initial_params: Initial parameter values
            max_iterations: Maximum number of iterations
            beta1: First moment decay rate
            beta2: Second moment decay rate
            epsilon: Small constant for numerical stability
            
        Returns:
            Optimization results dictionary
        """
        if initial_params is None:
            initial_params = np.random.uniform(-np.pi, np.pi, len(self.parameters))
        
        params = initial_params.copy()
        m = np.zeros_like(params)  # First moment
        v = np.zeros_like(params)  # Second moment
        learning_rate = QuantumConfig.LEARNING_RATE
        
        for t in range(1, max_iterations + 1):
            grad = self._gradient(params)
            
            # Update biased moments
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            # Update parameters
            params -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            
            # Check convergence
            if np.linalg.norm(grad) < QuantumConfig.CONVERGENCE_THRESHOLD:
                print(f"Converged at iteration {t}")
                break
        
        return self._prepare_results(params)
    
    def optimize_scipy(self, 
                      method: str = 'COBYLA',
                      initial_params: Optional[np.ndarray] = None) -> Dict:
        """
        Optimizes using scipy optimization methods.
        
        Args:
            method: Optimization method ('COBYLA', 'Nelder-Mead', 'BFGS', etc.)
            initial_params: Initial parameter values
            
        Returns:
            Optimization results dictionary
        """
        if initial_params is None:
            initial_params = np.random.uniform(-np.pi, np.pi, len(self.parameters))
        
        result = minimize(
            self._cost_wrapper,
            initial_params,
            method=method,
            options={'maxiter': QuantumConfig.MAX_ITERATIONS}
        )
        
        return self._prepare_results(result.x)
    
    def optimize_direct_decomposition(self) -> Dict:
        """
        Uses direct unitary decomposition for single qubit optimization.
        
        Returns:
            Optimization results dictionary
        """
        if self.n_qubits != 1:
            raise ValueError("Direct decomposition only works for single qubits")
        
        initial_state = np.array([1.0, 0.0], dtype=complex)
        theta, phi, lambda_ = QuantumGates.decompose_unitary(
            self.target_state, initial_state
        )
        
        # For single layer, single qubit
        if self.n_layers == 1:
            optimal_params = np.array([theta, phi, lambda_])
        else:
            # Distribute across layers
            optimal_params = np.tile([theta/self.n_layers, phi/self.n_layers, 
                                     lambda_/self.n_layers], self.n_layers)
        
        return self._prepare_results(optimal_params)
    
    def _prepare_results(self, final_params: np.ndarray) -> Dict:
        """
        Prepares optimization results.
        
        Args:
            final_params: Final optimized parameters
            
        Returns:
            Results dictionary
        """
        final_state = self._evaluate_circuit(final_params)
        final_fidelity = QuantumStateAnalyzer.fidelity(final_state, self.target_state)
        
        return {
            'optimal_parameters': final_params,
            'optimal_state': final_state,
            'fidelity': final_fidelity,
            'cost': self.cost_function(final_state),
            'history': self.history,
            'n_iterations': len(self.history['cost']),
            'best_parameters': self.best_parameters,
            'best_fidelity': self.best_fidelity
        }
    
    def get_optimized_circuit(self, parameters: np.ndarray = None) -> QuantumCircuit:
        """
        Returns optimized quantum circuit.
        
        Args:
            parameters: Parameters to use (default: best found)
            
        Returns:
            Quantum circuit
        """
        if parameters is None:
            parameters = self.best_parameters
        
        builder = QuantumCircuitBuilder(self.n_qubits)
        builder.from_parameters(parameters.tolist(), self.param_circuit)
        return builder.get_circuit()