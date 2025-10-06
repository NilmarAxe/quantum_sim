import numpy as np
from typing import Callable
from src.quantum.state import QuantumStateAnalyzer


class CostFunctions:
    """Collection of cost functions for state optimization."""
    
    @staticmethod
    def infidelity(target_state: np.ndarray) -> Callable:
        """
        Creates infidelity cost function.
        
        Args:
            target_state: Desired quantum state
            
        Returns:
            Cost function that takes current state and returns infidelity
        """
        target_state = target_state / np.linalg.norm(target_state)
        
        def cost(current_state: np.ndarray) -> float:
            fidelity = QuantumStateAnalyzer.fidelity(current_state, target_state)
            return 1.0 - fidelity
        
        return cost
    
    @staticmethod
    def trace_distance_cost(target_state: np.ndarray) -> Callable:
        """
        Creates trace distance cost function.
        
        Args:
            target_state: Desired quantum state
            
        Returns:
            Cost function based on trace distance
        """
        target_state = target_state / np.linalg.norm(target_state)
        
        def cost(current_state: np.ndarray) -> float:
            return QuantumStateAnalyzer.trace_distance(current_state, target_state)
        
        return cost
    
    @staticmethod
    def expectation_value_cost(observable: np.ndarray, target_value: float) -> Callable:
        """
        Creates cost function based on expectation value.
        
        Args:
            observable: Observable operator
            target_value: Target expectation value
            
        Returns:
            Cost function
        """
        def cost(current_state: np.ndarray) -> float:
            exp_val = QuantumStateAnalyzer.expectation_value(current_state, observable)
            return np.abs(exp_val - target_value) ** 2
        
        return cost
    
    @staticmethod
    def entropy_cost(target_entropy: float) -> Callable:
        """
        Creates cost function based on entropy.
        
        Args:
            target_entropy: Target entropy value
            
        Returns:
            Cost function
        """
        def cost(current_state: np.ndarray) -> float:
            entropy = QuantumStateAnalyzer.von_neumann_entropy(current_state)
            return (entropy - target_entropy) ** 2
        
        return cost
    
    @staticmethod
    def bloch_vector_cost(target_x: float, target_y: float, target_z: float) -> Callable:
        """
        Creates cost function based on Bloch sphere coordinates.
        
        Args:
            target_x, target_y, target_z: Target Bloch coordinates
            
        Returns:
            Cost function
        """
        def cost(current_state: np.ndarray) -> float:
            x, y, z = QuantumStateAnalyzer.bloch_coordinates(current_state)
            return (x - target_x)**2 + (y - target_y)**2 + (z - target_z)**2
        
        return cost
    
    @staticmethod
    def composite_cost(cost_functions: list, weights: list = None) -> Callable:
        """
        Creates weighted composite cost function.
        
        Args:
            cost_functions: List of cost functions
            weights: List of weights (default: equal weights)
            
        Returns:
            Composite cost function
        """
        if weights is None:
            weights = [1.0] * len(cost_functions)
        
        if len(weights) != len(cost_functions):
            raise ValueError("Number of weights must match number of cost functions")
        
        def cost(current_state: np.ndarray) -> float:
            total = 0.0
            for w, cf in zip(weights, cost_functions):
                total += w * cf(current_state)
            return total
        
        return cost
    
    @staticmethod
    def regularized_cost(base_cost: Callable, 
                        regularization_strength: float = 0.01) -> Callable:
        """
        Adds L2 regularization to a cost function.
        
        Args:
            base_cost: Base cost function
            regularization_strength: Regularization parameter
            
        Returns:
            Regularized cost function
        """
        def cost(current_state: np.ndarray) -> float:
            base_value = base_cost(current_state)
            reg_term = regularization_strength * np.linalg.norm(current_state) ** 2
            return base_value + reg_term
        
        return cost
    
    @staticmethod
    def hilbert_schmidt_cost(target_state: np.ndarray) -> Callable:
        """
        Creates Hilbert-Schmidt distance cost function.
        
        Args:
            target_state: Desired quantum state
            
        Returns:
            Cost function based on Hilbert-Schmidt distance
        """
        target_state = target_state / np.linalg.norm(target_state)
        target_rho = np.outer(target_state, np.conj(target_state))
        
        def cost(current_state: np.ndarray) -> float:
            current_rho = np.outer(current_state, np.conj(current_state))
            diff = target_rho - current_rho
            return np.real(np.trace(diff @ diff.conj().T))
        
        return cost
    
    @staticmethod
    def purity_cost(target_purity: float) -> Callable:
        """
        Creates cost function based on state purity.
        
        Args:
            target_purity: Target purity value (0 to 1)
            
        Returns:
            Cost function
        """
        def cost(current_state: np.ndarray) -> float:
            purity = QuantumStateAnalyzer.purity(current_state)
            return (purity - target_purity) ** 2
        
        return cost
    
    @staticmethod
    def overlap_cost(target_state: np.ndarray) -> Callable:
        """
        Creates cost function based on state overlap.
        
        Args:
            target_state: Desired quantum state
            
        Returns:
            Cost function that minimizes negative overlap
        """
        target_state = target_state / np.linalg.norm(target_state)
        
        def cost(current_state: np.ndarray) -> float:
            overlap = QuantumStateAnalyzer.overlap(current_state, target_state)
            return 1.0 - np.abs(overlap)
        
        return cost
    
    @staticmethod
    def energy_cost(hamiltonian: np.ndarray) -> Callable:
        """
        Creates cost function for energy minimization.
        
        Args:
            hamiltonian: Hamiltonian operator matrix
            
        Returns:
            Cost function that computes expectation value of Hamiltonian
        """
        def cost(current_state: np.ndarray) -> float:
            energy = QuantumStateAnalyzer.expectation_value(current_state, hamiltonian)
            return np.real(energy)
        
        return cost
    
    @staticmethod
    def variance_cost(observable: np.ndarray, target_variance: float = 0.0) -> Callable:
        """
        Creates cost function based on measurement variance.
        
        Args:
            observable: Observable operator
            target_variance: Target variance value
            
        Returns:
            Cost function
        """
        def cost(current_state: np.ndarray) -> float:
            exp_val = QuantumStateAnalyzer.expectation_value(current_state, observable)
            exp_val_sq = QuantumStateAnalyzer.expectation_value(
                current_state, 
                observable @ observable
            )
            variance = np.real(exp_val_sq - exp_val ** 2)
            return (variance - target_variance) ** 2
        
        return cost
    
    @staticmethod
    def entanglement_cost(target_concurrence: float) -> Callable:
        """
        Creates cost function for two-qubit entanglement.
        
        Args:
            target_concurrence: Target concurrence value (0 to 1)
            
        Returns:
            Cost function
        """
        def cost(current_state: np.ndarray) -> float:
            if len(current_state) != 4:
                raise ValueError("Entanglement cost requires two-qubit state")
            
            concurrence = QuantumStateAnalyzer.concurrence(current_state)
            return (concurrence - target_concurrence) ** 2
        
        return cost
    
    @staticmethod
    def adaptive_cost(target_state: np.ndarray, 
                     phase: str = 'exploration') -> Callable:
        """
        Creates adaptive cost function that changes behavior based on optimization phase.
        
        Args:
            target_state: Desired quantum state
            phase: 'exploration' or 'exploitation'
            
        Returns:
            Adaptive cost function
        """
        target_state = target_state / np.linalg.norm(target_state)
        
        if phase == 'exploration':
            # More forgiving during exploration
            def cost(current_state: np.ndarray) -> float:
                fidelity = QuantumStateAnalyzer.fidelity(current_state, target_state)
                return np.sqrt(1.0 - fidelity)
        else:
            # More precise during exploitation
            def cost(current_state: np.ndarray) -> float:
                fidelity = QuantumStateAnalyzer.fidelity(current_state, target_state)
                return (1.0 - fidelity) ** 2
        
        return cost
    
    @staticmethod
    def smooth_cost(base_cost: Callable, 
                   smoothing_factor: float = 0.1) -> Callable:
        """
        Creates smoothed version of a cost function.
        
        Args:
            base_cost: Base cost function to smooth
            smoothing_factor: Amount of smoothing (0 to 1)
            
        Returns:
            Smoothed cost function
        """
        def cost(current_state: np.ndarray) -> float:
            base_value = base_cost(current_state)
            # Add small Gaussian noise for smoothing
            noise = smoothing_factor * np.random.randn()
            return base_value + noise
        
        return cost
    
    @staticmethod
    def barrier_cost(base_cost: Callable, 
                    barrier_strength: float = 1000.0) -> Callable:
        """
        Adds barrier to prevent non-normalized states.
        
        Args:
            base_cost: Base cost function
            barrier_strength: Strength of normalization barrier
            
        Returns:
            Cost function with barrier
        """
        def cost(current_state: np.ndarray) -> float:
            base_value = base_cost(current_state)
            
            # Add barrier for non-normalized states
            norm = np.linalg.norm(current_state)
            barrier = barrier_strength * (norm - 1.0) ** 2
            
            return base_value + barrier
        
        return cost