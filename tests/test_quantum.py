import pytest
import numpy as np
from src.quantum.gates import QuantumGates
from src.quantum.circuit import QuantumCircuitBuilder
from src.quantum.state import QuantumStateAnalyzer
from src.optimization.optimizer import StateOptimizer
from src.optimization.cost_functions import CostFunctions


class TestQuantumGates:
    """Tests for quantum gates."""
    
    def test_hadamard_gate(self):
        """Test Hadamard gate properties."""
        H = QuantumGates.hadamard()
        
        # H should be unitary
        assert np.allclose(H @ H.conj().T, np.eye(2))
        
        # H should be self-inverse
        assert np.allclose(H @ H, np.eye(2))
    
    def test_pauli_gates(self):
        """Test Pauli gate properties."""
        X = QuantumGates.pauli_x()
        Y = QuantumGates.pauli_y()
        Z = QuantumGates.pauli_z()
        
        # All should be unitary
        assert np.allclose(X @ X.conj().T, np.eye(2))
        assert np.allclose(Y @ Y.conj().T, np.eye(2))
        assert np.allclose(Z @ Z.conj().T, np.eye(2))
        
        # All should be self-inverse
        assert np.allclose(X @ X, np.eye(2))
        assert np.allclose(Y @ Y, np.eye(2))
        assert np.allclose(Z @ Z, np.eye(2))
    
    def test_rotation_gates(self):
        """Test rotation gate properties."""
        theta = np.pi / 4
        
        Rx = QuantumGates.rotation_x(theta)
        Ry = QuantumGates.rotation_y(theta)
        Rz = QuantumGates.rotation_z(theta)
        
        # All should be unitary
        assert np.allclose(Rx @ Rx.conj().T, np.eye(2))
        assert np.allclose(Ry @ Ry.conj().T, np.eye(2))
        assert np.allclose(Rz @ Rz.conj().T, np.eye(2))
    
    def test_cnot_gate(self):
        """Test CNOT gate properties."""
        CNOT = QuantumGates.cnot()
        
        # Should be unitary
        assert np.allclose(CNOT @ CNOT.conj().T, np.eye(4))
        
        # Should be self-inverse
        assert np.allclose(CNOT @ CNOT, np.eye(4))
    
    def test_gate_application(self):
        """Test applying gates to states."""
        state = np.array([1, 0], dtype=complex)
        H = QuantumGates.hadamard()
        
        new_state = QuantumGates.apply_gate(state, H)
        expected = np.array([1, 1]) / np.sqrt(2)
        
        assert np.allclose(new_state, expected)


class TestQuantumCircuit:
    """Tests for quantum circuit builder."""
    
    def test_circuit_initialization(self):
        """Test circuit initialization."""
        builder = QuantumCircuitBuilder(n_qubits=2)
        assert builder.n_qubits == 2
        assert len(builder.qr) == 2
    
    def test_hadamard_circuit(self):
        """Test Hadamard gate in circuit."""
        builder = QuantumCircuitBuilder(n_qubits=1)
        builder.add_hadamard(0)
        
        state = builder.get_statevector()
        expected = np.array([1, 1]) / np.sqrt(2)
        
        assert np.allclose(state, expected)
    
    def test_pauli_x_circuit(self):
        """Test Pauli-X gate in circuit."""
        builder = QuantumCircuitBuilder(n_qubits=1)
        builder.add_pauli_x(0)
        
        state = builder.get_statevector()
        expected = np.array([0, 1])
        
        assert np.allclose(state, expected)
    
    def test_bell_state(self):
        """Test Bell state creation."""
        builder = QuantumCircuitBuilder(n_qubits=2)
        builder.create_bell_state(0, 1)
        
        state = builder.get_statevector()
        expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
        
        assert np.allclose(state, expected)
    
    def test_circuit_execution(self):
        """Test circuit execution with measurements."""
        builder = QuantumCircuitBuilder(n_qubits=1)
        builder.add_pauli_x(0)
        
        counts = builder.execute(shots=1000)
        
        # Should measure |1⟩ with high probability
        assert '1' in counts
        assert counts['1'] > 900


class TestQuantumState:
    """Tests for quantum state analyzer."""
    
    def test_fidelity_identical(self):
        """Test fidelity of identical states."""
        state = np.array([1, 0], dtype=complex)
        fidelity = QuantumStateAnalyzer.fidelity(state, state)
        
        assert np.isclose(fidelity, 1.0)
    
    def test_fidelity_orthogonal(self):
        """Test fidelity of orthogonal states."""
        state1 = np.array([1, 0], dtype=complex)
        state2 = np.array([0, 1], dtype=complex)
        
        fidelity = QuantumStateAnalyzer.fidelity(state1, state2)
        
        assert np.isclose(fidelity, 0.0)
    
    def test_purity_pure_state(self):
        """Test purity of pure state."""
        state = np.array([1, 0], dtype=complex)
        purity = QuantumStateAnalyzer.purity(state)
        
        assert np.isclose(purity, 1.0)
    
    def test_bloch_coordinates(self):
        """Test Bloch sphere coordinate calculation."""
        # |0⟩ state should be at north pole
        state = np.array([1, 0], dtype=complex)
        x, y, z = QuantumStateAnalyzer.bloch_coordinates(state)
        
        assert np.isclose(x, 0.0)
        assert np.isclose(y, 0.0)
        assert np.isclose(z, 1.0)
        
        # |+⟩ state should be on equator
        state = np.array([1, 1]) / np.sqrt(2)
        x, y, z = QuantumStateAnalyzer.bloch_coordinates(state)
        
        assert np.isclose(x, 1.0)
        assert np.isclose(y, 0.0)
        assert np.isclose(z, 0.0, atol=1e-10)
    
    def test_probability_distribution(self):
        """Test probability distribution calculation."""
        state = np.array([1, 1]) / np.sqrt(2)
        probs = QuantumStateAnalyzer.probability_distribution(state)
        
        assert '0' in probs
        assert '1' in probs
        assert np.isclose(probs['0'], 0.5)
        assert np.isclose(probs['1'], 0.5)
    
    def test_expectation_value(self):
        """Test expectation value calculation."""
        state = np.array([1, 0], dtype=complex)
        observable = np.array([[1, 0], [0, -1]])  # Pauli-Z
        
        exp_val = QuantumStateAnalyzer.expectation_value(state, observable)
        
        assert np.isclose(exp_val, 1.0)
    
    def test_concurrence_entangled(self):
        """Test concurrence for entangled state."""
        # Bell state
        state = np.array([1, 0, 0, 1]) / np.sqrt(2)
        concurrence = QuantumStateAnalyzer.concurrence(state)
        
        # Bell state should have maximal entanglement
        assert np.isclose(concurrence, 1.0)
    
    def test_concurrence_separable(self):
        """Test concurrence for separable state."""
        # |00⟩ state
        state = np.array([1, 0, 0, 0], dtype=complex)
        concurrence = QuantumStateAnalyzer.concurrence(state)
        
        assert np.isclose(concurrence, 0.0)


class TestCostFunctions:
    """Tests for cost functions."""
    
    def test_infidelity_cost(self):
        """Test infidelity cost function."""
        target = np.array([1, 0], dtype=complex)
        cost_fn = CostFunctions.infidelity(target)
        
        # Same state should have zero cost
        assert np.isclose(cost_fn(target), 0.0)
        
        # Orthogonal state should have maximum cost
        orthogonal = np.array([0, 1], dtype=complex)
        assert np.isclose(cost_fn(orthogonal), 1.0)
    
    def test_trace_distance_cost(self):
        """Test trace distance cost function."""
        target = np.array([1, 0], dtype=complex)
        cost_fn = CostFunctions.trace_distance_cost(target)
        
        # Same state should have zero distance
        assert np.isclose(cost_fn(target), 0.0, atol=1e-10)
    
    def test_composite_cost(self):
        """Test composite cost function."""
        target = np.array([1, 0], dtype=complex)
        
        cost1 = CostFunctions.infidelity(target)
        cost2 = CostFunctions.trace_distance_cost(target)
        
        composite = CostFunctions.composite_cost([cost1, cost2], [0.5, 0.5])
        
        test_state = np.array([0, 1], dtype=complex)
        result = composite(test_state)
        
        assert result > 0


class TestStateOptimizer:
    """Tests for state optimizer."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        target = np.array([1, 1]) / np.sqrt(2)
        optimizer = StateOptimizer(target, n_qubits=1, n_layers=2)
        
        assert optimizer.n_qubits == 1
        assert optimizer.n_layers == 2
        assert len(optimizer.parameters) == 6  # 3 params per layer * 2 layers
    
    def test_direct_decomposition(self):
        """Test direct unitary decomposition."""
        target = np.array([1, 1]) / np.sqrt(2)  # |+⟩ state
        optimizer = StateOptimizer(target, n_qubits=1, n_layers=1)
        
        results = optimizer.optimize_direct_decomposition()
        
        assert results['fidelity'] > 0.99
        assert results['optimal_state'] is not None
    
    def test_optimization_convergence(self):
        """Test that optimization improves fidelity."""
        target = np.array([1, 1]) / np.sqrt(2)
        optimizer = StateOptimizer(target, n_qubits=1, n_layers=2)
        
        results = optimizer.optimize_scipy(method='COBYLA')
        
        # Should achieve reasonable fidelity
        assert results['fidelity'] > 0.9
        
        # History should show improvement
        assert len(results['history']['fidelity']) > 0
        initial_fidelity = results['history']['fidelity'][0]
        final_fidelity = results['history']['fidelity'][-1]
        assert final_fidelity >= initial_fidelity
    
    def test_optimizer_with_custom_target(self):
        """Test optimizer with custom target state."""
        # Target: equal superposition of |0⟩ and |1⟩ with phase
        target = np.array([1, 1j]) / np.sqrt(2)
        optimizer = StateOptimizer(target, n_qubits=1, n_layers=3)
        
        results = optimizer.optimize_scipy(method='COBYLA')
        
        assert results['fidelity'] > 0.85


class TestIntegration:
    """Integration tests."""
    
    def test_full_pipeline(self):
        """Test complete optimization pipeline."""
        # Define target state
        target = np.array([1, 1]) / np.sqrt(2)
        
        # Create optimizer
        optimizer = StateOptimizer(target, n_qubits=1, n_layers=2)
        
        # Optimize
        results = optimizer.optimize_scipy(method='COBYLA')
        
        # Verify results
        assert 'optimal_parameters' in results
        assert 'optimal_state' in results
        assert 'fidelity' in results
        assert results['fidelity'] > 0.9
        
        # Get optimized circuit
        circuit = optimizer.get_optimized_circuit()
        assert circuit is not None
    
    def test_hadamard_gate_optimization(self):
        """Test optimizing to reach Hadamard output."""
        # Initial state |0⟩
        # Target state |+⟩ (Hadamard of |0⟩)
        target = np.array([1, 1]) / np.sqrt(2)
        
        optimizer = StateOptimizer(target, n_qubits=1, n_layers=1)
        results = optimizer.optimize_direct_decomposition()
        
        # Should achieve high fidelity
        assert results['fidelity'] > 0.99
        
        # Verify the final state
        final_state = results['optimal_state']
        fidelity = QuantumStateAnalyzer.fidelity(final_state, target)
        assert fidelity > 0.99


if __name__ == '__main__':
    pytest.main([__file__, '-v'])