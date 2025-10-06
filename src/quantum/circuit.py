from typing import Dict, List, Optional, Union
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

from config.settings import QuantumConfig


class QuantumCircuitBuilder:
    """Builds and executes quantum circuits with various configurations."""
    
    def __init__(self, n_qubits: int = 1):
        """
        Initialize circuit builder.
        
        Args:
            n_qubits: Number of qubits in the circuit
        """
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, 'q')
        self.cr = ClassicalRegister(n_qubits, 'c')
        self.circuit = QuantumCircuit(self.qr, self.cr)
        self.simulator = AerSimulator()
        
    def reset(self):
        """Resets the circuit to initial state."""
        self.circuit = QuantumCircuit(self.qr, self.cr)
        
    def add_hadamard(self, qubit: int = 0):
        """Adds Hadamard gate to specified qubit."""
        self.circuit.h(qubit)
        return self
        
    def add_pauli_x(self, qubit: int = 0):
        """Adds Pauli-X gate to specified qubit."""
        self.circuit.x(qubit)
        return self
        
    def add_pauli_y(self, qubit: int = 0):
        """Adds Pauli-Y gate to specified qubit."""
        self.circuit.y(qubit)
        return self
        
    def add_pauli_z(self, qubit: int = 0):
        """Adds Pauli-Z gate to specified qubit."""
        self.circuit.z(qubit)
        return self
        
    def add_rotation_x(self, theta: float, qubit: int = 0):
        """Adds RX rotation gate."""
        self.circuit.rx(theta, qubit)
        return self
        
    def add_rotation_y(self, theta: float, qubit: int = 0):
        """Adds RY rotation gate."""
        self.circuit.ry(theta, qubit)
        return self
        
    def add_rotation_z(self, phi: float, qubit: int = 0):
        """Adds RZ rotation gate."""
        self.circuit.rz(phi, qubit)
        return self
        
    def add_u3(self, theta: float, phi: float, lambda_: float, qubit: int = 0):
        """Adds U3 universal gate."""
        self.circuit.u(theta, phi, lambda_, qubit)
        return self
        
    def add_cnot(self, control: int, target: int):
        """Adds CNOT gate."""
        self.circuit.cx(control, target)
        return self
        
    def add_measurement(self, qubits: Optional[List[int]] = None):
        """
        Adds measurement to specified qubits.
        
        Args:
            qubits: List of qubit indices to measure. If None, measures all.
        """
        if qubits is None:
            self.circuit.measure(self.qr, self.cr)
        else:
            for i, q in enumerate(qubits):
                self.circuit.measure(q, i)
        return self
        
    def get_statevector(self) -> np.ndarray:
        """
        Returns the statevector of the current circuit.
        
        Returns:
            Complex numpy array representing the quantum state
        """
        sv = Statevector.from_instruction(self.circuit)
        return sv.data
        
    def execute(self, shots: int = QuantumConfig.SHOTS) -> Dict[str, int]:
        """
        Executes the circuit and returns measurement results.
        
        Args:
            shots: Number of measurement shots
            
        Returns:
            Dictionary of measurement outcomes
        """
        # Add measurements if not present
        if not any(isinstance(instr.operation, type(self.circuit.measure(0, 0).operation)) 
                   for instr, _, _ in self.circuit.data):
            self.add_measurement()
        
        compiled_circuit = transpile(self.circuit, self.simulator)
        job = self.simulator.run(compiled_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        return counts
        
    def get_circuit(self) -> QuantumCircuit:
        """Returns the current quantum circuit."""
        return self.circuit
        
    def from_parameters(self, parameters: List[float], 
                       base_circuit: QuantumCircuit) -> 'QuantumCircuitBuilder':
        """
        Binds parameters to a parameterized circuit.
        
        Args:
            parameters: List of parameter values
            base_circuit: Parameterized quantum circuit
            
        Returns:
            Self for chaining
        """
        param_dict = {p: v for p, v in zip(base_circuit.parameters, parameters)}
        self.circuit = base_circuit.assign_parameters(param_dict)
        return self
        
    def apply_custom_gate(self, gate_matrix: np.ndarray, qubits: List[int]):
        """
        Applies a custom unitary gate.
        
        Args:
            gate_matrix: Unitary matrix of the gate
            qubits: Target qubits
        """
        self.circuit.unitary(gate_matrix, qubits, label='custom')
        return self
        
    def create_bell_state(self, qubit1: int = 0, qubit2: int = 1):
        """Creates a Bell state between two qubits."""
        self.circuit.h(qubit1)
        self.circuit.cx(qubit1, qubit2)
        return self
        
    def create_ghz_state(self):
        """Creates a GHZ state across all qubits."""
        self.circuit.h(0)
        for i in range(self.n_qubits - 1):
            self.circuit.cx(i, i + 1)
        return self
        
    def draw(self, output: str = 'text') -> str:
        """
        Returns a drawing of the circuit.
        
        Args:
            output: Output format ('text', 'mpl', 'latex')
        """
        return self.circuit.draw(output=output)
        
    def depth(self) -> int:
        """Returns the depth of the circuit."""
        return self.circuit.depth()
        
    def count_ops(self) -> Dict[str, int]:
        """Returns operation counts in the circuit."""
        return self.circuit.count_ops()