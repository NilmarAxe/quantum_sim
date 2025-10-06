# Quantum State Manipulation System

A comprehensive framework for quantum circuit simulation and state optimization, featuring advanced optimization algorithms to "force" desired quantum states.

## Features

- **Quantum Gate Implementation**: Complete library of single and multi-qubit gates
- **Circuit Building**: Intuitive API for constructing quantum circuits
- **State Analysis**: Tools for analyzing quantum states (fidelity, entropy, entanglement)
- **Advanced Optimization**: Multiple algorithms to manipulate states:
  - Direct unitary decomposition
  - Gradient descent
  - Adam optimizer
  - Scipy optimizers (COBYLA, Nelder-Mead, BFGS)
- **Visualization**: Bloch sphere, state distributions, optimization convergence
- **Comprehensive Testing**: Full test suite with pytest

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
quantum_sim/
├── src/
│   ├── quantum/          # Core quantum operations
│   │   ├── gates.py      # Quantum gates
│   │   ├── circuit.py    # Circuit builder
│   │   └── state.py      # State analysis
│   ├── optimization/     # Optimization algorithms
│   │   ├── optimizer.py  # Main optimizer class
│   │   └── cost_functions.py
│   └── visualization/    # Plotting utilities
│       └── plotter.py
├── examples/             # Usage examples
│   ├── basic_hadamard.py
│   ├── state_manipulation.py
│   └── advanced_optimization.py
├── tests/                # Unit tests
├── config/               # Configuration
└── main.py              # Interactive CLI
```

## Quick Start

### Interactive Mode
```bash
python main.py
```

### Basic Hadamard Gate
```python
from src.quantum.circuit import QuantumCircuitBuilder
from src.quantum.state import QuantumStateAnalyzer

# Create circuit and apply Hadamard
builder = QuantumCircuitBuilder(n_qubits=1)
builder.add_hadamard(0)

# Get state
state = builder.get_statevector()
print(f"State: {state}")

# Analyze
probs = QuantumStateAnalyzer.probability_distribution(state)
print(f"Probabilities: {probs}")
```

### Force a Desired State
```python
import numpy as np
from src.optimization.optimizer import StateOptimizer

# Define target state (e.g., |+⟩)
target = np.array([1, 1]) / np.sqrt(2)

# Create optimizer
optimizer = StateOptimizer(target, n_qubits=1, n_layers=2)

# Optimize to force this state
results = optimizer.optimize_scipy(method='COBYLA')

print(f"Achieved fidelity: {results['fidelity']:.6f}")
print(f"Optimal parameters: {results['optimal_parameters']}")
```

### Advanced Optimization
```python
# Compare different optimization methods
from src.optimization.optimizer import StateOptimizer

target = np.array([0.6+0.2j, 0.7-0.3j])
target = target / np.linalg.norm(target)

# Method 1: Direct decomposition (fastest)
opt1 = StateOptimizer(target, n_qubits=1, n_layers=1)
result1 = opt1.optimize_direct_decomposition()

# Method 2: Numerical optimization
opt2 = StateOptimizer(target, n_qubits=1, n_layers=2)
result2 = opt2.optimize_scipy(method='COBYLA')

# Method 3: Adam optimizer
opt3 = StateOptimizer(target, n_qubits=1, n_layers=2)
result3 = opt3.optimize_adam(max_iterations=100)
```

## Examples

Run individual examples:
```bash
python examples/basic_hadamard.py
python examples/state_manipulation.py
python examples/advanced_optimization.py
```

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run specific test:
```bash
pytest tests/test_quantum.py::TestQuantumGates::test_hadamard_gate -v
```

## Key Concepts

### State Manipulation
The system allows you to "force" a quantum circuit to produce any desired single-qubit state by:
1. Defining a target state
2. Creating a parameterized variational circuit
3. Optimizing parameters to maximize fidelity with the target

### Optimization Methods

- **Direct Decomposition**: Analytical solution for single qubits (instant, perfect fidelity)
- **COBYLA**: Constrained optimization by linear approximation
- **Nelder-Mead**: Simplex-based gradient-free method
- **Adam**: Adaptive moment estimation with momentum
- **Gradient Descent**: First-order iterative optimization

### Fidelity Metrics
- **State Fidelity**: Overlap between achieved and target states (0 to 1)
- **Trace Distance**: Metric on the space of density matrices
- **Bloch Distance**: Euclidean distance on Bloch sphere

## Configuration

Edit `config/settings.py` to customize:
- Number of shots for simulation
- Optimization parameters (learning rate, convergence threshold)
- Visualization settings
- Default target states

## Visualization

The system provides rich visualizations:
- **Bloch Sphere**: 3D representation of single-qubit states
- **State Distributions**: Probability distributions and amplitudes
- **Optimization History**: Convergence plots, parameter evolution
- **State Comparison**: Side-by-side analysis of states

## Advanced Features

### Custom Cost Functions
```python
from src.optimization.cost_functions import CostFunctions

# Create custom cost function
def my_cost(target_state):
    def cost(current_state):
        # Your custom logic here
        return some_value
    return cost

optimizer.cost_function = my_cost(target_state)
```

### Multi-Layer Circuits
```python
# Increase expressibility with more layers
optimizer = StateOptimizer(target, n_qubits=1, n_layers=5)
```

### Parameter Landscape Exploration
```python
# Run multiple optimizations with random initializations
for i in range(20):
    initial_params = np.random.uniform(-np.pi, np.pi, len(params))
    result = optimizer.optimize_scipy(initial_params=initial_params)
```

## Performance Tips

1. **Direct decomposition** is fastest for single qubits
2. **COBYLA** works well for most cases
3. Increase **n_layers** for complex targets
4. Use **Adam** for smoother convergence
5. Multiple **random initializations** help escape local minima

## Contributing

Contributions are welcome! Areas for improvement:
- Multi-qubit state optimization
- Additional gate sets
- Hardware-specific noise models
- GPU acceleration
- More visualization options

## License

This project is provided as-is for educational and research purposes.

## References

- Nielsen & Chuang: "Quantum Computation and Quantum Information"
- Qiskit Documentation: https://qiskit.org/documentation/
- Quantum optimization: VQE and QAOA algorithms

## Support

For issues or questions, please refer to the inline documentation in the code or examine the example files.
