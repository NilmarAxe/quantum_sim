import numpy as np
from src.quantum.circuit import QuantumCircuitBuilder
from src.quantum.state import QuantumStateAnalyzer
from src.visualization.plotter import QuantumVisualizer


def main():
    """Demonstrates basic Hadamard gate application."""
    
    print("=" * 60)
    print("BASIC HADAMARD GATE EXAMPLE")
    print("=" * 60)
    
    # Create quantum circuit with 1 qubit
    builder = QuantumCircuitBuilder(n_qubits=1)
    
    # Initial state |0⟩
    initial_state = builder.get_statevector()
    print(f"\nInitial state |0⟩:")
    print(f"Statevector: {initial_state}")
    print(f"Probability distribution: {QuantumStateAnalyzer.probability_distribution(initial_state)}")
    
    # Apply Hadamard gate
    builder.add_hadamard(0)
    
    # Get resulting state |+⟩
    final_state = builder.get_statevector()
    print(f"\nFinal state |+⟩ (after Hadamard):")
    print(f"Statevector: {final_state}")
    print(f"Probability distribution: {QuantumStateAnalyzer.probability_distribution(final_state)}")
    
    # Calculate Bloch coordinates
    x, y, z = QuantumStateAnalyzer.bloch_coordinates(final_state)
    print(f"\nBloch sphere coordinates:")
    print(f"X: {x:.4f}, Y: {y:.4f}, Z: {z:.4f}")
    
    # Measure fidelity with expected |+⟩ state
    expected_plus = np.array([1, 1]) / np.sqrt(2)
    fidelity = QuantumStateAnalyzer.fidelity(final_state, expected_plus)
    print(f"\nFidelity with |+⟩ state: {fidelity:.6f}")
    
    # Execute circuit with measurements
    print(f"\nExecuting circuit with 1000 shots...")
    counts = builder.execute(shots=1000)
    print(f"Measurement results: {counts}")
    
    # Visualize
    print("\nGenerating visualizations...")
    
    # Bloch sphere
    QuantumVisualizer.plot_bloch_sphere(
        final_state,
        title="Hadamard Gate Output on Bloch Sphere"
    )
    
    # State distribution
    QuantumVisualizer.plot_state_distribution(
        final_state,
        title="Hadamard Gate Output Distribution"
    )
    
    # Compare initial and final states
    QuantumVisualizer.plot_state_comparison(
        initial_state,
        final_state,
        labels=("Initial |0⟩", "Final |+⟩"),
        title="State Evolution: Hadamard Gate"
    )
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()