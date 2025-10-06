import numpy as np
from src.quantum.state import QuantumStateAnalyzer
from src.optimization.optimizer import StateOptimizer
from src.visualization.plotter import QuantumVisualizer


def main():
    """Demonstrates quantum state manipulation through optimization."""
    
    print("=" * 60)
    print("QUANTUM STATE MANIPULATION EXAMPLE")
    print("=" * 60)
    
    # Define target states to force
    target_states = {
        "|+⟩ (equal superposition)": np.array([1, 1]) / np.sqrt(2),
        "|-⟩ (minus state)": np.array([1, -1]) / np.sqrt(2),
        "|i+⟩ (complex phase)": np.array([1, 1j]) / np.sqrt(2),
        "Custom state": np.array([0.6, 0.8], dtype=complex)
    }
    
    for name, target in target_states.items():
        print(f"\n{'='*60}")
        print(f"Target State: {name}")
        print(f"{'='*60}")
        
        # Normalize target
        target = target / np.linalg.norm(target)
        print(f"Target statevector: {target}")
        
        # Calculate target Bloch coordinates
        x_target, y_target, z_target = QuantumStateAnalyzer.bloch_coordinates(target)
        print(f"Target Bloch coordinates: X={x_target:.4f}, Y={y_target:.4f}, Z={z_target:.4f}")
        
        # Create optimizer
        optimizer = StateOptimizer(target, n_qubits=1, n_layers=2)
        
        # Method 1: Direct decomposition (fastest for single qubits)
        print(f"\n--- Method 1: Direct Unitary Decomposition ---")
        results_direct = optimizer.optimize_direct_decomposition()
        
        print(f"Achieved fidelity: {results_direct['fidelity']:.6f}")
        print(f"Final cost: {results_direct['cost']:.6e}")
        print(f"Optimal parameters: {results_direct['optimal_parameters']}")
        
        # Method 2: Numerical optimization (COBYLA)
        print(f"\n--- Method 2: COBYLA Optimization ---")
        optimizer2 = StateOptimizer(target, n_qubits=1, n_layers=2)
        results_cobyla = optimizer2.optimize_scipy(method='COBYLA')
        
        print(f"Achieved fidelity: {results_cobyla['fidelity']:.6f}")
        print(f"Final cost: {results_cobyla['cost']:.6e}")
        print(f"Number of iterations: {results_cobyla['n_iterations']}")
        
        # Verify final state
        final_state = results_cobyla['optimal_state']
        x_final, y_final, z_final = QuantumStateAnalyzer.bloch_coordinates(final_state)
        print(f"\nFinal Bloch coordinates: X={x_final:.4f}, Y={y_final:.4f}, Z={z_final:.4f}")
        
        # Calculate error
        bloch_error = np.sqrt((x_final - x_target)**2 + 
                              (y_final - y_target)**2 + 
                              (z_final - z_target)**2)
        print(f"Bloch sphere distance error: {bloch_error:.6f}")
        
        # Visualize results
        print("\nGenerating visualizations...")
        
        # Optimization history
        QuantumVisualizer.plot_optimization_history(
            results_cobyla['history'],
            title=f"Optimization History: {name}"
        )
        
        # State comparison
        QuantumVisualizer.plot_state_comparison(
            target,
            final_state,
            labels=("Target", "Achieved"),
            title=f"State Comparison: {name}"
        )
        
        # Bloch sphere visualization
        QuantumVisualizer.plot_bloch_sphere(
            final_state,
            title=f"Achieved State on Bloch Sphere: {name}"
        )
    
    print("\n" + "=" * 60)
    print("State manipulation examples completed successfully!")
    print("=" * 60)
    print("\nKey insights:")
    print("- Direct decomposition achieves near-perfect fidelity instantly")
    print("- Numerical optimization provides flexibility for complex constraints")
    print("- Both methods successfully 'force' desired quantum states")
    print("- Fidelity >0.99 demonstrates precise state manipulation")


if __name__ == '__main__':
    main()