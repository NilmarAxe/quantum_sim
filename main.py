import sys
import numpy as np
from src.quantum.circuit import QuantumCircuitBuilder
from src.quantum.state import QuantumStateAnalyzer
from src.optimization.optimizer import StateOptimizer
from src.visualization.plotter import QuantumVisualizer
from config.settings import QuantumConfig


def print_banner():
    """Prints system banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║     QUANTUM STATE MANIPULATION SYSTEM                     ║
    ║     Advanced Quantum Circuit Optimization Framework       ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_menu():
    """Prints main menu."""
    menu = """
    ┌───────────────────────────────────────────────────────────┐
    │  MAIN MENU                                                │
    ├───────────────────────────────────────────────────────────┤
    │  1. Basic Hadamard Gate Simulation                        │
    │  2. State Manipulation (Force Desired State)              │
    │  3. Custom Target State Optimization                      │
    │  4. Compare Optimization Algorithms                       │
    │  5. Multi-Qubit Entanglement (Bell States)                │
    │  6. Parameter Landscape Analysis                          │
    │  7. Run All Examples                                      │
    │  8. Interactive State Designer                            │
    │  0. Exit                                                  │
    └───────────────────────────────────────────────────────────┘
    """
    print(menu)


def hadamard_simulation():
    """Runs basic Hadamard gate simulation."""
    print("\n" + "=" * 60)
    print("HADAMARD GATE SIMULATION")
    print("=" * 60)
    
    builder = QuantumCircuitBuilder(n_qubits=1)
    
    # Initial state
    initial = builder.get_statevector()
    print(f"\nInitial state |0⟩: {initial}")
    
    # Apply Hadamard
    builder.add_hadamard(0)
    final = builder.get_statevector()
    
    print(f"Final state |+⟩: {final}")
    print(f"Probability distribution: {QuantumStateAnalyzer.probability_distribution(final)}")
    
    # Visualize
    QuantumVisualizer.plot_bloch_sphere(final, title="Hadamard Gate Output")
    QuantumVisualizer.plot_state_comparison(initial, final, 
                                           labels=("Initial", "After Hadamard"))
    
    print("\n✓ Simulation complete!")


def force_desired_state():
    """Optimizes circuit to force a specific state."""
    print("\n" + "=" * 60)
    print("STATE MANIPULATION - FORCE DESIRED STATE")
    print("=" * 60)
    
    print("\nPredefined target states:")
    print("1. |+⟩ (equal superposition)")
    print("2. |-⟩ (minus state)")
    print("3. |i+⟩ (complex phase)")
    print("4. Custom arbitrary state")
    
    choice = input("\nSelect target state (1-4): ").strip()
    
    targets = {
        '1': np.array([1, 1]) / np.sqrt(2),
        '2': np.array([1, -1]) / np.sqrt(2),
        '3': np.array([1, 1j]) / np.sqrt(2),
        '4': None
    }
    
    if choice == '4':
        try:
            real1 = float(input("Enter real part of amplitude 0: "))
            imag1 = float(input("Enter imaginary part of amplitude 0: "))
            real2 = float(input("Enter real part of amplitude 1: "))
            imag2 = float(input("Enter imaginary part of amplitude 1: "))
            target = np.array([real1 + 1j*imag1, real2 + 1j*imag2])
            target = target / np.linalg.norm(target)
        except ValueError:
            print("Invalid input. Using default |+⟩ state.")
            target = np.array([1, 1]) / np.sqrt(2)
    else:
        target = targets.get(choice, np.array([1, 1]) / np.sqrt(2))
    
    print(f"\nTarget state: {target}")
    
    # Optimize
    optimizer = StateOptimizer(target, n_qubits=1, n_layers=2)
    print("\nOptimizing circuit...")
    results = optimizer.optimize_scipy(method='COBYLA')
    
    print(f"\n✓ Optimization complete!")
    print(f"  Achieved fidelity: {results['fidelity']:.6f}")
    print(f"  Final cost: {results['cost']:.6e}")
    print(f"  Iterations: {results['n_iterations']}")
    
    # Visualize
    QuantumVisualizer.plot_optimization_history(results['history'])
    QuantumVisualizer.plot_state_comparison(target, results['optimal_state'],
                                           labels=("Target", "Achieved"))
    QuantumVisualizer.plot_bloch_sphere(results['optimal_state'],
                                       title="Optimized State")


def custom_optimization():
    """Custom target state optimization with advanced settings."""
    print("\n" + "=" * 60)
    print("CUSTOM TARGET STATE OPTIMIZATION")
    print("=" * 60)
    
    try:
        n_layers = int(input("\nNumber of variational layers (1-5): "))
        n_layers = max(1, min(5, n_layers))
    except ValueError:
        n_layers = 2
        print(f"Using default: {n_layers} layers")
    
    print("\nOptimization methods:")
    print("1. COBYLA")
    print("2. Nelder-Mead")
    print("3. Adam")
    print("4. Direct Decomposition")
    
    method_choice = input("Select method (1-4): ").strip()
    methods = {'1': 'COBYLA', '2': 'NELDER-MEAD', '3': 'adam', '4': 'direct'}
    method = methods.get(method_choice, 'COBYLA')
    
    # Random target state
    target = np.random.randn(2) + 1j * np.random.randn(2)
    target = target / np.linalg.norm(target)
    
    print(f"\nRandom target state: {target}")
    print(f"Using {method} with {n_layers} layers")
    
    optimizer = StateOptimizer(target, n_qubits=1, n_layers=n_layers)
    
    print("\nOptimizing...")
    if method == 'direct':
        results = optimizer.optimize_direct_decomposition()
    elif method == 'adam':
        results = optimizer.optimize_adam()
    else:
        results = optimizer.optimize_scipy(method=method)
    
    print(f"\n✓ Optimization complete!")
    print(f"  Fidelity: {results['fidelity']:.6f}")
    print(f"  Iterations: {results['n_iterations']}")
    
    QuantumVisualizer.plot_optimization_history(results['history'])


def compare_algorithms():
    """Compares different optimization algorithms."""
    print("\n" + "=" * 60)
    print("OPTIMIZATION ALGORITHM COMPARISON")
    print("=" * 60)
    
    target = np.array([0.6 + 0.2j, 0.7 - 0.3j])
    target = target / np.linalg.norm(target)
    
    print(f"\nTarget state: {target}")
    print("\nComparing algorithms...")
    
    algorithms = [
        ('Direct Decomposition', 'direct'),
        ('COBYLA', 'COBYLA'),
        ('Nelder-Mead', 'NELDER-MEAD')
    ]
    
    results = []
    
    for name, method in algorithms:
        print(f"\n  Running {name}...")
        optimizer = StateOptimizer(target, n_qubits=1, n_layers=2)
        
        if method == 'direct':
            result = optimizer.optimize_direct_decomposition()
        else:
            result = optimizer.optimize_scipy(method=method)
        
        results.append((name, result))
        print(f"    Fidelity: {result['fidelity']:.6f}, Iterations: {result['n_iterations']}")
    
    print("\n" + "-" * 60)
    print("COMPARISON SUMMARY")
    print("-" * 60)
    for name, result in results:
        print(f"{name:20s}: Fidelity = {result['fidelity']:.6f}, "
              f"Iterations = {result['n_iterations']:4d}")
    
    # Visualize best result
    best = max(results, key=lambda x: x[1]['fidelity'])
    print(f"\n✓ Best: {best[0]}")
    QuantumVisualizer.plot_optimization_history(best[1]['history'])


def bell_state_generation():
    """Generates and analyzes Bell states."""
    print("\n" + "=" * 60)
    print("BELL STATE GENERATION")
    print("=" * 60)
    
    builder = QuantumCircuitBuilder(n_qubits=2)
    builder.create_bell_state(0, 1)
    
    state = builder.get_statevector()
    
    print(f"\nBell state |Φ+⟩: {state}")
    print(f"Probability distribution: {QuantumStateAnalyzer.probability_distribution(state)}")
    
    # Calculate entanglement
    concurrence = QuantumStateAnalyzer.concurrence(state)
    print(f"\nConcurrence (entanglement measure): {concurrence:.6f}")
    
    if concurrence > 0.99:
        print("✓ State is maximally entangled!")
    
    QuantumVisualizer.plot_state_distribution(state, title="Bell State Distribution")


def parameter_landscape():
    """Analyzes optimization parameter landscape."""
    print("\n" + "=" * 60)
    print("PARAMETER LANDSCAPE ANALYSIS")
    print("=" * 60)
    
    target = np.array([1, 1j]) / np.sqrt(2)
    n_runs = 15
    
    print(f"\nTarget state: {target}")
    print(f"Running {n_runs} optimizations with random initializations...\n")
    
    results = []
    
    for i in range(n_runs):
        optimizer = StateOptimizer(target, n_qubits=1, n_layers=2)
        initial_params = np.random.uniform(-np.pi, np.pi, len(optimizer.parameters))
        
        result = optimizer.optimize_scipy(method='COBYLA', initial_params=initial_params)
        results.append(result)
        
        print(f"  Run {i+1:2d}/{n_runs}: Fidelity = {result['fidelity']:.6f}")
    
    fidelities = [r['fidelity'] for r in results]
    
    print("\n" + "-" * 60)
    print("LANDSCAPE STATISTICS")
    print("-" * 60)
    print(f"Mean fidelity:    {np.mean(fidelities):.6f}")
    print(f"Std deviation:    {np.std(fidelities):.6f}")
    print(f"Min fidelity:     {np.min(fidelities):.6f}")
    print(f"Max fidelity:     {np.max(fidelities):.6f}")
    print(f"Success rate (>0.99): {sum(f > 0.99 for f in fidelities)/n_runs*100:.1f}%")


def run_all_examples():
    """Runs all example modules."""
    print("\n" + "=" * 60)
    print("RUNNING ALL EXAMPLES")
    print("=" * 60)
    
    print("\nThis will run:")
    print("  1. Basic Hadamard example")
    print("  2. State manipulation example")
    print("  3. Advanced optimization example")
    
    confirm = input("\nProceed? (y/n): ").strip().lower()
    
    if confirm == 'y':
        print("\n" + ">" * 60)
        try:
            from examples.basic_hadamard import main as basic_main
            basic_main()
        except Exception as e:
            print(f"Error running basic example: {e}")
        
        print("\n" + ">" * 60)
        try:
            from examples.state_manipulation import main as manip_main
            manip_main()
        except Exception as e:
            print(f"Error running manipulation example: {e}")
        
        print("\n" + ">" * 60)
        try:
            from examples.advanced_optimization import main as adv_main
            adv_main()
        except Exception as e:
            print(f"Error running advanced example: {e}")
        
        print("\n✓ All examples completed!")
    else:
        print("Cancelled.")


def interactive_designer():
    """Interactive quantum state designer."""
    print("\n" + "=" * 60)
    print("INTERACTIVE QUANTUM STATE DESIGNER")
    print("=" * 60)
    
    print("\nDesign your quantum state using Bloch sphere coordinates:")
    
    try:
        theta = float(input("Enter θ (0 to π): "))
        phi = float(input("Enter φ (0 to 2π): "))
        
        # Create state from Bloch angles
        target = np.array([
            np.cos(theta / 2),
            np.exp(1j * phi) * np.sin(theta / 2)
        ])
        
        print(f"\nDesigned state: {target}")
        x, y, z = QuantumStateAnalyzer.bloch_coordinates(target)
        print(f"Bloch coordinates: X={x:.4f}, Y={y:.4f}, Z={z:.4f}")
        
        # Optimize
        print("\nOptimizing circuit to achieve this state...")
        optimizer = StateOptimizer(target, n_qubits=1, n_layers=2)
        results = optimizer.optimize_scipy(method='COBYLA')
        
        print(f"\n✓ Circuit optimized!")
        print(f"  Fidelity: {results['fidelity']:.6f}")
        
        # Visualize
        QuantumVisualizer.plot_bloch_sphere(results['optimal_state'],
                                           title="Your Designed State")
        
        # Show circuit
        circuit = optimizer.get_optimized_circuit()
        print("\nOptimized Circuit:")
        print(circuit.draw(output='text'))
        
    except ValueError as e:
        print(f"Invalid input: {e}")


def main():
    """Main program loop."""
    print_banner()
    
    while True:
        print_menu()
        choice = input("Enter your choice (0-8): ").strip()
        
        if choice == '0':
            print("\nExiting Quantum State Manipulation System.")
            print("Thank you for using the system!")
            sys.exit(0)
        elif choice == '1':
            hadamard_simulation()
        elif choice == '2':
            force_desired_state()
        elif choice == '3':
            custom_optimization()
        elif choice == '4':
            compare_algorithms()
        elif choice == '5':
            bell_state_generation()
        elif choice == '6':
            parameter_landscape()
        elif choice == '7':
            run_all_examples()
        elif choice == '8':
            interactive_designer()
        else:
            print("\n✗ Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)