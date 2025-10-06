import numpy as np
import time
from src.quantum.state import QuantumStateAnalyzer
from src.optimization.optimizer import StateOptimizer
from src.optimization.cost_functions import CostFunctions
from src.visualization.plotter import QuantumVisualizer


def compare_optimizers(target_state: np.ndarray, n_layers: int = 2):
    """
    Compares different optimization algorithms.
    
    Args:
        target_state: Target quantum state
        n_layers: Number of variational layers
    """
    print(f"\n{'='*60}")
    print("OPTIMIZER COMPARISON")
    print(f"{'='*60}")
    
    optimizers_config = [
        ('Direct Decomposition', 'direct'),
        ('COBYLA', 'cobyla'),
        ('Nelder-Mead', 'nelder_mead'),
        ('Adam', 'adam')
    ]
    
    results = []
    
    for name, method in optimizers_config:
        print(f"\n--- {name} ---")
        
        optimizer = StateOptimizer(target_state, n_qubits=1, n_layers=n_layers)
        
        start_time = time.time()
        
        if method == 'direct':
            result = optimizer.optimize_direct_decomposition()
        elif method == 'adam':
            result = optimizer.optimize_adam(max_iterations=100)
        else:
            result = optimizer.optimize_scipy(method=method.upper().replace('_', '-'))
        
        elapsed_time = time.time() - start_time
        
        print(f"Fidelity: {result['fidelity']:.6f}")
        print(f"Cost: {result['cost']:.6e}")
        print(f"Iterations: {result['n_iterations']}")
        print(f"Time: {elapsed_time:.3f} seconds")
        
        result['method'] = name
        result['time'] = elapsed_time
        results.append(result)
    
    return results


def explore_parameter_landscape():
    """Explores the optimization landscape."""
    print(f"\n{'='*60}")
    print("PARAMETER LANDSCAPE EXPLORATION")
    print(f"{'='*60}")
    
    target = np.array([1, 1j]) / np.sqrt(2)
    
    # Run multiple optimizations with random initializations
    n_runs = 20
    results = []
    
    print(f"\nRunning {n_runs} optimizations with random initializations...")
    
    for i in range(n_runs):
        optimizer = StateOptimizer(target, n_qubits=1, n_layers=2)
        
        # Random initialization
        initial_params = np.random.uniform(-np.pi, np.pi, len(optimizer.parameters))
        
        result = optimizer.optimize_scipy(
            method='COBYLA',
            initial_params=initial_params
        )
        
        results.append(result)
        
        if (i + 1) % 5 == 0:
            print(f"Completed {i + 1}/{n_runs} runs")
    
    # Analyze results
    fidelities = [r['fidelity'] for r in results]
    
    print(f"\n--- Landscape Analysis ---")
    print(f"Mean fidelity: {np.mean(fidelities):.6f}")
    print(f"Std fidelity: {np.std(fidelities):.6f}")
    print(f"Min fidelity: {np.min(fidelities):.6f}")
    print(f"Max fidelity: {np.max(fidelities):.6f}")
    print(f"Success rate (fidelity > 0.99): {sum(f > 0.99 for f in fidelities) / n_runs * 100:.1f}%")
    
    return results


def multi_layer_analysis(target_state: np.ndarray):
    """Analyzes effect of increasing circuit depth."""
    print(f"\n{'='*60}")
    print("MULTI-LAYER DEPTH ANALYSIS")
    print(f"{'='*60}")
    
    layer_counts = [1, 2, 3, 4, 5]
    results_by_depth = {}
    
    for n_layers in layer_counts:
        print(f"\n--- Testing {n_layers} layer(s) ---")
        
        optimizer = StateOptimizer(target_state, n_qubits=1, n_layers=n_layers)
        result = optimizer.optimize_scipy(method='COBYLA')
        
        results_by_depth[n_layers] = result
        
        print(f"Fidelity: {result['fidelity']:.6f}")
        print(f"Iterations: {result['n_iterations']}")
        print(f"Parameters: {len(result['optimal_parameters'])}")
    
    # Summary
    print(f"\n--- Depth Analysis Summary ---")
    for n_layers, result in results_by_depth.items():
        print(f"{n_layers} layer(s): Fidelity = {result['fidelity']:.6f}, "
              f"Iterations = {result['n_iterations']}, "
              f"Params = {len(result['optimal_parameters'])}")
    
    return results_by_depth


def complex_target_optimization():
    """Optimizes for complex target states with specific properties."""
    print(f"\n{'='*60}")
    print("COMPLEX TARGET STATE OPTIMIZATION")
    print(f"{'='*60}")
    
    # Create target with specific Bloch coordinates
    theta_target = np.pi / 3  # 60 degrees from north pole
    phi_target = np.pi / 4    # 45 degrees in xy-plane
    
    target = np.array([
        np.cos(theta_target / 2),
        np.exp(1j * phi_target) * np.sin(theta_target / 2)
    ])
    
    print(f"\nTarget Bloch angles: θ = {theta_target:.4f}, φ = {phi_target:.4f}")
    x, y, z = QuantumStateAnalyzer.bloch_coordinates(target)
    print(f"Target Bloch coordinates: X = {x:.4f}, Y = {y:.4f}, Z = {z:.4f}")
    
    # Optimize with multiple cost functions
    print(f"\n--- Using Infidelity Cost ---")
    optimizer1 = StateOptimizer(target, n_qubits=1, n_layers=3)
    results1 = optimizer1.optimize_scipy(method='COBYLA')
    print(f"Fidelity: {results1['fidelity']:.6f}")
    print(f"Iterations: {results1['n_iterations']}")
    
    # Use Bloch coordinate cost
    print(f"\n--- Using Bloch Coordinate Cost ---")
    optimizer2 = StateOptimizer(target, n_qubits=1, n_layers=3)
    optimizer2.cost_function = CostFunctions.bloch_vector_cost(x, y, z)
    results2 = optimizer2.optimize_scipy(method='COBYLA')
    print(f"Fidelity: {results2['fidelity']:.6f}")
    print(f"Iterations: {results2['n_iterations']}")
    
    # Visualize both results
    QuantumVisualizer.plot_state_comparison(
        results1['optimal_state'],
        results2['optimal_state'],
        labels=("Infidelity Cost", "Bloch Cost"),
        title="Comparison of Cost Functions"
    )
    
    return results1, results2


def convergence_analysis():
    """Analyzes convergence properties of different optimizers."""
    print(f"\n{'='*60}")
    print("CONVERGENCE ANALYSIS")
    print(f"{'='*60}")
    
    target = np.array([0.8, 0.6], dtype=complex)
    target = target / np.linalg.norm(target)
    
    print(f"\nTarget state: {target}")
    
    # Test different layer counts
    layer_configs = [1, 2, 3, 4]
    
    for n_layers in layer_configs:
        print(f"\n--- {n_layers} Layer(s) ---")
        
        optimizer = StateOptimizer(target, n_qubits=1, n_layers=n_layers)
        result = optimizer.optimize_scipy(method='COBYLA')
        
        # Analyze convergence rate
        if len(result['history']['fidelity']) > 10:
            initial_fid = result['history']['fidelity'][0]
            mid_fid = result['history']['fidelity'][len(result['history']['fidelity'])//2]
            final_fid = result['history']['fidelity'][-1]
            
            print(f"Initial fidelity: {initial_fid:.6f}")
            print(f"Mid fidelity: {mid_fid:.6f}")
            print(f"Final fidelity: {final_fid:.6f}")
            print(f"Convergence speed: {(final_fid - initial_fid) / result['n_iterations']:.6f} per iteration")


def robustness_test():
    """Tests robustness to different initial conditions."""
    print(f"\n{'='*60}")
    print("ROBUSTNESS TEST")
    print(f"{'='*60}")
    
    target = np.array([1, 1]) / np.sqrt(2)
    
    print(f"\nTarget state: |+⟩")
    print("Testing with different initial parameter distributions...")
    
    initialization_methods = [
        ('Uniform [-π, π]', lambda n: np.random.uniform(-np.pi, np.pi, n)),
        ('Uniform [-π/2, π/2]', lambda n: np.random.uniform(-np.pi/2, np.pi/2, n)),
        ('Normal (0, 1)', lambda n: np.random.randn(n)),
        ('Zeros', lambda n: np.zeros(n))
    ]
    
    for name, init_fn in initialization_methods:
        print(f"\n--- {name} ---")
        
        successes = 0
        fidelities = []
        
        for _ in range(10):
            optimizer = StateOptimizer(target, n_qubits=1, n_layers=2)
            initial_params = init_fn(len(optimizer.parameters))
            
            result = optimizer.optimize_scipy(method='COBYLA', initial_params=initial_params)
            fidelities.append(result['fidelity'])
            
            if result['fidelity'] > 0.99:
                successes += 1
        
        print(f"Success rate: {successes}/10 ({successes*10}%)")
        print(f"Mean fidelity: {np.mean(fidelities):.6f}")
        print(f"Std fidelity: {np.std(fidelities):.6f}")


def cost_function_comparison():
    """Compares different cost functions."""
    print(f"\n{'='*60}")
    print("COST FUNCTION COMPARISON")
    print(f"{'='*60}")
    
    target = np.array([0.6 + 0.3j, 0.7 - 0.2j])
    target = target / np.linalg.norm(target)
    
    print(f"\nTarget state: {target}")
    
    # Get Bloch coordinates for target
    x_t, y_t, z_t = QuantumStateAnalyzer.bloch_coordinates(target)
    
    cost_configs = [
        ('Infidelity', CostFunctions.infidelity(target)),
        ('Trace Distance', CostFunctions.trace_distance_cost(target)),
        ('Bloch Vector', CostFunctions.bloch_vector_cost(x_t, y_t, z_t)),
        ('Overlap', CostFunctions.overlap_cost(target)),
        ('Hilbert-Schmidt', CostFunctions.hilbert_schmidt_cost(target))
    ]
    
    results = []
    
    for name, cost_fn in cost_configs:
        print(f"\n--- {name} Cost Function ---")
        
        optimizer = StateOptimizer(target, n_qubits=1, n_layers=2)
        optimizer.cost_function = cost_fn
        
        result = optimizer.optimize_scipy(method='COBYLA')
        
        print(f"Fidelity: {result['fidelity']:.6f}")
        print(f"Iterations: {result['n_iterations']}")
        print(f"Final cost: {result['cost']:.6e}")
        
        results.append((name, result))
    
    # Summary
    print(f"\n--- Cost Function Performance Summary ---")
    for name, result in results:
        print(f"{name:20s}: Fidelity = {result['fidelity']:.6f}, Iterations = {result['n_iterations']:4d}")


def optimization_landscape_visualization():
    """Creates visualization of optimization landscape."""
    print(f"\n{'='*60}")
    print("OPTIMIZATION LANDSCAPE VISUALIZATION")
    print(f"{'='*60}")
    
    target = np.array([1, 1j]) / np.sqrt(2)
    
    print(f"\nGenerating landscape data...")
    
    # Sample parameter space
    n_samples = 30
    results = []
    
    for i in range(n_samples):
        optimizer = StateOptimizer(target, n_qubits=1, n_layers=1)
        
        # Random initialization in parameter space
        initial_params = np.random.uniform(-np.pi, np.pi, len(optimizer.parameters))
        result = optimizer.optimize_scipy(method='COBYLA', initial_params=initial_params)
        
        results.append(result)
        
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{n_samples} samples")
    
    print("\nVisualizing parameter landscape...")
    
    if len(results) > 0 and len(results[0]['optimal_parameters']) >= 2:
        try:
            QuantumVisualizer.plot_fidelity_landscape(
                results,
                param_indices=(0, 1),
                title="Fidelity Landscape in Parameter Space"
            )
        except Exception as e:
            print(f"Visualization error: {e}")
            print("Continuing with other analyses...")


def main():
    """Run all advanced optimization examples."""
    
    print("=" * 60)
    print("ADVANCED QUANTUM OPTIMIZATION EXAMPLES")
    print("=" * 60)
    
    # Define a challenging target state
    target_state = np.array([0.6 + 0.2j, 0.7 - 0.3j])
    target_state = target_state / np.linalg.norm(target_state)
    
    print(f"\nMain target state: {target_state}")
    x, y, z = QuantumStateAnalyzer.bloch_coordinates(target_state)
    print(f"Bloch coordinates: X={x:.4f}, Y={y:.4f}, Z={z:.4f}")
    
    # 1. Compare optimizers
    print("\n" + ">"*60)
    print("SECTION 1: OPTIMIZER COMPARISON")
    print(">"*60)
    optimizer_results = compare_optimizers(target_state, n_layers=2)
    
    # Find best optimizer
    best_result = max(optimizer_results, key=lambda r: r['fidelity'])
    print(f"\n*** Best optimizer: {best_result['method']} ***")
    print(f"*** Achieved fidelity: {best_result['fidelity']:.6f} ***")
    
    # Visualize best result
    QuantumVisualizer.plot_optimization_history(
        best_result['history'],
        title=f"Best Optimizer: {best_result['method']}"
    )
    
    # 2. Explore parameter landscape
    print("\n" + ">"*60)
    print("SECTION 2: PARAMETER LANDSCAPE EXPLORATION")
    print(">"*60)
    landscape_results = explore_parameter_landscape()
    
    # 3. Multi-layer analysis
    print("\n" + ">"*60)
    print("SECTION 3: MULTI-LAYER DEPTH ANALYSIS")
    print(">"*60)
    depth_results = multi_layer_analysis(target_state)
    
    # 4. Complex target optimization
    print("\n" + ">"*60)
    print("SECTION 4: COMPLEX TARGET OPTIMIZATION")
    print(">"*60)
    complex_results = complex_target_optimization()
    
    # 5. Convergence analysis
    print("\n" + ">"*60)
    print("SECTION 5: CONVERGENCE ANALYSIS")
    print(">"*60)
    convergence_analysis()
    
    # 6. Robustness test
    print("\n" + ">"*60)
    print("SECTION 6: ROBUSTNESS TEST")
    print(">"*60)
    robustness_test()
    
    # 7. Cost function comparison
    print("\n" + ">"*60)
    print("SECTION 7: COST FUNCTION COMPARISON")
    print(">"*60)
    cost_function_comparison()
    
    # 8. Landscape visualization
    print("\n" + ">"*60)
    print("SECTION 8: LANDSCAPE VISUALIZATION")
    print(">"*60)
    optimization_landscape_visualization()
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    print("\nOptimizer Performance:")
    for result in optimizer_results:
        print(f"  {result['method']:20s}: Fidelity = {result['fidelity']:.6f}, "
              f"Time = {result['time']:.3f}s")
    
    print("\nKey Findings:")
    print("- Direct decomposition provides instant near-perfect results")
    print("- Numerical optimizers achieve high fidelity with proper tuning")
    print("- Multiple layers improve convergence reliability")
    print("- Different cost functions lead to similar final states")
    print("- Random initialization explores diverse parameter landscapes")
    print("- Robustness varies with initialization strategy")
    print("- Convergence speed increases with appropriate circuit depth")
    
    print("\n" + "=" * 60)
    print("Advanced optimization examples completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()