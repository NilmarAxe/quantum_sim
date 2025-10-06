class QuantumConfig:
    """Configuration parameters for quantum operations."""
    
    # Simulation parameters
    SHOTS = 8192
    OPTIMIZATION_SHOTS = 4096
    
    # Optimization parameters
    MAX_ITERATIONS = 200
    CONVERGENCE_THRESHOLD = 1e-6
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    
    # Gate parameters
    GATE_PRECISION = 1e-10
    PHASE_PRECISION = 1e-8
    
    # Visualization settings
    FIGURE_SIZE = (12, 8)
    DPI = 100
    STYLE = 'seaborn-v0_8-darkgrid'
    
    # State manipulation
    DEFAULT_TARGET_STATE = [1/2**0.5, 1/2**0.5]  # |+> state
    FIDELITY_THRESHOLD = 0.99
    
    # Optimization methods
    AVAILABLE_OPTIMIZERS = ['gradient_descent', 'adam', 'nelder_mead', 'cobyla']
    DEFAULT_OPTIMIZER = 'adam'
    
    # Logging
    VERBOSE = True
    LOG_INTERVAL = 10