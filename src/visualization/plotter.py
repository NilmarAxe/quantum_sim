import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional

from src.quantum.state import QuantumStateAnalyzer
from config.settings import QuantumConfig


class QuantumVisualizer:
    """Visualizes quantum states and optimization results."""
    
    @staticmethod
    def plot_bloch_sphere(state: np.ndarray, 
                         title: str = "Bloch Sphere Representation",
                         save_path: Optional[str] = None):
        """
        Plots state on Bloch sphere.
        
        Args:
            state: Single qubit state
            title: Plot title
            save_path: Path to save figure
        """
        if len(state) != 2:
            raise ValueError("Bloch sphere visualization requires single qubit state")
        
        x, y, z = QuantumStateAnalyzer.bloch_coordinates(state)
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(xs, ys, zs, alpha=0.1, color='cyan')
        
        # Draw axes
        ax.plot([0, 0], [0, 0], [-1.2, 1.2], 'k-', linewidth=2)
        ax.plot([0, 0], [-1.2, 1.2], [0, 0], 'k-', linewidth=2)
        ax.plot([-1.2, 1.2], [0, 0], [0, 0], 'k-', linewidth=2)
        
        # Labels
        ax.text(0, 0, 1.3, '|0⟩', fontsize=14, ha='center')
        ax.text(0, 0, -1.3, '|1⟩', fontsize=14, ha='center')
        ax.text(1.3, 0, 0, '|+⟩', fontsize=14, ha='center')
        ax.text(0, 1.3, 0, '|+i⟩', fontsize=14, ha='center')
        
        # Plot state vector
        ax.quiver(0, 0, 0, x, y, z, arrow_length_ratio=0.1, 
                 color='red', linewidth=3, label='State')
        
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title, fontsize=16)
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=QuantumConfig.DPI, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_state_distribution(state: np.ndarray,
                               title: str = "State Probability Distribution",
                               save_path: Optional[str] = None):
        """
        Plots probability distribution of state.
        
        Args:
            state: Quantum state
            title: Plot title
            save_path: Path to save figure
        """
        probs = QuantumStateAnalyzer.probability_distribution(state)
        
        labels = list(probs.keys())
        values = list(probs.values())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar plot
        colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
        ax1.bar(labels, values, color=colors)
        ax1.set_xlabel('Basis State', fontsize=12)
        ax1.set_ylabel('Probability', fontsize=12)
        ax1.set_title('Measurement Probabilities', fontsize=14)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Amplitudes (real and imaginary)
        state_normalized = state / np.linalg.norm(state)
        indices = np.arange(len(state_normalized))
        width = 0.35
        
        ax2.bar(indices - width/2, np.real(state_normalized), width, 
               label='Real', alpha=0.8, color='blue')
        ax2.bar(indices + width/2, np.imag(state_normalized), width,
               label='Imaginary', alpha=0.8, color='red')
        ax2.set_xlabel('Amplitude Index', fontsize=12)
        ax2.set_ylabel('Amplitude Value', fontsize=12)
        ax2.set_title('State Amplitudes', fontsize=14)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.suptitle(title, fontsize=16, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=QuantumConfig.DPI, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_optimization_history(history: Dict,
                                  title: str = "Optimization History",
                                  save_path: Optional[str] = None):
        """
        Plots optimization convergence history.
        
        Args:
            history: Optimization history dictionary
            title: Plot title
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        iterations = range(len(history['cost']))
        
        # Cost evolution
        axes[0, 0].plot(iterations, history['cost'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Iteration', fontsize=12)
        axes[0, 0].set_ylabel('Cost', fontsize=12)
        axes[0, 0].set_title('Cost Function Evolution', fontsize=14)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        # Fidelity evolution
        axes[0, 1].plot(iterations, history['fidelity'], 'g-', linewidth=2)
        axes[0, 1].axhline(y=QuantumConfig.FIDELITY_THRESHOLD, color='r', 
                          linestyle='--', label='Threshold')
        axes[0, 1].set_xlabel('Iteration', fontsize=12)
        axes[0, 1].set_ylabel('Fidelity', fontsize=12)
        axes[0, 1].set_title('State Fidelity Evolution', fontsize=14)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        axes[0, 1].set_ylim([0, 1.05])
        
        # Parameter evolution
        if history['parameters']:
            params_array = np.array(history['parameters'])
            for i in range(min(5, params_array.shape[1])):
                axes[1, 0].plot(iterations, params_array[:, i], 
                              label=f'θ_{i}', alpha=0.7)
            axes[1, 0].set_xlabel('Iteration', fontsize=12)
            axes[1, 0].set_ylabel('Parameter Value', fontsize=12)
            axes[1, 0].set_title('Parameter Evolution (first 5)', fontsize=14)
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
        
        # Gradient norm
        if history['gradients']:
            grad_norms = [np.linalg.norm(g) for g in history['gradients']]
            axes[1, 1].plot(range(len(grad_norms)), grad_norms, 'r-', linewidth=2)
            axes[1, 1].set_xlabel('Iteration', fontsize=12)
            axes[1, 1].set_ylabel('Gradient Norm', fontsize=12)
            axes[1, 1].set_title('Gradient Magnitude', fontsize=14)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_yscale('log')
        
        plt.suptitle(title, fontsize=16, y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=QuantumConfig.DPI, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_state_comparison(state1: np.ndarray, 
                             state2: np.ndarray,
                             labels: tuple = ("Initial", "Target"),
                             title: str = "State Comparison",
                             save_path: Optional[str] = None):
        """
        Compares two quantum states.
        
        Args:
            state1: First state
            state2: Second state
            labels: Labels for states
            title: Plot title
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Probability distributions
        probs1 = QuantumStateAnalyzer.probability_distribution(state1)
        probs2 = QuantumStateAnalyzer.probability_distribution(state2)
        
        all_keys = sorted(set(probs1.keys()) | set(probs2.keys()))
        values1 = [probs1.get(k, 0) for k in all_keys]
        values2 = [probs2.get(k, 0) for k in all_keys]
        
        x = np.arange(len(all_keys))
        width = 0.35
        
        axes[0].bar(x - width/2, values1, width, label=labels[0], alpha=0.8)
        axes[0].bar(x + width/2, values2, width, label=labels[1], alpha=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(all_keys, rotation=45)
        axes[0].set_xlabel('Basis State', fontsize=12)
        axes[0].set_ylabel('Probability', fontsize=12)
        axes[0].set_title('Probability Distributions', fontsize=14)
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Amplitude comparison
        n = max(len(state1), len(state2))
        indices = np.arange(n)
        
        state1_padded = np.pad(state1, (0, n - len(state1)))
        state2_padded = np.pad(state2, (0, n - len(state2)))
        
        axes[1].plot(indices, np.abs(state1_padded), 'o-', 
                    label=f'{labels[0]} |amplitude|', markersize=8)
        axes[1].plot(indices, np.abs(state2_padded), 's-', 
                    label=f'{labels[1]} |amplitude|', markersize=8)
        axes[1].set_xlabel('Index', fontsize=12)
        axes[1].set_ylabel('Amplitude Magnitude', fontsize=12)
        axes[1].set_title('Amplitude Magnitudes', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Phase comparison
        phases1 = np.angle(state1_padded)
        phases2 = np.angle(state2_padded)
        
        axes[2].plot(indices, phases1, 'o-', label=f'{labels[0]} phase', markersize=8)
        axes[2].plot(indices, phases2, 's-', label=f'{labels[1]} phase', markersize=8)
        axes[2].set_xlabel('Index', fontsize=12)
        axes[2].set_ylabel('Phase (radians)', fontsize=12)
        axes[2].set_title('Phase Comparison', fontsize=14)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        fidelity = QuantumStateAnalyzer.fidelity(state1, state2)
        plt.suptitle(f'{title} (Fidelity: {fidelity:.4f})', fontsize=16, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=QuantumConfig.DPI, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_fidelity_landscape(optimizer_results: List[Dict],
                               param_indices: tuple = (0, 1),
                               title: str = "Fidelity Landscape",
                               save_path: Optional[str] = None):
        """
        Plots 2D fidelity landscape for parameter space exploration.
        
        Args:
            optimizer_results: List of optimization results
            param_indices: Which parameters to plot
            title: Plot title
            save_path: Path to save figure
        """
        params_list = [r['optimal_parameters'] for r in optimizer_results]
        fidelities = [r['fidelity'] for r in optimizer_results]
        
        params_array = np.array(params_list)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        x = params_array[:, param_indices[0]]
        y = params_array[:, param_indices[1]]
        z = np.array(fidelities)
        
        surf = ax.plot_trisurf(x, y, z, cmap=cm.viridis, alpha=0.8)
        
        ax.set_xlabel(f'Parameter {param_indices[0]}', fontsize=12)
        ax.set_ylabel(f'Parameter {param_indices[1]}', fontsize=12)
        ax.set_zlabel('Fidelity', fontsize=12)
        ax.set_title(title, fontsize=16)
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        if save_path:
            plt.savefig(save_path, dpi=QuantumConfig.DPI, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()