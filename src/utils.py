"""
Enhanced Visualization and Utility Module for Cooperative Multi-Agent System.
Provides animated visualizations, heatmaps, statistical analysis, and reporting.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import logging
import os
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Use Agg backend for non-interactive environments
import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)


# Custom color schemes
COLOR_SCHEMES = {
    'default': {
        'empty': '#E8F5E9',
        'obstacle': '#424242',
        'item': '#FFC107',
        'collected': '#C8E6C9',
        'agents': ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA'],
        'paths': ['#EF5350', '#42A5F5', '#66BB6A', '#FFA726', '#AB47BC'],
        'heatmap': 'YlOrRd'
    },
    'dark': {
        'empty': '#1E1E1E',
        'obstacle': '#000000',
        'item': '#FFD700',
        'collected': '#2E7D32',
        'agents': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
        'paths': ['#FF8A80', '#84FFFF', '#80D8FF', '#B2FF59', '#FFD180'],
        'heatmap': 'plasma'
    },
    'professional': {
        'empty': '#F5F5F5',
        'obstacle': '#263238',
        'item': '#FF9800',
        'collected': '#4CAF50',
        'agents': ['#D32F2F', '#1976D2', '#388E3C', '#F57C00', '#7B1FA2'],
        'paths': ['#EF5350', '#42A5F5', '#66BB6A', '#FFA726', '#AB47BC'],
        'heatmap': 'viridis'
    }
}


@dataclass
class VisualizationFrame:
    """Single frame of animation data."""
    step: int
    grid: np.ndarray
    agent_positions: List[Tuple[int, int]]
    agent_paths: List[List[Tuple[int, int]]]
    items_remaining: int
    collection_counts: List[int]


class Visualizer:
    """
    Enhanced visualization system with:
    - Animated GIF generation
    - Heatmap visualization
    - Comparative analysis charts
    - Statistical dashboards
    """
    
    def __init__(
        self, 
        output_dir: str = "images",
        color_scheme: str = "default",
        dpi: int = 300,
        animation_fps: int = 2
    ):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory for saving visualizations
            color_scheme: Color scheme name
            dpi: Image resolution
            animation_fps: Frames per second for animations
        """
        self.output_dir = output_dir
        self.colors = COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES['default'])
        self.dpi = dpi
        self.animation_fps = animation_fps
        
        # Animation frames
        self.frames: List[VisualizationFrame] = []
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Visualizer initialized: output_dir={output_dir}, scheme={color_scheme}")
    
    def capture_frame(
        self,
        step: int,
        environment: Any,
        agents: List[Any]
    ) -> None:
        """Capture a frame for animation."""
        frame = VisualizationFrame(
            step=step,
            grid=environment.grid.copy(),
            agent_positions=[a.position for a in agents],
            agent_paths=[list(a.path) for a in agents],
            items_remaining=environment.items_remaining(),
            collection_counts=[a.collected_items for a in agents]
        )
        self.frames.append(frame)
    
    def visualize_grid(
        self, 
        environment: Any, 
        agents: List[Any], 
        step_number: int = None,
        show_paths: bool = True,
        filename: str = "grid_visualization.png"
    ) -> str:
        """
        Create comprehensive grid visualization with paths.
        
        Args:
            environment: Environment object
            agents: List of agents
            step_number: Current simulation step
            show_paths: Whether to show agent paths
            filename: Output filename
            
        Returns:
            Path to saved image
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left: Current grid state
        self._draw_grid_state(axes[0], environment, agents, step_number)
        
        # Right: Agent paths
        if show_paths:
            self._draw_agent_paths(axes[1], environment, agents)
        else:
            self._draw_grid_legend(axes[1], agents)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        logger.info(f"Saved grid visualization: {filepath}")
        return filepath
    
    def _draw_grid_state(
        self, 
        ax: plt.Axes, 
        environment: Any, 
        agents: List[Any],
        step_number: int = None
    ) -> None:
        """Draw the current grid state."""
        rows, cols = environment.rows, environment.cols
        
        # Create colored grid
        for i in range(rows):
            for j in range(cols):
                cell = environment.grid[i, j]
                if cell == 1:
                    color = self.colors['obstacle']
                elif cell == 2:
                    color = self.colors['item']
                else:
                    color = self.colors['empty']
                
                rect = patches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    linewidth=0.5, edgecolor='gray',
                    facecolor=color
                )
                ax.add_patch(rect)
        
        # Draw agents
        for idx, agent in enumerate(agents):
            x, y = agent.position
            color = self.colors['agents'][idx % len(self.colors['agents'])]
            circle = patches.Circle(
                (y, x), 0.35,
                facecolor=color, edgecolor='white', linewidth=2
            )
            ax.add_patch(circle)
            ax.text(y, x, str(agent.id), ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white')
        
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.grid(True, linewidth=0.3, alpha=0.5)
        
        title = "Grid State"
        if step_number is not None:
            title += f" - Step {step_number}"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
    
    def _draw_agent_paths(
        self, 
        ax: plt.Axes, 
        environment: Any, 
        agents: List[Any]
    ) -> None:
        """Draw agent movement paths."""
        rows, cols = environment.rows, environment.cols
        
        # Draw obstacles as background
        for i in range(rows):
            for j in range(cols):
                if environment.grid[i, j] == 1:
                    rect = patches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        facecolor=self.colors['obstacle'], alpha=0.3
                    )
                    ax.add_patch(rect)
        
        # Draw paths
        for idx, agent in enumerate(agents):
            if len(agent.path) > 1:
                path_array = np.array(agent.path)
                color = self.colors['paths'][idx % len(self.colors['paths'])]
                ax.plot(
                    path_array[:, 1], path_array[:, 0],
                    color=color, linewidth=2.5, marker='o', markersize=4,
                    label=f'Agent {agent.id} ({len(agent.path)} steps, {agent.collected_items} items)',
                    alpha=0.8
                )
                
                # Mark start and end
                ax.plot(path_array[0, 1], path_array[0, 0], 's',
                       color=color, markersize=10, markeredgecolor='black')
                ax.plot(path_array[-1, 1], path_array[-1, 0], '*',
                       color=color, markersize=15, markeredgecolor='black')
        
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('Agent Movement Paths', fontsize=14, fontweight='bold')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.legend(loc='upper right', fontsize=9)
    
    def _draw_grid_legend(self, ax: plt.Axes, agents: List[Any]) -> None:
        """Draw a legend panel."""
        ax.axis('off')
        
        legend_items = [
            (self.colors['empty'], 'Empty Cell'),
            (self.colors['obstacle'], 'Obstacle'),
            (self.colors['item'], 'Item'),
        ]
        
        for idx, agent in enumerate(agents):
            color = self.colors['agents'][idx % len(self.colors['agents'])]
            legend_items.append((color, f'Agent {agent.id}'))
        
        for i, (color, label) in enumerate(legend_items):
            y = 0.9 - i * 0.1
            ax.add_patch(patches.Rectangle((0.1, y - 0.03), 0.1, 0.06, 
                                          facecolor=color, edgecolor='black'))
            ax.text(0.25, y, label, fontsize=12, va='center')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Legend', fontsize=14, fontweight='bold')
    
    def create_animation(
        self, 
        environment: Any,
        filename: str = "simulation_animation.gif"
    ) -> Optional[str]:
        """
        Create animated GIF of the simulation.
        
        Args:
            environment: Environment object (for grid dimensions)
            filename: Output filename
            
        Returns:
            Path to saved animation, or None if no frames
        """
        if not self.frames:
            logger.warning("No frames captured for animation")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        rows = environment.rows
        cols = environment.cols
        
        def animate(frame_idx):
            ax.clear()
            frame = self.frames[frame_idx]
            
            # Draw grid
            for i in range(rows):
                for j in range(cols):
                    cell = frame.grid[i, j]
                    if cell == 1:
                        color = self.colors['obstacle']
                    elif cell == 2:
                        color = self.colors['item']
                    else:
                        color = self.colors['empty']
                    
                    rect = patches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        linewidth=0.5, edgecolor='gray', facecolor=color
                    )
                    ax.add_patch(rect)
            
            # Draw agents and their trails
            for idx, (pos, path) in enumerate(zip(frame.agent_positions, frame.agent_paths)):
                color = self.colors['agents'][idx % len(self.colors['agents'])]
                
                # Draw path trail
                if len(path) > 1:
                    path_array = np.array(path)
                    ax.plot(path_array[:, 1], path_array[:, 0],
                           color=color, linewidth=1.5, alpha=0.5)
                
                # Draw agent
                x, y = pos
                circle = patches.Circle(
                    (y, x), 0.35,
                    facecolor=color, edgecolor='white', linewidth=2
                )
                ax.add_patch(circle)
                ax.text(y, x, str(idx), ha='center', va='center',
                       fontsize=10, fontweight='bold', color='white')
            
            ax.set_xlim(-0.5, cols - 0.5)
            ax.set_ylim(rows - 0.5, -0.5)
            ax.set_aspect('equal')
            ax.set_title(f'Step {frame.step} | Items: {frame.items_remaining} remaining',
                        fontsize=12, fontweight='bold')
            ax.grid(True, linewidth=0.3, alpha=0.3)
            
            return ax.patches
        
        anim = animation.FuncAnimation(
            fig, animate, frames=len(self.frames),
            interval=1000 // self.animation_fps, blit=False
        )
        
        filepath = os.path.join(self.output_dir, filename)
        anim.save(filepath, writer='pillow', fps=self.animation_fps)
        plt.close()
        
        logger.info(f"Saved animation: {filepath} ({len(self.frames)} frames)")
        return filepath
    
    def plot_heatmap(
        self, 
        environment: Any,
        filename: str = "visit_heatmap.png"
    ) -> str:
        """
        Create heatmap of cell visit frequency.
        
        Args:
            environment: Environment object with visit_counts
            filename: Output filename
            
        Returns:
            Path to saved image
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        heatmap_data = environment.get_heatmap_data()
        
        # Mask obstacles
        masked_data = np.ma.masked_where(environment.grid == 1, heatmap_data)
        
        im = ax.imshow(
            masked_data, 
            cmap=self.colors['heatmap'],
            interpolation='nearest'
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Visit Frequency', fontsize=12)
        
        # Mark obstacles
        for i in range(environment.rows):
            for j in range(environment.cols):
                if environment.grid[i, j] == 1:
                    rect = patches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        facecolor=self.colors['obstacle'], edgecolor='gray'
                    )
                    ax.add_patch(rect)
        
        ax.set_title('Cell Visit Frequency Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_xticks(range(environment.cols))
        ax.set_yticks(range(environment.rows))
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved heatmap: {filepath}")
        return filepath
    
    def plot_collection_progress(
        self, 
        collection_history: Dict[int, List[int]],
        filename: str = "collection_progress.png"
    ) -> str:
        """
        Plot items collected over time for each agent.
        
        Args:
            collection_history: Dict mapping agent_id to list of cumulative counts
            filename: Output filename
            
        Returns:
            Path to saved image
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for agent_id, history in collection_history.items():
            color = self.colors['agents'][agent_id % len(self.colors['agents'])]
            ax.plot(
                history, 
                label=f'Agent {agent_id}',
                marker='o', linewidth=2, markersize=4,
                color=color
            )
        
        # Add total line
        if collection_history:
            total = [sum(h[i] for h in collection_history.values()) 
                    for i in range(len(list(collection_history.values())[0]))]
            ax.plot(total, label='Total', linestyle='--', linewidth=2.5, 
                   color='black', marker='s', markersize=5)
        
        ax.set_xlabel('Simulation Step', fontsize=12)
        ax.set_ylabel('Items Collected', fontsize=12)
        ax.set_title('Item Collection Progress Over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add annotations for final values
        for agent_id, history in collection_history.items():
            if history:
                ax.annotate(
                    f'{history[-1]}',
                    xy=(len(history) - 1, history[-1]),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=10, fontweight='bold'
                )
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved collection progress: {filepath}")
        return filepath
    
    def plot_performance_metrics(
        self, 
        agents: List[Any], 
        total_steps: int, 
        total_items: int,
        filename: str = "performance_metrics.png"
    ) -> str:
        """
        Create comprehensive performance metrics dashboard.
        
        Args:
            agents: List of agents
            total_steps: Total simulation steps
            total_items: Total items in environment
            filename: Output filename
            
        Returns:
            Path to saved image
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        agent_ids = [a.id for a in agents]
        colors = [self.colors['agents'][i % len(self.colors['agents'])] for i in agent_ids]
        
        # 1. Items Collected per Agent
        items = [a.collected_items for a in agents]
        axes[0, 0].bar(agent_ids, items, color=colors, edgecolor='black', linewidth=1.5)
        axes[0, 0].set_title('Items Collected', fontweight='bold', fontsize=12)
        axes[0, 0].set_xlabel('Agent ID')
        axes[0, 0].set_ylabel('Items')
        axes[0, 0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(items):
            axes[0, 0].text(i, v + 0.1, str(v), ha='center', fontweight='bold')
        
        # 2. Total Moves per Agent
        moves = [len(a.path) - 1 for a in agents]
        axes[0, 1].bar(agent_ids, moves, color=colors, edgecolor='black', linewidth=1.5)
        axes[0, 1].set_title('Total Moves', fontweight='bold', fontsize=12)
        axes[0, 1].set_xlabel('Agent ID')
        axes[0, 1].set_ylabel('Moves')
        axes[0, 1].grid(axis='y', alpha=0.3)
        for i, v in enumerate(moves):
            axes[0, 1].text(i, v + 0.1, str(v), ha='center', fontweight='bold')
        
        # 3. Efficiency (Items per Move)
        efficiency = [a.collected_items / max(len(a.path) - 1, 1) for a in agents]
        axes[0, 2].bar(agent_ids, efficiency, color=colors, edgecolor='black', linewidth=1.5)
        axes[0, 2].set_title('Efficiency (Items/Move)', fontweight='bold', fontsize=12)
        axes[0, 2].set_xlabel('Agent ID')
        axes[0, 2].set_ylabel('Efficiency')
        axes[0, 2].grid(axis='y', alpha=0.3)
        for i, v in enumerate(efficiency):
            axes[0, 2].text(i, v + 0.01, f'{v:.2f}', ha='center', fontweight='bold')
        
        # 4. Algorithm & Role Distribution (Pie Chart)
        if hasattr(agents[0], 'algorithm'):
            algo_counts = {}
            for a in agents:
                algo = a.algorithm.value
                algo_counts[algo] = algo_counts.get(algo, 0) + 1
            axes[1, 0].pie(
                algo_counts.values(), 
                labels=algo_counts.keys(),
                autopct='%1.0f%%',
                colors=colors[:len(algo_counts)]
            )
            axes[1, 0].set_title('Algorithm Distribution', fontweight='bold', fontsize=12)
        else:
            axes[1, 0].text(0.5, 0.5, 'BFS Only', ha='center', va='center', fontsize=14)
            axes[1, 0].set_title('Algorithm', fontweight='bold', fontsize=12)
        
        # 5. Cells Explored per Agent
        if hasattr(agents[0], 'cells_explored'):
            explored = [len(a.cells_explored) for a in agents]
        else:
            explored = moves  # Fallback
        axes[1, 1].bar(agent_ids, explored, color=colors, edgecolor='black', linewidth=1.5)
        axes[1, 1].set_title('Cells Explored', fontweight='bold', fontsize=12)
        axes[1, 1].set_xlabel('Agent ID')
        axes[1, 1].set_ylabel('Cells')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        # 6. System-Wide Summary
        total_collected = sum(items)
        total_moves = sum(moves)
        avg_efficiency = total_collected / max(total_moves, 1)
        
        summary_labels = ['Steps', 'Items\nCollected', 'Total\nMoves', 'Avg\nEfficiency']
        summary_values = [total_steps, total_collected, total_moves, avg_efficiency]
        bar_colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
        
        bars = axes[1, 2].bar(summary_labels, summary_values, color=bar_colors, edgecolor='black')
        axes[1, 2].set_title('System Summary', fontweight='bold', fontsize=12)
        axes[1, 2].set_ylabel('Value')
        axes[1, 2].grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, summary_values):
            height = bar.get_height()
            axes[1, 2].text(
                bar.get_x() + bar.get_width() / 2., height,
                f'{val:.2f}' if isinstance(val, float) else str(val),
                ha='center', va='bottom', fontweight='bold'
            )
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved performance metrics: {filepath}")
        return filepath
    
    def plot_algorithm_comparison(
        self,
        comparison_data: Dict[str, Dict[str, float]],
        filename: str = "algorithm_comparison.png"
    ) -> str:
        """
        Create algorithm comparison chart.
        
        Args:
            comparison_data: Dict with algorithm names as keys and metrics dicts as values
            filename: Output filename
            
        Returns:
            Path to saved image
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        algorithms = list(comparison_data.keys())
        metrics = ['steps', 'efficiency', 'exploration']
        titles = ['Steps to Complete', 'Efficiency (Items/Move)', 'Exploration Coverage (%)']
        colors = ['#2196F3', '#4CAF50', '#FF9800']
        
        for ax, metric, title, color in zip(axes, metrics, titles, colors):
            values = [comparison_data[algo].get(metric, 0) for algo in algorithms]
            bars = ax.bar(algorithms, values, color=color, edgecolor='black')
            ax.set_title(title, fontweight='bold')
            ax.set_ylabel('Value')
            ax.grid(axis='y', alpha=0.3)
            
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold'
                )
        
        plt.suptitle('Algorithm Performance Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved algorithm comparison: {filepath}")
        return filepath


# ==================== UTILITY FUNCTIONS ====================

def print_simulation_status(
    step: int, 
    agents: List[Any], 
    environment: Any
) -> None:
    """Print status of all agents and environment."""
    print(f"\n--- Step {step} ---")
    for agent in agents:
        status = agent.get_status()
        target_str = f", Target: {status['current_target']}" if status.get('current_target') else ""
        role_str = f", Role: {status.get('role', 'N/A')}"
        print(f"  Agent {status['id']}: Position {status['position']}, "
              f"Items: {status['collected_items']}{target_str}{role_str}")
    print(f"  Items Remaining: {environment.items_remaining()}")


def generate_summary_report(
    agents: List[Any], 
    total_steps: int, 
    total_items: int,
    shared_knowledge: Any = None
) -> Dict[str, Any]:
    """
    Generate comprehensive simulation summary.
    
    Args:
        agents: List of agents
        total_steps: Total simulation steps
        total_items: Total items in environment
        shared_knowledge: Optional shared knowledge object
        
    Returns:
        Dictionary of summary statistics
    """
    print("\n" + "=" * 70)
    print("  SIMULATION COMPLETE - PERFORMANCE REPORT")
    print("=" * 70)
    
    total_collected = sum(a.collected_items for a in agents)
    total_moves = sum(len(a.path) - 1 for a in agents)
    
    for agent in agents:
        status = agent.get_status()
        efficiency = status['collected_items'] / max(status['path_length'], 1)
        
        print(f"\n🤖 Agent {status['id']}:")
        print(f"   Position: {status['position']}")
        print(f"   Items Collected: {status['collected_items']}")
        print(f"   Total Moves: {status['path_length']}")
        print(f"   Efficiency: {efficiency:.2f} items/move")
        
        if 'algorithm' in status:
            print(f"   Algorithm: {status['algorithm']}")
        if 'role' in status:
            print(f"   Role: {status['role']}")
        if 'cells_explored' in status:
            print(f"   Cells Explored: {status['cells_explored']}")
    
    print(f"\n{'SYSTEM METRICS':^70}")
    print("-" * 70)
    
    collection_rate = (total_collected / total_items) * 100 if total_items > 0 else 0
    avg_efficiency = total_moves / max(total_collected, 1)
    
    print(f"  📊 Total Items: {total_items}")
    print(f"  ✅ Items Collected: {total_collected}")
    print(f"  📈 Collection Rate: {collection_rate:.1f}%")
    print(f"  ⏱️  Total Steps: {total_steps}")
    print(f"  🚶 Total Moves: {total_moves}")
    print(f"  ⚡ Avg Steps/Item: {avg_efficiency:.2f}")
    
    # Efficiency rating
    if avg_efficiency < 5:
        rating = "🌟 Excellent"
    elif avg_efficiency < 10:
        rating = "✅ Good"
    elif avg_efficiency < 20:
        rating = "⚠️ Fair"
    else:
        rating = "❌ Needs Improvement"
    
    print(f"  🏆 Efficiency Rating: {rating}")
    
    if shared_knowledge:
        stats = shared_knowledge.get_statistics()
        print(f"\n{'COOPERATION STATISTICS':^70}")
        print("-" * 70)
        print(f"  📍 Visited Cells: {stats.get('visited_cells', 'N/A')}")
        print(f"  🎯 Claim Conflicts: {stats.get('claim_conflicts', 0)}")
        print(f"  💬 Messages Sent: {stats.get('messages_sent', 0)}")
        print(f"  🏷️  Allocation Method: {stats.get('allocation_method', 'N/A')}")
    
    print("=" * 70)
    
    # Return summary dict
    return {
        'total_steps': total_steps,
        'total_items': total_items,
        'items_collected': total_collected,
        'collection_rate': collection_rate,
        'total_moves': total_moves,
        'avg_efficiency': avg_efficiency,
        'rating': rating,
        'agent_stats': [a.get_status() for a in agents]
    }


def setup_logging(level: str = "INFO", log_file: str = None) -> None:
    """
    Configure logging for the simulation.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs to
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=handlers
    )
    
    logger.info(f"Logging configured: level={level}")


def get_timestamp() -> str:
    """Get current timestamp for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
