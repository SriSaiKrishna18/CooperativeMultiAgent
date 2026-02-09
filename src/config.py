"""
Configuration management for the Cooperative Multi-Agent System.
Provides centralized settings for environment, agents, and simulation parameters.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional
import json


class PathfindingAlgorithm(Enum):
    """Available pathfinding algorithms."""
    BFS = "bfs"
    ASTAR = "astar"
    DIJKSTRA = "dijkstra"


class AgentRole(Enum):
    """Agent roles for specialized behavior."""
    EXPLORER = "explorer"      # Prioritizes discovering new cells
    COLLECTOR = "collector"    # Prioritizes collecting items
    COORDINATOR = "coordinator"  # Balances exploration and collection


class AgentStrategy(Enum):
    """Behavioral strategies for agents."""
    GREEDY = "greedy"           # Always go for nearest item
    BALANCED = "balanced"       # Balance between exploration and collection
    COOPERATIVE = "cooperative"  # Prioritize team efficiency over individual


class AllocationMethod(Enum):
    """Task allocation methods for cooperation."""
    SIMPLE_CLAIM = "simple_claim"   # First-come-first-served
    AUCTION = "auction"             # Bid-based allocation
    PRIORITY = "priority"           # Priority-based allocation
    ZONE_BASED = "zone_based"       # Territorial division


@dataclass
class GridConfig:
    """Configuration for the grid environment."""
    rows: int = 10
    cols: int = 10
    obstacle_density: float = 0.15  # Percentage of grid filled with obstacles
    item_density: float = 0.10      # Percentage of grid filled with items
    seed: Optional[int] = None      # Random seed for reproducibility
    
    # Predefined grid layouts
    use_predefined: bool = False
    predefined_layout: Optional[List[List[int]]] = None


@dataclass
class AgentConfig:
    """Configuration for individual agents."""
    num_agents: int = 3
    start_positions: Optional[List[Tuple[int, int]]] = None
    default_algorithm: PathfindingAlgorithm = PathfindingAlgorithm.ASTAR
    default_role: AgentRole = AgentRole.COLLECTOR
    default_strategy: AgentStrategy = AgentStrategy.COOPERATIVE
    vision_radius: int = -1  # -1 means full visibility
    max_energy: int = 100    # Energy constraint (-1 for unlimited)
    energy_per_move: int = 1


@dataclass
class CooperationConfig:
    """Configuration for cooperation mechanisms."""
    allocation_method: AllocationMethod = AllocationMethod.AUCTION
    enable_communication: bool = True
    message_delay: int = 0  # Simulation steps for message delivery
    enable_negotiation: bool = True
    zone_assignment: bool = False


@dataclass
class SimulationConfig:
    """Configuration for simulation execution."""
    max_steps: int = 100
    verbose: bool = True
    generate_animations: bool = True
    save_visualizations: bool = True
    output_directory: str = "images"
    logging_level: str = "INFO"
    benchmark_mode: bool = False  # Run multiple algorithms for comparison


@dataclass
class VisualizationConfig:
    """Configuration for visualization outputs."""
    figure_dpi: int = 300
    animation_fps: int = 2
    show_heatmap: bool = True
    show_paths: bool = True
    show_statistics: bool = True
    color_scheme: str = "default"


@dataclass
class Config:
    """Main configuration container."""
    grid: GridConfig = field(default_factory=GridConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    cooperation: CooperationConfig = field(default_factory=CooperationConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: dict) -> 'Config':
        """Create config from dictionary."""
        config = cls()
        
        if 'grid' in data:
            for key, value in data['grid'].items():
                if hasattr(config.grid, key):
                    setattr(config.grid, key, value)
        
        if 'agent' in data:
            for key, value in data['agent'].items():
                if key == 'default_algorithm':
                    value = PathfindingAlgorithm(value)
                elif key == 'default_role':
                    value = AgentRole(value)
                elif key == 'default_strategy':
                    value = AgentStrategy(value)
                if hasattr(config.agent, key):
                    setattr(config.agent, key, value)
        
        if 'cooperation' in data:
            for key, value in data['cooperation'].items():
                if key == 'allocation_method':
                    value = AllocationMethod(value)
                if hasattr(config.cooperation, key):
                    setattr(config.cooperation, key, value)
        
        if 'simulation' in data:
            for key, value in data['simulation'].items():
                if hasattr(config.simulation, key):
                    setattr(config.simulation, key, value)
        
        if 'visualization' in data:
            for key, value in data['visualization'].items():
                if hasattr(config.visualization, key):
                    setattr(config.visualization, key, value)
        
        return config
    
    def to_json(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        data = {
            'grid': {
                'rows': self.grid.rows,
                'cols': self.grid.cols,
                'obstacle_density': self.grid.obstacle_density,
                'item_density': self.grid.item_density,
                'seed': self.grid.seed,
                'use_predefined': self.grid.use_predefined,
            },
            'agent': {
                'num_agents': self.agent.num_agents,
                'default_algorithm': self.agent.default_algorithm.value,
                'default_role': self.agent.default_role.value,
                'default_strategy': self.agent.default_strategy.value,
                'vision_radius': self.agent.vision_radius,
                'max_energy': self.agent.max_energy,
                'energy_per_move': self.agent.energy_per_move,
            },
            'cooperation': {
                'allocation_method': self.cooperation.allocation_method.value,
                'enable_communication': self.cooperation.enable_communication,
                'message_delay': self.cooperation.message_delay,
                'enable_negotiation': self.cooperation.enable_negotiation,
                'zone_assignment': self.cooperation.zone_assignment,
            },
            'simulation': {
                'max_steps': self.simulation.max_steps,
                'verbose': self.simulation.verbose,
                'generate_animations': self.simulation.generate_animations,
                'save_visualizations': self.simulation.save_visualizations,
                'output_directory': self.simulation.output_directory,
                'logging_level': self.simulation.logging_level,
                'benchmark_mode': self.simulation.benchmark_mode,
            },
            'visualization': {
                'figure_dpi': self.visualization.figure_dpi,
                'animation_fps': self.visualization.animation_fps,
                'show_heatmap': self.visualization.show_heatmap,
                'show_paths': self.visualization.show_paths,
                'show_statistics': self.visualization.show_statistics,
                'color_scheme': self.visualization.color_scheme,
            }
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# Default configurations for common scenarios
DEFAULT_CONFIG = Config()

SMALL_GRID_CONFIG = Config(
    grid=GridConfig(rows=5, cols=5, obstacle_density=0.2, item_density=0.12)
)

LARGE_GRID_CONFIG = Config(
    grid=GridConfig(rows=20, cols=20, obstacle_density=0.15, item_density=0.08),
    agent=AgentConfig(num_agents=5),
    simulation=SimulationConfig(max_steps=200)
)

BENCHMARK_CONFIG = Config(
    simulation=SimulationConfig(benchmark_mode=True, verbose=False),
    visualization=VisualizationConfig(show_statistics=True)
)
