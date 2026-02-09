"""
Enhanced Grid Environment for Cooperative Multi-Agent System.
Supports configurable sizes, weighted terrain, and dynamic obstacles.
"""
import numpy as np
from typing import List, Tuple, Optional, Set, Dict
from dataclasses import dataclass, field
import random
import logging

from config import GridConfig, PathfindingAlgorithm


logger = logging.getLogger(__name__)


@dataclass
class Cell:
    """Represents a single cell in the grid."""
    cell_type: int  # 0=empty, 1=obstacle, 2=item
    weight: float = 1.0  # Movement cost for weighted algorithms
    visited_count: int = 0  # Track visit frequency for heatmap
    discovered_by: Optional[int] = None  # Agent ID that first discovered this cell


class CellType:
    """Constants for cell types."""
    EMPTY = 0
    OBSTACLE = 1
    ITEM = 2
    COLLECTED = 3  # Previously had item, now collected


class Environment:
    """
    Enhanced grid environment with support for:
    - Configurable grid sizes (5x5 to 50x50)
    - Weighted terrain for Dijkstra's algorithm
    - Dynamic obstacles
    - Partial observability
    - Visit tracking for heatmaps
    """
    
    # Predefined scenario layouts
    SCENARIOS = {
        'simple': np.array([
            [0, 1, 0, 2, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [2, 0, 1, 1, 0],
            [0, 0, 0, 2, 0]
        ]),
        'maze': np.array([
            [0, 0, 1, 0, 2, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 0, 1, 1, 0, 1, 2],
            [2, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 1, 1, 1, 0, 1, 1],
            [0, 1, 0, 0, 0, 0, 0, 0, 2, 0],
            [2, 1, 0, 0, 1, 1, 0, 0, 0, 0]
        ]),
        'open': np.array([
            [2, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0, 2]
        ]),
        'clustered': np.array([
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 2, 2, 1, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 2, 2, 0],
            [0, 0, 0, 0, 0, 0, 1, 2, 0, 0],
            [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 1, 0, 0, 0, 2, 2, 0],
            [0, 2, 0, 1, 0, 0, 0, 2, 0, 0]
        ])
    }
    
    def __init__(self, config: Optional[GridConfig] = None, grid: Optional[np.ndarray] = None):
        """
        Initialize the environment.
        
        Args:
            config: Grid configuration settings
            grid: Optional predefined grid layout
        """
        self.config = config or GridConfig()
        
        if grid is not None:
            self.grid = np.array(grid)
        elif self.config.use_predefined and self.config.predefined_layout:
            self.grid = np.array(self.config.predefined_layout)
        else:
            self.grid = self._generate_random_grid()
        
        self.rows, self.cols = self.grid.shape
        self.total_items = np.count_nonzero(self.grid == CellType.ITEM)
        self.initial_items = self.total_items
        
        # Weighted terrain (default all 1.0)
        self.weights = np.ones_like(self.grid, dtype=float)
        
        # Visit tracking for heatmap
        self.visit_counts = np.zeros_like(self.grid, dtype=int)
        
        # Item collection history
        self.collection_history: List[Dict] = []
        
        # Dynamic obstacles state
        self.dynamic_obstacles: List[Tuple[int, int]] = []
        
        logger.info(f"Environment initialized: {self.rows}x{self.cols} grid, {self.total_items} items")
    
    def _generate_random_grid(self) -> np.ndarray:
        """Generate a random grid based on configuration."""
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)
        
        rows, cols = self.config.rows, self.config.cols
        total_cells = rows * cols
        
        # Calculate number of obstacles and items
        num_obstacles = int(total_cells * self.config.obstacle_density)
        num_items = int(total_cells * self.config.item_density)
        
        # Start with empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Get all positions except corners (for agent spawns)
        all_positions = [(i, j) for i in range(rows) for j in range(cols)]
        corners = [(0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1)]
        available = [p for p in all_positions if p not in corners]
        
        random.shuffle(available)
        
        # Place obstacles
        for i in range(min(num_obstacles, len(available))):
            pos = available.pop()
            grid[pos[0], pos[1]] = CellType.OBSTACLE
        
        # Place items
        for i in range(min(num_items, len(available))):
            pos = available.pop()
            grid[pos[0], pos[1]] = CellType.ITEM
        
        # Ensure grid is traversable using BFS from corner
        if not self._is_traversable(grid, (0, 0)):
            logger.warning("Generated grid not fully traversable, regenerating...")
            return self._generate_random_grid()
        
        return grid
    
    def _is_traversable(self, grid: np.ndarray, start: Tuple[int, int]) -> bool:
        """Check if all non-obstacle cells are reachable from start."""
        rows, cols = grid.shape
        visited = set()
        queue = [start]
        visited.add(start)
        
        while queue:
            x, y = queue.pop(0)
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    if (nx, ny) not in visited and grid[nx, ny] != CellType.OBSTACLE:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        
        # Count non-obstacle cells
        non_obstacles = np.count_nonzero(grid != CellType.OBSTACLE)
        return len(visited) >= non_obstacles * 0.9  # Allow 10% unreachable
    
    @classmethod
    def from_scenario(cls, scenario_name: str) -> 'Environment':
        """Create environment from predefined scenario."""
        if scenario_name not in cls.SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(cls.SCENARIOS.keys())}")
        return cls(grid=cls.SCENARIOS[scenario_name].copy())
    
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within bounds and not an obstacle."""
        x, y = pos
        return (0 <= x < self.rows and 
                0 <= y < self.cols and 
                self.grid[x, y] != CellType.OBSTACLE)
    
    def get_cell(self, pos: Tuple[int, int]) -> Optional[int]:
        """Get the value at a specific position."""
        x, y = pos
        if 0 <= x < self.rows and 0 <= y < self.cols:
            return int(self.grid[x, y])
        return None
    
    def get_weight(self, pos: Tuple[int, int]) -> float:
        """Get movement cost for a position (used by Dijkstra)."""
        x, y = pos
        if 0 <= x < self.rows and 0 <= y < self.cols:
            return self.weights[x, y]
        return float('inf')
    
    def set_weight(self, pos: Tuple[int, int], weight: float) -> None:
        """Set movement cost for a position."""
        x, y = pos
        if 0 <= x < self.rows and 0 <= y < self.cols:
            self.weights[x, y] = weight
    
    def collect_item(self, pos: Tuple[int, int], agent_id: int = -1, step: int = -1) -> bool:
        """Remove item from position if it exists."""
        x, y = pos
        if self.grid[x, y] == CellType.ITEM:
            self.grid[x, y] = CellType.EMPTY
            self.collection_history.append({
                'position': pos,
                'agent_id': agent_id,
                'step': step
            })
            logger.debug(f"Item collected at {pos} by agent {agent_id}")
            return True
        return False
    
    def get_possible_moves(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Return all valid neighboring positions."""
        x, y = pos
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        moves = []
        for dx, dy in directions:
            new_pos = (x + dx, y + dy)
            if self.is_valid_position(new_pos):
                moves.append(new_pos)
        return moves
    
    def get_weighted_moves(self, pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        """Return valid moves with their weights for Dijkstra."""
        moves = self.get_possible_moves(pos)
        return [(move, self.get_weight(move)) for move in moves]
    
    def items_remaining(self) -> int:
        """Count remaining items in the grid."""
        return np.count_nonzero(self.grid == CellType.ITEM)
    
    def find_all_items(self) -> List[Tuple[int, int]]:
        """Return positions of all items in the grid."""
        items = []
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i, j] == CellType.ITEM:
                    items.append((i, j))
        return items
    
    def get_visible_cells(self, pos: Tuple[int, int], vision_radius: int = -1) -> Set[Tuple[int, int]]:
        """Get cells visible from a position within vision radius."""
        if vision_radius < 0:
            # Full visibility
            return {(i, j) for i in range(self.rows) for j in range(self.cols)
                    if self.grid[i, j] != CellType.OBSTACLE}
        
        visible = set()
        x, y = pos
        for i in range(max(0, x - vision_radius), min(self.rows, x + vision_radius + 1)):
            for j in range(max(0, y - vision_radius), min(self.cols, y + vision_radius + 1)):
                # Manhattan distance check
                if abs(i - x) + abs(j - y) <= vision_radius:
                    visible.add((i, j))
        return visible
    
    def record_visit(self, pos: Tuple[int, int]) -> None:
        """Record a visit to a cell for heatmap generation."""
        x, y = pos
        if 0 <= x < self.rows and 0 <= y < self.cols:
            self.visit_counts[x, y] += 1
    
    def get_heatmap_data(self) -> np.ndarray:
        """Get visit counts for heatmap visualization."""
        return self.visit_counts.copy()
    
    def add_dynamic_obstacle(self, pos: Tuple[int, int]) -> bool:
        """Add a dynamic obstacle at position."""
        x, y = pos
        if self.grid[x, y] == CellType.EMPTY:
            self.grid[x, y] = CellType.OBSTACLE
            self.dynamic_obstacles.append(pos)
            return True
        return False
    
    def remove_dynamic_obstacle(self, pos: Tuple[int, int]) -> bool:
        """Remove a dynamic obstacle from position."""
        if pos in self.dynamic_obstacles:
            x, y = pos
            self.grid[x, y] = CellType.EMPTY
            self.dynamic_obstacles.remove(pos)
            return True
        return False
    
    def move_dynamic_obstacles(self) -> None:
        """Move all dynamic obstacles randomly (for advanced scenarios)."""
        for obstacle in self.dynamic_obstacles[:]:
            moves = self.get_possible_moves(obstacle)
            if moves:
                new_pos = random.choice(moves)
                self.remove_dynamic_obstacle(obstacle)
                self.add_dynamic_obstacle(new_pos)
    
    def get_statistics(self) -> Dict:
        """Get current environment statistics."""
        return {
            'grid_size': (self.rows, self.cols),
            'total_cells': self.rows * self.cols,
            'obstacles': np.count_nonzero(self.grid == CellType.OBSTACLE),
            'initial_items': self.initial_items,
            'remaining_items': self.items_remaining(),
            'collected_items': self.initial_items - self.items_remaining(),
            'collection_rate': (self.initial_items - self.items_remaining()) / max(self.initial_items, 1),
            'total_visits': int(np.sum(self.visit_counts)),
            'max_cell_visits': int(np.max(self.visit_counts)),
            'dynamic_obstacles': len(self.dynamic_obstacles)
        }
    
    def display_grid(self) -> None:
        """Print the current state of the grid."""
        symbols = {0: '.', 1: '#', 2: '*', 3: 'o'}
        print("\nCurrent Grid State:")
        print("  " + " ".join(str(i) for i in range(self.cols)))
        for i, row in enumerate(self.grid):
            print(f"{i} " + " ".join(symbols.get(int(cell), '?') for cell in row))
        print(f"\nLegend: . = empty, # = obstacle, * = item, o = collected")
    
    def copy(self) -> 'Environment':
        """Create a deep copy of the environment."""
        config_copy = GridConfig(
            rows=self.config.rows,
            cols=self.config.cols,
            obstacle_density=self.config.obstacle_density,
            item_density=self.config.item_density,
            seed=self.config.seed
        )
        env = Environment(config=config_copy, grid=self.grid.copy())
        env.weights = self.weights.copy()
        env.visit_counts = self.visit_counts.copy()
        return env
