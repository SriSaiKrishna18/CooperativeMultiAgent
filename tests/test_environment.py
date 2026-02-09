"""
Unit tests for Environment module.
Tests grid operations, validation, and cell management.
"""
import pytest
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from environment import Environment, GridConfig, CellType


class TestEnvironmentInitialization:
    """Test environment initialization."""
    
    def test_default_environment(self):
        """Test creating environment with default config."""
        env = Environment()
        
        assert env.rows > 0
        assert env.cols > 0
        assert env.total_items >= 0
    
    def test_custom_grid_config(self):
        """Test environment with custom grid configuration."""
        config = GridConfig(rows=10, cols=10, obstacle_density=0.2, item_density=0.1)
        env = Environment(config)
        
        assert env.rows == 10
        assert env.cols == 10
    
    def test_predefined_grid(self):
        """Test environment with predefined grid layout."""
        grid = [
            [0, 1, 2],
            [0, 0, 0],
            [2, 0, 1]
        ]
        env = Environment(grid=np.array(grid))
        
        assert env.rows == 3
        assert env.cols == 3
        assert env.total_items == 2
    
    def test_scenario_loading(self):
        """Test loading predefined scenarios."""
        env = Environment.from_scenario('simple')
        
        assert env.rows == 5
        assert env.cols == 5
        assert env.total_items > 0


class TestGridOperations:
    """Test grid cell operations."""
    
    @pytest.fixture
    def test_env(self):
        """Create test environment."""
        grid = [
            [0, 1, 2],
            [0, 0, 0],
            [2, 0, 1]
        ]
        return Environment(grid=np.array(grid))
    
    def test_is_valid_position(self, test_env):
        """Test position validation."""
        # Valid positions
        assert test_env.is_valid_position((0, 0))
        assert test_env.is_valid_position((1, 1))
        
        # Obstacles
        assert not test_env.is_valid_position((0, 1))
        assert not test_env.is_valid_position((2, 2))
        
        # Out of bounds
        assert not test_env.is_valid_position((-1, 0))
        assert not test_env.is_valid_position((3, 0))
    
    def test_get_cell(self, test_env):
        """Test getting cell values."""
        assert test_env.get_cell((0, 0)) == CellType.EMPTY
        assert test_env.get_cell((0, 1)) == CellType.OBSTACLE
        assert test_env.get_cell((0, 2)) == CellType.ITEM
        assert test_env.get_cell((10, 10)) is None
    
    def test_get_possible_moves(self, test_env):
        """Test getting valid neighboring positions."""
        moves = test_env.get_possible_moves((1, 1))
        
        # Should have moves in all 4 directions except blocked
        assert (1, 0) in moves  # Left
        assert (1, 2) in moves  # Right
        assert (2, 1) in moves  # Down
        # (0, 1) should not be in moves - it's an obstacle


class TestItemCollection:
    """Test item collection functionality."""
    
    @pytest.fixture
    def env_with_items(self):
        """Create environment with items."""
        grid = [
            [0, 0, 2],
            [0, 0, 0],
            [2, 0, 0]
        ]
        return Environment(grid=np.array(grid))
    
    def test_collect_item_success(self, env_with_items):
        """Test successful item collection."""
        initial_items = env_with_items.items_remaining()
        
        result = env_with_items.collect_item((0, 2), agent_id=0, step=1)
        
        assert result is True
        assert env_with_items.items_remaining() == initial_items - 1
    
    def test_collect_item_empty_cell(self, env_with_items):
        """Test collecting from empty cell."""
        result = env_with_items.collect_item((0, 0))
        
        assert result is False
    
    def test_find_all_items(self, env_with_items):
        """Test finding all item positions."""
        items = env_with_items.find_all_items()
        
        assert len(items) == 2
        assert (0, 2) in items
        assert (2, 0) in items


class TestWeightedTerrain:
    """Test weighted terrain functionality."""
    
    def test_default_weights(self):
        """Test that default weights are 1.0."""
        env = Environment()
        
        for i in range(env.rows):
            for j in range(env.cols):
                assert env.get_weight((i, j)) == 1.0
    
    def test_set_weight(self):
        """Test setting cell weights."""
        env = Environment()
        env.set_weight((0, 0), 5.0)
        
        assert env.get_weight((0, 0)) == 5.0
    
    def test_get_weighted_moves(self):
        """Test getting moves with weights."""
        env = Environment()
        env.set_weight((0, 1), 2.0)
        
        moves = env.get_weighted_moves((0, 0))
        
        # Should return tuples of (position, weight)
        assert len(moves) > 0
        for pos, weight in moves:
            assert isinstance(pos, tuple)
            assert isinstance(weight, float)


class TestVisibility:
    """Test visibility and vision radius functionality."""
    
    def test_full_visibility(self):
        """Test full visibility (vision_radius=-1)."""
        config = GridConfig(rows=5, cols=5)
        env = Environment(config)
        
        visible = env.get_visible_cells((2, 2), vision_radius=-1)
        
        # Should see all non-obstacle cells
        assert len(visible) > 0
    
    def test_limited_visibility(self):
        """Test limited visibility radius."""
        grid = np.zeros((10, 10))
        env = Environment(grid=grid)
        
        visible = env.get_visible_cells((5, 5), vision_radius=2)
        
        # Should only see cells within Manhattan distance 2
        for cell in visible:
            distance = abs(cell[0] - 5) + abs(cell[1] - 5)
            assert distance <= 2


class TestHeatmapTracking:
    """Test visit tracking for heatmaps."""
    
    def test_record_visit(self):
        """Test recording cell visits."""
        env = Environment()
        
        env.record_visit((0, 0))
        env.record_visit((0, 0))
        env.record_visit((1, 1))
        
        heatmap = env.get_heatmap_data()
        
        assert heatmap[0, 0] == 2
        assert heatmap[1, 1] == 1


class TestDynamicObstacles:
    """Test dynamic obstacle functionality."""
    
    def test_add_dynamic_obstacle(self):
        """Test adding dynamic obstacles."""
        grid = np.zeros((5, 5))
        env = Environment(grid=grid)
        
        result = env.add_dynamic_obstacle((2, 2))
        
        assert result is True
        assert env.get_cell((2, 2)) == CellType.OBSTACLE
        assert len(env.dynamic_obstacles) == 1
    
    def test_remove_dynamic_obstacle(self):
        """Test removing dynamic obstacles."""
        grid = np.zeros((5, 5))
        env = Environment(grid=grid)
        
        env.add_dynamic_obstacle((2, 2))
        result = env.remove_dynamic_obstacle((2, 2))
        
        assert result is True
        assert env.get_cell((2, 2)) == CellType.EMPTY
        assert len(env.dynamic_obstacles) == 0


class TestEnvironmentStatistics:
    """Test environment statistics."""
    
    def test_get_statistics(self):
        """Test getting environment statistics."""
        env = Environment.from_scenario('simple')
        
        # Collect some items
        items = env.find_all_items()
        if items:
            env.collect_item(items[0])
        
        stats = env.get_statistics()
        
        assert 'grid_size' in stats
        assert 'total_cells' in stats
        assert 'obstacles' in stats
        assert 'initial_items' in stats
        assert 'remaining_items' in stats
        assert 'collection_rate' in stats


class TestEnvironmentCopy:
    """Test environment copying."""
    
    def test_copy_creates_independent_instance(self):
        """Test that copy creates independent environment."""
        env1 = Environment.from_scenario('simple')
        env2 = env1.copy()
        
        # Modify env1
        items = env1.find_all_items()
        if items:
            env1.collect_item(items[0])
        
        # env2 should be unaffected
        assert env1.items_remaining() < env2.items_remaining()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
