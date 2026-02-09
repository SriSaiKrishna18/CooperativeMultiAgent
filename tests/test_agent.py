"""
Unit tests for Agent module.
Tests pathfinding algorithms, movement, and decision making.
"""
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent import Agent
from environment import Environment, GridConfig
from cooperative_logic import CooperativeKnowledge
from config import PathfindingAlgorithm, AgentRole, AgentStrategy


class TestAgentInitialization:
    """Test agent initialization and basic properties."""
    
    def test_agent_creation(self):
        """Test basic agent creation."""
        agent = Agent(agent_id=0, start_position=(0, 0))
        assert agent.id == 0
        assert agent.position == (0, 0)
        assert agent.collected_items == 0
        assert len(agent.path) == 1
    
    def test_agent_with_custom_config(self):
        """Test agent with custom configuration."""
        agent = Agent(
            agent_id=1,
            start_position=(2, 3),
            algorithm=PathfindingAlgorithm.ASTAR,
            role=AgentRole.EXPLORER,
            strategy=AgentStrategy.GREEDY,
            max_energy=50,
            vision_radius=3
        )
        assert agent.algorithm == PathfindingAlgorithm.ASTAR
        assert agent.role == AgentRole.EXPLORER
        assert agent.strategy == AgentStrategy.GREEDY
        assert agent.energy == 50
        assert agent.vision_radius == 3
    
    def test_agent_status(self):
        """Test agent status reporting."""
        agent = Agent(agent_id=0, start_position=(1, 1))
        status = agent.get_status()
        
        assert 'id' in status
        assert 'position' in status
        assert 'collected_items' in status
        assert 'path_length' in status


class TestAgentMovement:
    """Test agent movement functionality."""
    
    def test_basic_movement(self):
        """Test basic agent movement."""
        agent = Agent(agent_id=0, start_position=(0, 0))
        success = agent.move((0, 1), step=1)
        
        assert success
        assert agent.position == (0, 1)
        assert len(agent.path) == 2
    
    def test_movement_with_energy(self):
        """Test movement with energy constraints."""
        agent = Agent(
            agent_id=0, 
            start_position=(0, 0),
            max_energy=5,
            energy_per_move=2
        )
        
        # Move twice - should work
        agent.move((0, 1), step=1)
        agent.move((0, 2), step=2)
        assert agent.energy == 1
        
        # Third move should fail - not enough energy
        success = agent.move((0, 3), step=3)
        assert not success
        assert agent.position == (0, 2)
    
    def test_collect_item(self):
        """Test item collection."""
        agent = Agent(agent_id=0, start_position=(0, 0))
        agent.collect_item(step=1)
        
        assert agent.collected_items == 1
        assert agent.current_target is None


class TestPathfinding:
    """Test pathfinding algorithms."""
    
    @pytest.fixture
    def simple_env(self):
        """Create a simple test environment."""
        config = GridConfig(rows=5, cols=5, use_predefined=True)
        config.predefined_layout = [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ]
        return Environment(config)
    
    def test_bfs_simple_path(self, simple_env):
        """Test BFS finds a path."""
        agent = Agent(
            agent_id=0, 
            start_position=(0, 0),
            algorithm=PathfindingAlgorithm.BFS
        )
        
        path = agent.bfs_path((0, 0), (4, 4), simple_env)
        
        assert len(path) > 0
        assert path[0] == (0, 0)
        assert path[-1] == (4, 4)
    
    def test_astar_simple_path(self, simple_env):
        """Test A* finds a path."""
        agent = Agent(
            agent_id=0,
            start_position=(0, 0),
            algorithm=PathfindingAlgorithm.ASTAR
        )
        
        path = agent.astar_path((0, 0), (4, 4), simple_env)
        
        assert len(path) > 0
        assert path[0] == (0, 0)
        assert path[-1] == (4, 4)
    
    def test_dijkstra_simple_path(self, simple_env):
        """Test Dijkstra finds a path."""
        agent = Agent(
            agent_id=0,
            start_position=(0, 0),
            algorithm=PathfindingAlgorithm.DIJKSTRA
        )
        
        path = agent.dijkstra_path((0, 0), (4, 4), simple_env)
        
        assert len(path) > 0
        assert path[0] == (0, 0)
        assert path[-1] == (4, 4)
    
    def test_path_avoids_obstacles(self, simple_env):
        """Test that paths avoid obstacles."""
        agent = Agent(agent_id=0, start_position=(0, 0))
        path = agent.find_path((0, 0), (0, 4), simple_env)
        
        # Check no position in path is an obstacle
        for pos in path:
            assert simple_env.is_valid_position(pos)
    
    def test_no_path_blocked(self):
        """Test behavior when no path exists."""
        config = GridConfig(rows=3, cols=3, use_predefined=True)
        config.predefined_layout = [
            [0, 1, 2],
            [1, 1, 1],
            [0, 0, 0]
        ]
        env = Environment(config)
        
        agent = Agent(agent_id=0, start_position=(0, 0))
        path = agent.find_path((0, 0), (0, 2), env)
        
        # Should return empty path since blocked
        assert len(path) == 0


class TestDecisionMaking:
    """Test agent decision making."""
    
    @pytest.fixture
    def env_with_items(self):
        """Create environment with items."""
        config = GridConfig(rows=5, cols=5, use_predefined=True)
        config.predefined_layout = [
            [0, 0, 0, 0, 2],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0]
        ]
        return Environment(config)
    
    def test_collector_chooses_item(self, env_with_items):
        """Test that collector role prioritizes items."""
        agent = Agent(
            agent_id=0,
            start_position=(2, 2),
            role=AgentRole.COLLECTOR
        )
        
        knowledge = CooperativeKnowledge()
        # Add known items
        knowledge.add_known_item((0, 4))
        knowledge.add_known_item((4, 0))
        
        move = agent.choose_move(env_with_items, knowledge)
        
        # Agent should move toward an item
        assert move != (2, 2)  # Should not stay in place


class TestBidding:
    """Test auction bidding functionality."""
    
    def test_calculate_bid(self):
        """Test bid calculation."""
        config = GridConfig(rows=5, cols=5)
        env = Environment(config)
        
        agent = Agent(
            agent_id=0,
            start_position=(0, 0),
            role=AgentRole.COLLECTOR
        )
        
        bid = agent.calculate_bid((2, 2), env)
        
        assert bid > 0
        assert bid < float('inf')
    
    def test_collector_gets_priority(self):
        """Test that collectors get better bids."""
        config = GridConfig(rows=5, cols=5)
        env = Environment(config)
        
        collector = Agent(
            agent_id=0,
            start_position=(0, 0),
            role=AgentRole.COLLECTOR
        )
        
        explorer = Agent(
            agent_id=1,
            start_position=(0, 0),
            role=AgentRole.EXPLORER
        )
        
        collector_bid = collector.calculate_bid((2, 2), env)
        explorer_bid = explorer.calculate_bid((2, 2), env)
        
        # Collector should have lower (better) bid
        assert collector_bid < explorer_bid


class TestAgentCommunication:
    """Test agent message passing."""
    
    def test_send_message(self):
        """Test sending messages."""
        agent = Agent(agent_id=0, start_position=(0, 0))
        
        msg = agent.send_message('item_found', {'position': (2, 2)})
        
        assert msg['sender'] == 0
        assert msg['type'] == 'item_found'
        assert len(agent.sent_messages) == 1
    
    def test_receive_message(self):
        """Test receiving messages."""
        agent = Agent(agent_id=0, start_position=(0, 0))
        
        msg = {'sender': 1, 'type': 'release', 'content': (2, 2)}
        agent.receive_message(msg)
        
        assert len(agent.messages) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
