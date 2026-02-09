"""
Unit tests for Cooperative Logic module.
Tests task allocation, message passing, and coordination mechanisms.
"""
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cooperative_logic import (
    CooperativeKnowledge, update_shared_knowledge,
    MessageType, AllocationMethod
)
from environment import Environment, GridConfig
from agent import Agent
from config import CooperationConfig


class TestCooperativeKnowledgeInitialization:
    """Test cooperative knowledge initialization."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        knowledge = CooperativeKnowledge()
        
        assert len(knowledge.visited) == 0
        assert len(knowledge.known_items) == 0
        assert len(knowledge.collected_items) == 0
        assert len(knowledge.claimed_items) == 0
    
    def test_with_config(self):
        """Test initialization with config."""
        config = CooperationConfig(
            allocation_method=AllocationMethod.AUCTION,
            enable_communication=True
        )
        knowledge = CooperativeKnowledge(config)
        
        assert knowledge.config.allocation_method == AllocationMethod.AUCTION


class TestBasicKnowledge:
    """Test basic knowledge operations."""
    
    @pytest.fixture
    def knowledge(self):
        return CooperativeKnowledge()
    
    def test_mark_visited(self, knowledge):
        """Test marking cells as visited."""
        knowledge.mark_visited((0, 0))
        knowledge.mark_visited((1, 1))
        
        assert (0, 0) in knowledge.visited
        assert (1, 1) in knowledge.visited
        assert knowledge.is_visited((0, 0))
    
    def test_add_known_item(self, knowledge):
        """Test adding known items."""
        knowledge.add_known_item((2, 2))
        
        assert (2, 2) in knowledge.known_items
    
    def test_mark_collected(self, knowledge):
        """Test marking items as collected."""
        knowledge.add_known_item((2, 2))
        knowledge.mark_collected((2, 2))
        
        assert (2, 2) not in knowledge.known_items
        assert (2, 2) in knowledge.collected_items
    
    def test_collected_item_not_re_added(self, knowledge):
        """Test that collected items can't be re-added."""
        knowledge.add_known_item((2, 2))
        knowledge.mark_collected((2, 2))
        knowledge.add_known_item((2, 2))  # Try to re-add
        
        assert (2, 2) not in knowledge.known_items


class TestSimpleClaim:
    """Test simple claim allocation."""
    
    @pytest.fixture
    def knowledge(self):
        config = CooperationConfig(allocation_method=AllocationMethod.SIMPLE_CLAIM)
        return CooperativeKnowledge(config)
    
    def test_successful_claim(self, knowledge):
        """Test successful item claim."""
        knowledge.add_known_item((2, 2))
        result = knowledge.claim_item((2, 2), agent_id=0)
        
        assert result is True
        assert knowledge.claimed_items[(2, 2)] == 0
    
    def test_claim_conflict(self, knowledge):
        """Test claim conflict."""
        knowledge.add_known_item((2, 2))
        knowledge.claim_item((2, 2), agent_id=0)
        result = knowledge.claim_item((2, 2), agent_id=1)
        
        assert result is False
        assert knowledge.claimed_items[(2, 2)] == 0
        assert knowledge.claim_conflicts == 1
    
    def test_get_available_items(self, knowledge):
        """Test getting available items."""
        knowledge.add_known_item((1, 1))
        knowledge.add_known_item((2, 2))
        knowledge.add_known_item((3, 3))
        knowledge.claim_item((2, 2), agent_id=0)
        
        available = knowledge.get_available_items()
        
        assert (1, 1) in available
        assert (2, 2) not in available
        assert (3, 3) in available
    
    def test_release_item(self, knowledge):
        """Test releasing a claimed item."""
        knowledge.add_known_item((2, 2))
        knowledge.claim_item((2, 2), agent_id=0)
        result = knowledge.release_item((2, 2), agent_id=0)
        
        assert result is True
        assert (2, 2) not in knowledge.claimed_items


class TestPriorityClaim:
    """Test priority-based allocation."""
    
    @pytest.fixture
    def knowledge(self):
        config = CooperationConfig(allocation_method=AllocationMethod.PRIORITY)
        return CooperativeKnowledge(config)
    
    def test_higher_priority_wins(self, knowledge):
        """Test that lower priority value wins."""
        knowledge.add_known_item((2, 2))
        
        knowledge.claim_item((2, 2), agent_id=0, priority=5.0)
        knowledge.claim_item((2, 2), agent_id=1, priority=3.0)  # Lower = better
        
        assert knowledge.claimed_items[(2, 2)] == 1


class TestAuctionAllocation:
    """Test auction-based allocation."""
    
    @pytest.fixture
    def knowledge(self):
        config = CooperationConfig(allocation_method=AllocationMethod.AUCTION)
        return CooperativeKnowledge(config)
    
    def test_auction_created(self, knowledge):
        """Test that auctions are created."""
        knowledge.add_known_item((2, 2))
        knowledge._start_auction((2, 2), agent_id=0, bid_value=5.0)
        
        assert (2, 2) in knowledge.active_auctions
        assert len(knowledge.active_auctions[(2, 2)]) == 1
    
    def test_multiple_bids(self, knowledge):
        """Test multiple bids on same item."""
        knowledge.add_known_item((2, 2))
        knowledge._start_auction((2, 2), agent_id=0, bid_value=5.0)
        knowledge._start_auction((2, 2), agent_id=1, bid_value=3.0)
        
        assert len(knowledge.active_auctions[(2, 2)]) == 2


class TestMessagePassing:
    """Test message passing functionality."""
    
    @pytest.fixture
    def knowledge(self):
        config = CooperationConfig(enable_communication=True, message_delay=0)
        return CooperativeKnowledge(config)
    
    def test_send_message(self, knowledge):
        """Test sending messages."""
        knowledge.send_message(
            sender_id=0,
            recipient_id=1,
            msg_type='item_found',
            content={'position': (2, 2)}
        )
        
        assert len(knowledge.message_queue) == 1
        assert knowledge.messages_sent == 1
    
    def test_broadcast_message(self, knowledge):
        """Test broadcasting messages."""
        knowledge._broadcast(MessageType.ITEM_FOUND, {'position': (2, 2)})
        
        assert len(knowledge.message_queue) == 1
        msg = knowledge.message_queue[0]
        assert msg.recipient_id == -1  # Broadcast
    
    def test_get_messages_for_agent(self, knowledge):
        """Test retrieving messages for specific agent."""
        knowledge.send_message(0, 1, 'item_found', {'position': (1, 1)})
        knowledge.send_message(0, 2, 'item_found', {'position': (2, 2)})
        knowledge.send_message(0, -1, 'item_found', {'position': (3, 3)})  # Broadcast
        
        messages_for_1 = knowledge.get_messages_for_agent(1)
        
        # Should get direct message + broadcast
        assert len(messages_for_1) == 2


class TestZoneAllocation:
    """Test zone-based allocation."""
    
    def test_zone_assignment(self):
        """Test zone assignment."""
        knowledge = CooperativeKnowledge()
        
        config = GridConfig(rows=10, cols=10)
        env = Environment(config)
        
        agents = [
            Agent(0, (0, 0)),
            Agent(1, (9, 9))
        ]
        
        knowledge.assign_zones(env, agents, method='vertical')
        
        assert len(knowledge.zones) == 2
        assert 0 in knowledge.agent_zones
        assert 1 in knowledge.agent_zones
    
    def test_get_agent_zone(self):
        """Test getting agent's zone."""
        knowledge = CooperativeKnowledge()
        
        config = GridConfig(rows=10, cols=10)
        env = Environment(config)
        
        agents = [Agent(0, (0, 0)), Agent(1, (9, 9))]
        knowledge.assign_zones(env, agents)
        
        zone = knowledge.get_agent_zone(0)
        
        assert zone is not None
        assert len(zone.cells) > 0


class TestStatistics:
    """Test cooperation statistics."""
    
    def test_get_statistics(self):
        """Test getting cooperation statistics."""
        knowledge = CooperativeKnowledge()
        
        knowledge.mark_visited((0, 0))
        knowledge.add_known_item((1, 1))
        knowledge.claim_item((1, 1), 0)
        
        stats = knowledge.get_statistics()
        
        assert stats['visited_cells'] == 1
        assert stats['known_items'] == 1
        assert stats['active_claims'] == 1


class TestUpdateSharedKnowledge:
    """Test the update_shared_knowledge function."""
    
    def test_updates_visited(self):
        """Test that agent position is marked as visited."""
        env = Environment.from_scenario('simple')
        knowledge = CooperativeKnowledge()
        agent = Agent(0, (0, 0))
        
        update_shared_knowledge(agent, env, knowledge)
        
        assert (0, 0) in knowledge.visited
    
    def test_discovers_items(self):
        """Test that visible items are discovered."""
        grid = [
            [0, 2, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        env = Environment(grid=grid)
        knowledge = CooperativeKnowledge()
        agent = Agent(0, (0, 0))
        
        update_shared_knowledge(agent, env, knowledge)
        
        assert (0, 1) in knowledge.known_items


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
