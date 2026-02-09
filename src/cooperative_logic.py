"""
Enhanced Cooperation Module for Multi-Agent System.
Implements auction-based allocation, message passing, and advanced coordination.
"""
import logging
from typing import Dict, Set, Tuple, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from config import AllocationMethod, CooperationConfig


logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of inter-agent messages."""
    ITEM_FOUND = "item_found"
    ITEM_CLAIMED = "item_claimed"
    ITEM_RELEASED = "item_released"
    ITEM_COLLECTED = "item_collected"
    BID_SUBMIT = "bid_submit"
    BID_WIN = "bid_win"
    BID_LOSE = "bid_lose"
    ZONE_ASSIGNED = "zone_assigned"
    HELP_REQUEST = "help_request"
    STATUS_UPDATE = "status_update"


@dataclass
class Message:
    """Inter-agent message structure."""
    sender_id: int
    recipient_id: int  # -1 for broadcast
    msg_type: MessageType
    content: Any
    timestamp: int
    delivered: bool = False


@dataclass
class Bid:
    """Bid structure for auction-based allocation."""
    agent_id: int
    item_position: Tuple[int, int]
    bid_value: float
    timestamp: int


@dataclass
class Zone:
    """Territory zone for zone-based allocation."""
    zone_id: int
    assigned_agent: int
    cells: Set[Tuple[int, int]] = field(default_factory=set)
    items: Set[Tuple[int, int]] = field(default_factory=set)


class CooperativeKnowledge:
    """
    Enhanced shared knowledge base with:
    - Multiple allocation methods (simple claim, auction, priority, zone-based)
    - Message passing system
    - Conflict resolution
    - Performance tracking
    """
    
    def __init__(self, config: Optional[CooperationConfig] = None):
        """
        Initialize cooperative knowledge base.
        
        Args:
            config: Cooperation configuration settings
        """
        self.config = config or CooperationConfig()
        
        # Core knowledge
        self.visited: Set[Tuple[int, int]] = set()
        self.known_items: Set[Tuple[int, int]] = set()
        self.collected_items: Set[Tuple[int, int]] = set()
        
        # Task allocation
        self.claimed_items: Dict[Tuple[int, int], int] = {}  # item -> agent_id
        self.claim_timestamps: Dict[Tuple[int, int], int] = {}  # item -> timestamp
        
        # Priority-based allocation
        self.claim_priorities: Dict[Tuple[int, int], float] = {}  # item -> priority
        
        # Auction system
        self.active_auctions: Dict[Tuple[int, int], List[Bid]] = {}
        self.auction_winners: Dict[Tuple[int, int], int] = {}
        self.auction_timeout: int = 3  # Steps before auction closes
        
        # Message passing
        self.message_queue: List[Message] = []
        self.message_history: List[Message] = []
        self.current_step: int = 0
        
        # Zone-based allocation
        self.zones: List[Zone] = []
        self.agent_zones: Dict[int, int] = {}  # agent_id -> zone_id
        
        # Statistics
        self.claim_conflicts: int = 0
        self.successful_claims: int = 0
        self.messages_sent: int = 0
        self.auctions_completed: int = 0
        
        logger.info(f"CooperativeKnowledge initialized with {self.config.allocation_method.value}")
    
    # ==================== CORE KNOWLEDGE ====================
    
    def mark_visited(self, position: Tuple[int, int]) -> None:
        """Mark a position as visited."""
        self.visited.add(position)
    
    def add_known_item(self, position: Tuple[int, int]) -> None:
        """Add a newly discovered item location."""
        if position not in self.collected_items:
            self.known_items.add(position)
            logger.debug(f"Item discovered at {position}")
    
    def mark_collected(self, position: Tuple[int, int], agent_id: int = -1) -> None:
        """Mark an item as collected."""
        if position in self.known_items:
            self.known_items.remove(position)
        self.collected_items.add(position)
        
        # Clean up claims
        if position in self.claimed_items:
            del self.claimed_items[position]
        if position in self.claim_timestamps:
            del self.claim_timestamps[position]
        if position in self.claim_priorities:
            del self.claim_priorities[position]
        if position in self.active_auctions:
            del self.active_auctions[position]
        if position in self.auction_winners:
            del self.auction_winners[position]
        
        # Broadcast collection
        if self.config.enable_communication:
            self._broadcast(MessageType.ITEM_COLLECTED, position)
        
        logger.debug(f"Item collected at {position} by agent {agent_id}")
    
    def is_visited(self, position: Tuple[int, int]) -> bool:
        """Check if position has been visited."""
        return position in self.visited
    
    # ==================== TASK ALLOCATION ====================
    
    def claim_item(
        self, 
        position: Tuple[int, int], 
        agent_id: int,
        priority: float = 0.0
    ) -> bool:
        """
        Attempt to claim an item using configured allocation method.
        
        Args:
            position: Item position
            agent_id: Claiming agent's ID
            priority: Priority value (lower = higher priority)
            
        Returns:
            True if claim was successful
        """
        if position not in self.known_items:
            return False
        
        if self.config.allocation_method == AllocationMethod.SIMPLE_CLAIM:
            return self._simple_claim(position, agent_id)
        elif self.config.allocation_method == AllocationMethod.PRIORITY:
            return self._priority_claim(position, agent_id, priority)
        elif self.config.allocation_method == AllocationMethod.AUCTION:
            return self._start_auction(position, agent_id, priority)
        elif self.config.allocation_method == AllocationMethod.ZONE_BASED:
            return self._zone_claim(position, agent_id)
        
        return self._simple_claim(position, agent_id)
    
    def _simple_claim(self, position: Tuple[int, int], agent_id: int) -> bool:
        """Simple first-come-first-served claiming."""
        if position in self.claimed_items:
            self.claim_conflicts += 1
            return False
        
        self.claimed_items[position] = agent_id
        self.claim_timestamps[position] = self.current_step
        self.successful_claims += 1
        
        if self.config.enable_communication:
            self._broadcast(MessageType.ITEM_CLAIMED, {'position': position, 'agent': agent_id})
        
        logger.debug(f"Agent {agent_id} claimed item at {position}")
        return True
    
    def _priority_claim(
        self, 
        position: Tuple[int, int], 
        agent_id: int,
        priority: float
    ) -> bool:
        """Priority-based claiming - lower priority value wins."""
        current_claimer = self.claimed_items.get(position)
        current_priority = self.claim_priorities.get(position, float('inf'))
        
        if current_claimer is None or priority < current_priority:
            # This agent has higher priority (lower value)
            if current_claimer is not None:
                self.claim_conflicts += 1
                # Notify previous claimer they lost
                if self.config.enable_communication:
                    self._send_message(
                        -1, current_claimer, 
                        MessageType.ITEM_RELEASED, 
                        position
                    )
            
            self.claimed_items[position] = agent_id
            self.claim_priorities[position] = priority
            self.claim_timestamps[position] = self.current_step
            self.successful_claims += 1
            
            if self.config.enable_communication:
                self._broadcast(MessageType.ITEM_CLAIMED, {'position': position, 'agent': agent_id})
            
            return True
        
        self.claim_conflicts += 1
        return False
    
    def _start_auction(
        self, 
        position: Tuple[int, int], 
        agent_id: int,
        bid_value: float
    ) -> bool:
        """Start or participate in an auction for an item."""
        bid = Bid(
            agent_id=agent_id,
            item_position=position,
            bid_value=bid_value,
            timestamp=self.current_step
        )
        
        if position not in self.active_auctions:
            self.active_auctions[position] = []
        
        self.active_auctions[position].append(bid)
        
        if self.config.enable_communication:
            self._broadcast(MessageType.BID_SUBMIT, {
                'position': position,
                'agent': agent_id,
                'bid': bid_value
            })
        
        # Auction will be resolved in process_auctions()
        return True
    
    def process_auctions(self, agents: List[Any]) -> Dict[Tuple[int, int], int]:
        """
        Process all active auctions and determine winners.
        
        Args:
            agents: List of agent objects for notification
            
        Returns:
            Dictionary of item positions to winning agent IDs
        """
        winners = {}
        auctions_to_close = []
        
        for position, bids in self.active_auctions.items():
            # Check if auction should close (has bids and timeout passed)
            if bids:
                oldest_bid = min(b.timestamp for b in bids)
                if self.current_step - oldest_bid >= self.auction_timeout:
                    auctions_to_close.append(position)
        
        for position in auctions_to_close:
            bids = self.active_auctions[position]
            
            # Winner has lowest bid value
            winner_bid = min(bids, key=lambda b: b.bid_value)
            winner_id = winner_bid.agent_id
            
            self.claimed_items[position] = winner_id
            self.auction_winners[position] = winner_id
            self.auctions_completed += 1
            winners[position] = winner_id
            
            # Notify all bidders
            for bid in bids:
                agent = next((a for a in agents if a.id == bid.agent_id), None)
                if agent:
                    if bid.agent_id == winner_id:
                        agent.win_bid(position)
                        if self.config.enable_communication:
                            self._send_message(-1, bid.agent_id, MessageType.BID_WIN, position)
                    else:
                        agent.lose_bid(position)
                        if self.config.enable_communication:
                            self._send_message(-1, bid.agent_id, MessageType.BID_LOSE, position)
            
            del self.active_auctions[position]
            logger.debug(f"Auction for {position} won by agent {winner_id}")
        
        return winners
    
    def _zone_claim(self, position: Tuple[int, int], agent_id: int) -> bool:
        """Zone-based claiming - items in agent's zone are auto-claimed."""
        agent_zone_id = self.agent_zones.get(agent_id)
        
        if agent_zone_id is None:
            # Agent has no zone, fall back to simple claim
            return self._simple_claim(position, agent_id)
        
        zone = next((z for z in self.zones if z.zone_id == agent_zone_id), None)
        if zone and position in zone.cells:
            # Item is in agent's zone
            return self._simple_claim(position, agent_id)
        
        # Item is in another agent's zone
        self.claim_conflicts += 1
        return False
    
    def release_item(self, position: Tuple[int, int], agent_id: int) -> bool:
        """Release a claimed item."""
        if position in self.claimed_items and self.claimed_items[position] == agent_id:
            del self.claimed_items[position]
            if position in self.claim_timestamps:
                del self.claim_timestamps[position]
            if position in self.claim_priorities:
                del self.claim_priorities[position]
            
            if self.config.enable_communication:
                self._broadcast(MessageType.ITEM_RELEASED, position)
            
            logger.debug(f"Agent {agent_id} released item at {position}")
            return True
        return False
    
    def get_available_items(self) -> Set[Tuple[int, int]]:
        """Get items that are known but not yet claimed."""
        return self.known_items - set(self.claimed_items.keys())
    
    def get_agent_claimed_items(self, agent_id: int) -> Set[Tuple[int, int]]:
        """Get all items claimed by a specific agent."""
        return {pos for pos, aid in self.claimed_items.items() if aid == agent_id}
    
    # ==================== MESSAGE PASSING ====================
    
    def _send_message(
        self, 
        sender_id: int, 
        recipient_id: int,
        msg_type: MessageType,
        content: Any
    ) -> None:
        """Internal method to send a message."""
        msg = Message(
            sender_id=sender_id,
            recipient_id=recipient_id,
            msg_type=msg_type,
            content=content,
            timestamp=self.current_step
        )
        self.message_queue.append(msg)
        self.messages_sent += 1
    
    def _broadcast(self, msg_type: MessageType, content: Any) -> None:
        """Broadcast message to all agents."""
        self._send_message(-1, -1, msg_type, content)
    
    def send_message(
        self, 
        sender_id: int, 
        recipient_id: int,
        msg_type: str,
        content: Any
    ) -> bool:
        """
        Send a message from one agent to another.
        
        Args:
            sender_id: Sending agent's ID
            recipient_id: Receiving agent's ID (-1 for broadcast)
            msg_type: Message type string
            content: Message content
            
        Returns:
            True if message was queued
        """
        if not self.config.enable_communication:
            return False
        
        try:
            msg_enum = MessageType(msg_type)
        except ValueError:
            msg_enum = MessageType.STATUS_UPDATE
        
        self._send_message(sender_id, recipient_id, msg_enum, content)
        return True
    
    def get_messages_for_agent(self, agent_id: int) -> List[Message]:
        """Get all pending messages for an agent."""
        messages = []
        remaining = []
        
        for msg in self.message_queue:
            if msg.recipient_id == agent_id or msg.recipient_id == -1:
                # Check message delay
                if self.current_step - msg.timestamp >= self.config.message_delay:
                    messages.append(msg)
                    msg.delivered = True
                else:
                    remaining.append(msg)
            else:
                remaining.append(msg)
        
        self.message_queue = remaining
        self.message_history.extend(messages)
        
        return messages
    
    def deliver_all_messages(self, agents: List[Any]) -> int:
        """Deliver messages to all agents."""
        delivered = 0
        for agent in agents:
            messages = self.get_messages_for_agent(agent.id)
            for msg in messages:
                agent.receive_message({
                    'sender': msg.sender_id,
                    'type': msg.msg_type.value,
                    'content': msg.content,
                    'timestamp': msg.timestamp
                })
                delivered += 1
        return delivered
    
    # ==================== ZONE MANAGEMENT ====================
    
    def assign_zones(
        self, 
        environment: Any, 
        agents: List[Any],
        method: str = 'quadrant'
    ) -> None:
        """
        Assign territorial zones to agents.
        
        Args:
            environment: Environment object
            agents: List of agents
            method: Zone division method ('quadrant', 'vertical', 'horizontal')
        """
        rows, cols = environment.rows, environment.cols
        num_agents = len(agents)
        
        self.zones.clear()
        self.agent_zones.clear()
        
        if method == 'quadrant' and num_agents == 4:
            # Divide into 4 quadrants
            mid_r, mid_c = rows // 2, cols // 2
            zone_bounds = [
                (0, 0, mid_r, mid_c),
                (0, mid_c, mid_r, cols),
                (mid_r, 0, rows, mid_c),
                (mid_r, mid_c, rows, cols)
            ]
        elif method == 'vertical':
            # Divide into vertical strips
            strip_width = cols // num_agents
            zone_bounds = [
                (0, i * strip_width, rows, (i + 1) * strip_width if i < num_agents - 1 else cols)
                for i in range(num_agents)
            ]
        else:  # horizontal
            # Divide into horizontal strips
            strip_height = rows // num_agents
            zone_bounds = [
                (i * strip_height, 0, (i + 1) * strip_height if i < num_agents - 1 else rows, cols)
                for i in range(num_agents)
            ]
        
        for i, (r1, c1, r2, c2) in enumerate(zone_bounds[:num_agents]):
            cells = set()
            items = set()
            for r in range(r1, r2):
                for c in range(c1, c2):
                    if environment.is_valid_position((r, c)):
                        cells.add((r, c))
                        if environment.get_cell((r, c)) == 2:
                            items.add((r, c))
            
            zone = Zone(zone_id=i, assigned_agent=agents[i].id, cells=cells, items=items)
            self.zones.append(zone)
            self.agent_zones[agents[i].id] = i
        
        # Notify agents
        if self.config.enable_communication:
            for zone in self.zones:
                self._send_message(
                    -1, zone.assigned_agent,
                    MessageType.ZONE_ASSIGNED,
                    {'zone_id': zone.zone_id, 'cells': len(zone.cells)}
                )
        
        logger.info(f"Assigned {len(self.zones)} zones using {method} method")
    
    def get_agent_zone(self, agent_id: int) -> Optional[Zone]:
        """Get the zone assigned to an agent."""
        zone_id = self.agent_zones.get(agent_id)
        if zone_id is not None:
            return next((z for z in self.zones if z.zone_id == zone_id), None)
        return None
    
    # ==================== STEP MANAGEMENT ====================
    
    def step(self) -> None:
        """Advance the simulation step counter."""
        self.current_step += 1
    
    # ==================== STATISTICS ====================
    
    def get_statistics(self) -> Dict:
        """Get cooperation statistics."""
        return {
            'visited_cells': len(self.visited),
            'known_items': len(self.known_items),
            'collected_items': len(self.collected_items),
            'active_claims': len(self.claimed_items),
            'claim_conflicts': self.claim_conflicts,
            'successful_claims': self.successful_claims,
            'messages_sent': self.messages_sent,
            'pending_messages': len(self.message_queue),
            'active_auctions': len(self.active_auctions),
            'auctions_completed': self.auctions_completed,
            'zones_assigned': len(self.zones),
            'allocation_method': self.config.allocation_method.value
        }
    
    def get_shared_knowledge(self) -> Dict:
        """Return dictionary of all shared knowledge."""
        return {
            'visited': self.visited,
            'known_items': self.known_items,
            'collected_items': self.collected_items,
            'claimed_items': self.claimed_items
        }


def update_shared_knowledge(
    agent: Any, 
    environment: Any, 
    shared_knowledge: CooperativeKnowledge
) -> None:
    """
    Update shared knowledge based on agent's observations.
    
    Args:
        agent: Agent object
        environment: Environment object
        shared_knowledge: Shared knowledge base
    """
    pos = agent.position
    shared_knowledge.mark_visited(pos)
    environment.record_visit(pos)
    
    # Check current cell for item
    if environment.get_cell(pos) == 2:
        shared_knowledge.add_known_item(pos)
    
    # Get visible cells based on vision radius
    vision_radius = getattr(agent, 'vision_radius', -1)
    visible_cells = environment.get_visible_cells(pos, vision_radius)
    
    # Share observations of visible cells
    for cell in visible_cells:
        if environment.get_cell(cell) == 2:
            shared_knowledge.add_known_item(cell)


def create_auction_round(
    items: Set[Tuple[int, int]],
    agents: List[Any],
    environment: Any,
    shared_knowledge: CooperativeKnowledge
) -> Dict[Tuple[int, int], int]:
    """
    Conduct an auction round for all available items.
    
    Args:
        items: Set of item positions to auction
        agents: List of agents
        environment: Environment object
        shared_knowledge: Shared knowledge base
        
    Returns:
        Dictionary mapping items to winning agent IDs
    """
    # Collect all bids
    for item in items:
        for agent in agents:
            bid = agent.calculate_bid(item, environment)
            if bid < float('inf'):
                agent.submit_bid(item, bid)
                shared_knowledge._start_auction(item, agent.id, bid)
    
    # Let auction timeout pass
    for _ in range(shared_knowledge.auction_timeout + 1):
        shared_knowledge.step()
    
    # Process auctions
    return shared_knowledge.process_auctions(agents)
