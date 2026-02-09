"""
Enhanced Agent Module for Cooperative Multi-Agent System.
Implements multiple pathfinding algorithms, agent roles, and behavior strategies.
"""
import random
import heapq
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set, Any
from enum import Enum

from config import (PathfindingAlgorithm, AgentRole, AgentStrategy, 
                    AgentConfig, DEFAULT_CONFIG)


logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Snapshot of agent state for tracking and analysis."""
    step: int
    position: Tuple[int, int]
    collected_items: int
    energy: int
    target: Optional[Tuple[int, int]]
    action: str


class Agent:
    """
    Enhanced autonomous agent with:
    - Multiple pathfinding algorithms (BFS, A*, Dijkstra)
    - Agent roles (Explorer, Collector, Coordinator)
    - Behavior strategies (Greedy, Balanced, Cooperative)
    - Energy constraints
    - Comprehensive state tracking
    """
    
    def __init__(
        self,
        agent_id: int,
        start_position: Tuple[int, int],
        algorithm: PathfindingAlgorithm = PathfindingAlgorithm.ASTAR,
        role: AgentRole = AgentRole.COLLECTOR,
        strategy: AgentStrategy = AgentStrategy.COOPERATIVE,
        max_energy: int = -1,
        energy_per_move: int = 1,
        vision_radius: int = -1
    ):
        """
        Initialize an agent.
        
        Args:
            agent_id: Unique identifier
            start_position: Starting (row, col) position
            algorithm: Pathfinding algorithm to use
            role: Agent's specialized role
            strategy: Behavioral strategy
            max_energy: Maximum energy (-1 for unlimited)
            energy_per_move: Energy cost per move
            vision_radius: How far agent can see (-1 for unlimited)
        """
        self.id = agent_id
        self.position = start_position
        self.algorithm = algorithm
        self.role = role
        self.strategy = strategy
        
        # Stats
        self.collected_items = 0
        self.path: List[Tuple[int, int]] = [start_position]
        self.current_target: Optional[Tuple[int, int]] = None
        self.planned_path: List[Tuple[int, int]] = []
        
        # Energy system
        self.max_energy = max_energy
        self.energy = max_energy if max_energy > 0 else float('inf')
        self.energy_per_move = energy_per_move
        
        # Vision
        self.vision_radius = vision_radius
        
        # Communication
        self.messages: List[Dict] = []
        self.sent_messages: List[Dict] = []
        
        # History
        self.state_history: List[AgentState] = []
        self.cells_explored: Set[Tuple[int, int]] = {start_position}
        
        # Bidding for auction-based allocation
        self.current_bid: float = 0.0
        self.pending_bids: Dict[Tuple[int, int], float] = {}
        
        logger.info(f"Agent {agent_id} initialized at {start_position} "
                   f"[{algorithm.value}, {role.value}, {strategy.value}]")
    
    def move(self, new_position: Tuple[int, int], step: int = -1) -> bool:
        """
        Move agent to a new position.
        
        Args:
            new_position: Target position
            step: Current simulation step
            
        Returns:
            True if move was successful
        """
        if self.energy < self.energy_per_move and self.max_energy > 0:
            logger.debug(f"Agent {self.id} has insufficient energy to move")
            return False
        
        self.position = new_position
        self.path.append(new_position)
        self.cells_explored.add(new_position)
        
        if self.max_energy > 0:
            self.energy -= self.energy_per_move
        
        # Record state
        self._record_state(step, "move")
        
        return True
    
    def collect_item(self, step: int = -1) -> None:
        """Collect an item at current position."""
        self.collected_items += 1
        self.current_target = None
        self.planned_path = []
        self._record_state(step, "collect")
        logger.debug(f"Agent {self.id} collected item (total: {self.collected_items})")
    
    def recharge(self, amount: int = -1) -> None:
        """Recharge agent's energy."""
        if self.max_energy > 0:
            if amount < 0:
                self.energy = self.max_energy
            else:
                self.energy = min(self.max_energy, self.energy + amount)
    
    def _record_state(self, step: int, action: str) -> None:
        """Record current state for history."""
        state = AgentState(
            step=step,
            position=self.position,
            collected_items=self.collected_items,
            energy=int(self.energy) if self.max_energy > 0 else -1,
            target=self.current_target,
            action=action
        )
        self.state_history.append(state)
    
    # ==================== PATHFINDING ALGORITHMS ====================
    
    def find_path(
        self, 
        start: Tuple[int, int], 
        goal: Tuple[int, int], 
        environment: Any
    ) -> List[Tuple[int, int]]:
        """
        Find path using the agent's configured algorithm.
        
        Args:
            start: Starting position
            goal: Goal position
            environment: Environment object
            
        Returns:
            List of positions from start to goal, or empty if no path
        """
        if self.algorithm == PathfindingAlgorithm.BFS:
            return self.bfs_path(start, goal, environment)
        elif self.algorithm == PathfindingAlgorithm.ASTAR:
            return self.astar_path(start, goal, environment)
        elif self.algorithm == PathfindingAlgorithm.DIJKSTRA:
            return self.dijkstra_path(start, goal, environment)
        else:
            return self.bfs_path(start, goal, environment)
    
    def bfs_path(
        self, 
        start: Tuple[int, int], 
        goal: Tuple[int, int], 
        environment: Any
    ) -> List[Tuple[int, int]]:
        """
        Breadth-First Search for unweighted shortest path.
        Time Complexity: O(V + E)
        """
        if start == goal:
            return [start]
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            for neighbor in environment.get_possible_moves(current):
                if neighbor in visited:
                    continue
                
                new_path = path + [neighbor]
                
                if neighbor == goal:
                    return new_path
                
                visited.add(neighbor)
                queue.append((neighbor, new_path))
        
        return []
    
    def astar_path(
        self, 
        start: Tuple[int, int], 
        goal: Tuple[int, int], 
        environment: Any
    ) -> List[Tuple[int, int]]:
        """
        A* pathfinding with Manhattan distance heuristic.
        Optimal for grid-based environments.
        Time Complexity: O(E log V)
        """
        if start == goal:
            return [start]
        
        def heuristic(pos: Tuple[int, int]) -> float:
            """Manhattan distance heuristic."""
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        
        # Priority queue: (f_score, g_score, position, path)
        # g_score as tiebreaker for equal f_scores
        heap = [(heuristic(start), 0, start, [start])]
        visited = set()
        g_scores = {start: 0}
        
        while heap:
            f_score, g_score, current, path = heapq.heappop(heap)
            
            if current in visited:
                continue
            
            if current == goal:
                return path
            
            visited.add(current)
            
            for neighbor in environment.get_possible_moves(current):
                if neighbor in visited:
                    continue
                
                # Cost to reach neighbor (default 1, can use weighted terrain)
                try:
                    move_cost = environment.get_weight(neighbor)
                except AttributeError:
                    move_cost = 1
                
                new_g = g_score + move_cost
                
                if neighbor not in g_scores or new_g < g_scores[neighbor]:
                    g_scores[neighbor] = new_g
                    f = new_g + heuristic(neighbor)
                    heapq.heappush(heap, (f, new_g, neighbor, path + [neighbor]))
        
        return []
    
    def dijkstra_path(
        self, 
        start: Tuple[int, int], 
        goal: Tuple[int, int], 
        environment: Any
    ) -> List[Tuple[int, int]]:
        """
        Dijkstra's algorithm for weighted shortest path.
        Best for environments with varying terrain costs.
        Time Complexity: O(E log V)
        """
        if start == goal:
            return [start]
        
        # Priority queue: (distance, position, path)
        heap = [(0, start, [start])]
        visited = set()
        distances = {start: 0}
        
        while heap:
            dist, current, path = heapq.heappop(heap)
            
            if current in visited:
                continue
            
            if current == goal:
                return path
            
            visited.add(current)
            
            for neighbor in environment.get_possible_moves(current):
                if neighbor in visited:
                    continue
                
                try:
                    weight = environment.get_weight(neighbor)
                except AttributeError:
                    weight = 1
                
                new_dist = dist + weight
                
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(heap, (new_dist, neighbor, path + [neighbor]))
        
        return []
    
    # ==================== DECISION MAKING ====================
    
    def choose_move(
        self, 
        environment: Any, 
        shared_knowledge: Any
    ) -> Tuple[int, int]:
        """
        Intelligent move selection based on role and strategy.
        
        Args:
            environment: Environment object
            shared_knowledge: Shared knowledge base
            
        Returns:
            Next position to move to
        """
        # Check energy constraint
        if self.max_energy > 0 and self.energy < self.energy_per_move:
            logger.debug(f"Agent {self.id} cannot move - out of energy")
            return self.position
        
        # If following a planned path, continue on it
        if self.planned_path and len(self.planned_path) > 1:
            next_pos = self.planned_path[1]
            self.planned_path = self.planned_path[1:]
            return next_pos
        
        # Role-based decision making
        if self.role == AgentRole.EXPLORER:
            return self._explorer_decision(environment, shared_knowledge)
        elif self.role == AgentRole.COORDINATOR:
            return self._coordinator_decision(environment, shared_knowledge)
        else:  # COLLECTOR
            return self._collector_decision(environment, shared_knowledge)
    
    def _collector_decision(
        self, 
        environment: Any, 
        shared_knowledge: Any
    ) -> Tuple[int, int]:
        """Decision logic for collector role - prioritize items."""
        available_items = shared_knowledge.get_available_items()
        
        if available_items:
            return self._find_and_claim_best_item(
                available_items, environment, shared_knowledge
            )
        
        # No items - explore
        return self._explore(environment, shared_knowledge)
    
    def _explorer_decision(
        self, 
        environment: Any, 
        shared_knowledge: Any
    ) -> Tuple[int, int]:
        """Decision logic for explorer role - prioritize new cells."""
        # First, explore unvisited cells
        possible = environment.get_possible_moves(self.position)
        unvisited = [m for m in possible if m not in shared_knowledge.visited]
        
        if unvisited:
            return random.choice(unvisited)
        
        # If all nearby explored, try to reach unexplored areas
        all_cells = set()
        for i in range(environment.rows):
            for j in range(environment.cols):
                if environment.is_valid_position((i, j)):
                    all_cells.add((i, j))
        
        unexplored = all_cells - shared_knowledge.visited
        if unexplored:
            # Find closest unexplored cell
            best_target = None
            best_path = None
            shortest = float('inf')
            
            for cell in list(unexplored)[:10]:  # Limit search
                path = self.find_path(self.position, cell, environment)
                if path and len(path) < shortest:
                    shortest = len(path)
                    best_path = path
                    best_target = cell
            
            if best_path and len(best_path) > 1:
                self.planned_path = best_path
                return best_path[1]
        
        # Fall back to random exploration
        if possible:
            return random.choice(possible)
        return self.position
    
    def _coordinator_decision(
        self, 
        environment: Any, 
        shared_knowledge: Any
    ) -> Tuple[int, int]:
        """Decision logic for coordinator - balance exploration and collection."""
        # Balance 50/50 between exploration and collection
        if random.random() < 0.5:
            # Try collection first
            available = shared_knowledge.get_available_items()
            if available:
                # Don't claim the nearest - leave some for others
                items = list(available)
                if len(items) > 1:
                    # Take a random item, not necessarily the nearest
                    target = random.choice(items)
                    path = self.find_path(self.position, target, environment)
                    if path and len(path) > 1:
                        shared_knowledge.claim_item(target, self.id)
                        self.current_target = target
                        self.planned_path = path
                        return path[1]
        
        # Explore
        return self._explore(environment, shared_knowledge)
    
    def _find_and_claim_best_item(
        self,
        available_items: Set[Tuple[int, int]],
        environment: Any,
        shared_knowledge: Any
    ) -> Tuple[int, int]:
        """Find the best item based on strategy and claim it."""
        best_item = None
        best_path = None
        
        if self.strategy == AgentStrategy.GREEDY:
            # Always go for nearest
            shortest = float('inf')
            for item in available_items:
                path = self.find_path(self.position, item, environment)
                if path and len(path) < shortest:
                    shortest = len(path)
                    best_path = path
                    best_item = item
        
        elif self.strategy == AgentStrategy.COOPERATIVE:
            # Consider other agents' positions and targets
            # Try to minimize total team travel distance
            best_score = float('inf')
            for item in available_items:
                path = self.find_path(self.position, item, environment)
                if path:
                    # Score based on distance + isolation (further from other items is better to spread out)
                    score = len(path)
                    if score < best_score:
                        best_score = score
                        best_path = path
                        best_item = item
        
        else:  # BALANCED
            # Mix of greedy and exploration
            items = list(available_items)
            if items:
                # 70% chance nearest, 30% random
                if random.random() < 0.7:
                    shortest = float('inf')
                    for item in items:
                        path = self.find_path(self.position, item, environment)
                        if path and len(path) < shortest:
                            shortest = len(path)
                            best_path = path
                            best_item = item
                else:
                    target = random.choice(items)
                    best_path = self.find_path(self.position, target, environment)
                    best_item = target
        
        if best_path and len(best_path) > 1 and best_item:
            shared_knowledge.claim_item(best_item, self.id)
            self.current_target = best_item
            self.planned_path = best_path
            return best_path[1]
        
        return self._explore(environment, shared_knowledge)
    
    def _explore(
        self, 
        environment: Any, 
        shared_knowledge: Any
    ) -> Tuple[int, int]:
        """Explore unvisited or random cells."""
        possible = environment.get_possible_moves(self.position)
        if possible:
            unvisited = [m for m in possible if m not in shared_knowledge.visited]
            if unvisited:
                return random.choice(unvisited)
            return random.choice(possible)
        return self.position
    
    # ==================== COMMUNICATION ====================
    
    def send_message(
        self, 
        msg_type: str, 
        content: Any, 
        recipient_id: int = -1
    ) -> Dict:
        """
        Send a message (for message-passing cooperation).
        
        Args:
            msg_type: Type of message (e.g., 'item_found', 'claim', 'release')
            content: Message content
            recipient_id: Target agent ID (-1 for broadcast)
            
        Returns:
            Message dictionary
        """
        message = {
            'sender': self.id,
            'recipient': recipient_id,
            'type': msg_type,
            'content': content,
            'timestamp': len(self.sent_messages)
        }
        self.sent_messages.append(message)
        return message
    
    def receive_message(self, message: Dict) -> None:
        """Receive and store a message."""
        self.messages.append(message)
    
    def process_messages(self) -> List[Dict]:
        """Process pending messages and return actions."""
        actions = []
        for msg in self.messages:
            if msg['type'] == 'release' and msg['content'] == self.current_target:
                # Item we were targeting was released, we can claim it
                actions.append({'action': 'reclaim', 'target': msg['content']})
        self.messages.clear()
        return actions
    
    # ==================== BIDDING (Auction-based allocation) ====================
    
    def calculate_bid(
        self, 
        item_position: Tuple[int, int], 
        environment: Any
    ) -> float:
        """
        Calculate bid value for an item (lower is better, like distance).
        
        Args:
            item_position: Position of the item
            environment: Environment object
            
        Returns:
            Bid value (lower = higher priority)
        """
        path = self.find_path(self.position, item_position, environment)
        if not path:
            return float('inf')
        
        # Base bid is path length
        bid = len(path)
        
        # Adjust based on role
        if self.role == AgentRole.COLLECTOR:
            bid *= 0.8  # Collectors get priority
        elif self.role == AgentRole.EXPLORER:
            bid *= 1.2  # Explorers less interested in items
        
        # Adjust based on current workload
        if self.current_target is not None:
            bid *= 1.5  # Already busy
        
        # Adjust based on energy
        if self.max_energy > 0:
            energy_ratio = self.energy / self.max_energy
            if energy_ratio < 0.3:
                bid *= 2.0  # Low energy, less likely to win
        
        return bid
    
    def submit_bid(self, item_position: Tuple[int, int], bid: float) -> None:
        """Submit a bid for an item."""
        self.pending_bids[item_position] = bid
        self.current_bid = bid
    
    def win_bid(self, item_position: Tuple[int, int]) -> None:
        """Called when agent wins a bid."""
        self.current_target = item_position
        if item_position in self.pending_bids:
            del self.pending_bids[item_position]
    
    def lose_bid(self, item_position: Tuple[int, int]) -> None:
        """Called when agent loses a bid."""
        if item_position in self.pending_bids:
            del self.pending_bids[item_position]
        if self.current_target == item_position:
            self.current_target = None
            self.planned_path = []
    
    # ==================== STATUS & METRICS ====================
    
    def get_status(self) -> Dict:
        """Return agent's current status."""
        return {
            'id': self.id,
            'position': self.position,
            'collected_items': self.collected_items,
            'path_length': len(self.path) - 1,  # Exclude start
            'current_target': self.current_target,
            'algorithm': self.algorithm.value,
            'role': self.role.value,
            'strategy': self.strategy.value,
            'energy': int(self.energy) if self.max_energy > 0 else 'unlimited',
            'cells_explored': len(self.cells_explored)
        }
    
    def get_efficiency(self) -> float:
        """Calculate efficiency (items per move)."""
        moves = len(self.path) - 1
        if moves == 0:
            return 0.0
        return self.collected_items / moves
    
    def get_exploration_coverage(self, total_cells: int) -> float:
        """Calculate exploration coverage percentage."""
        return len(self.cells_explored) / max(total_cells, 1)
    
    def __repr__(self) -> str:
        return (f"Agent(id={self.id}, pos={self.position}, "
                f"items={self.collected_items}, role={self.role.value})")
