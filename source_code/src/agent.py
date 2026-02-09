import random
from collections import deque

class Agent:
    """
    Represents an autonomous agent with intelligent pathfinding capabilities.
    """
    def __init__(self, agent_id, start_position):
        self.id = agent_id
        self.position = start_position
        self.collected_items = 0
        self.path = [start_position]  # Track movement history
        self.current_target = None  # Current item being pursued
        self.planned_path = []  # Path to current target
    
    def move(self, new_position):
        """Move agent to a new position."""
        self.position = new_position
        self.path.append(new_position)
    
    def collect_item(self):
        """Increment collected items counter."""
        self.collected_items += 1
        self.current_target = None
        self.planned_path = []
    
    def bfs_path(self, start, goal, environment):
        """
        Use BFS to find shortest path from start to goal.
        Returns list of positions representing the path.
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
        
        return []  # No path found
    
    def choose_move(self, environment, shared_knowledge):
        """
        Intelligent move selection using BFS pathfinding.
        Prioritizes known unclaimed items, then exploration.
        """
        # If following a planned path, continue on it
        if self.planned_path and len(self.planned_path) > 1:
            next_pos = self.planned_path[1]
            self.planned_path = self.planned_path[1:]
            return next_pos
        
        # Find nearest unclaimed item
        available_items = shared_knowledge.get_available_items()
        
        if available_items:
            # Find closest item using BFS
            best_item = None
            best_path = None
            shortest_distance = float('inf')
            
            for item in available_items:
                path = self.bfs_path(self.position, item, environment)
                if path and len(path) < shortest_distance:
                    shortest_distance = len(path)
                    best_path = path
                    best_item = item
            
            if best_path and len(best_path) > 1:
                # Claim the item
                shared_knowledge.claim_item(best_item, self.id)
                self.current_target = best_item
                self.planned_path = best_path
                return best_path[1]
        
        # No items available - explore unvisited cells
        possible_moves = environment.get_possible_moves(self.position)
        if possible_moves:
            unvisited = [move for move in possible_moves 
                        if move not in shared_knowledge.visited]
            
            if unvisited:
                return random.choice(unvisited)
            else:
                return random.choice(possible_moves)
        
        return self.position
    
    def get_status(self):
        """Return agent's current status."""
        return {
            'id': self.id,
            'position': self.position,
            'collected_items': self.collected_items,
            'path_length': len(self.path),
            'current_target': self.current_target
        }
