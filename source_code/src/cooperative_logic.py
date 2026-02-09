class CooperativeKnowledge:
    """
    Enhanced shared knowledge base with task allocation.
    Prevents redundant work through item claiming mechanism.
    """
    def __init__(self):
        self.visited = set()  # Cells visited by any agent
        self.known_items = set()  # Positions of known items
        self.collected_items = set()  # Positions of collected items
        self.claimed_items = {}  # {item_position: agent_id} - task allocation
    
    def mark_visited(self, position):
        """Mark a position as visited."""
        self.visited.add(position)
    
    def add_known_item(self, position):
        """Add a newly discovered item location."""
        if position not in self.collected_items:
            self.known_items.add(position)
    
    def mark_collected(self, position):
        """Mark an item as collected."""
        if position in self.known_items:
            self.known_items.remove(position)
        self.collected_items.add(position)
        if position in self.claimed_items:
            del self.claimed_items[position]
    
    def claim_item(self, position, agent_id):
        """Agent claims an item to prevent others from targeting it."""
        if position in self.known_items:
            self.claimed_items[position] = agent_id
    
    def get_available_items(self):
        """Get items that are known but not yet claimed."""
        return self.known_items - set(self.claimed_items.keys())
    
    def get_shared_knowledge(self):
        """Return dictionary of all shared knowledge."""
        return {
            'visited': self.visited,
            'known_items': self.known_items,
            'collected_items': self.collected_items,
            'claimed_items': self.claimed_items
        }
    
    def is_visited(self, position):
        """Check if position has been visited."""
        return position in self.visited


def update_shared_knowledge(agent, environment, shared_knowledge):
    """
    Update shared knowledge based on agent's observations.
    Agents share information about visited cells and discovered items.
    """
    pos = agent.position
    shared_knowledge.mark_visited(pos)
    
    # Check current cell for item
    if environment.get_cell(pos) == 2:
        shared_knowledge.add_known_item(pos)
    
    # Share observations of neighboring cells
    for neighbor in environment.get_possible_moves(pos):
        if environment.get_cell(neighbor) == 2:
            shared_knowledge.add_known_item(neighbor)
    
    # Discover all items initially (simulating shared vision)
    for item_pos in environment.find_all_items():
        shared_knowledge.add_known_item(item_pos)
