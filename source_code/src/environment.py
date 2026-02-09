import numpy as np

class Environment:
    """
    Represents the grid environment where agents operate.
    0 = Empty cell, 1 = Obstacle, 2 = Item
    """
    def __init__(self, grid=None):
        if grid is None:
            # Default grid setup
            self.grid = np.array([
                [0, 1, 0, 2, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [2, 0, 1, 1, 0],
                [0, 0, 0, 2, 0]
            ])
        else:
            self.grid = np.array(grid)
        
        self.rows, self.cols = self.grid.shape
        self.total_items = np.count_nonzero(self.grid == 2)
    
    def is_valid_position(self, pos):
        """Check if position is within bounds and not an obstacle."""
        x, y = pos
        return 0 <= x < self.rows and 0 <= y < self.cols and self.grid[x, y] != 1
    
    def get_cell(self, pos):
        """Get the value at a specific position."""
        x, y = pos
        if 0 <= x < self.rows and 0 <= y < self.cols:
            return self.grid[x, y]
        return None
    
    def collect_item(self, pos):
        """Remove item from position if it exists."""
        x, y = pos
        if self.grid[x, y] == 2:
            self.grid[x, y] = 0
            return True
        return False
    
    def get_possible_moves(self, pos):
        """Return all valid neighboring positions."""
        x, y = pos
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        possible = [(x + dx, y + dy) for dx, dy in directions]
        return [move for move in possible if self.is_valid_position(move)]
    
    def items_remaining(self):
        """Count remaining items in the grid."""
        return np.count_nonzero(self.grid == 2)
    
    def display_grid(self):
        """Print the current state of the grid."""
        print(self.grid)
    
    def find_all_items(self):
        """Return positions of all items in the grid."""
        items = []
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i, j] == 2:
                    items.append((i, j))
        return items
