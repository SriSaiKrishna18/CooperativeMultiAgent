# 🤖 Cooperative Multi-Agent System

## Overview

This project implements an **intelligent cooperative multi-agent system (MAS)** where autonomous agents work together to efficiently collect items in a 2D grid environment with obstacles. The system demonstrates key concepts in **artificial intelligence**, **pathfinding algorithms**, and **multi-agent coordination**.

### Key Features

✅ **Intelligent Pathfinding** - Agents use BFS (Breadth-First Search) to find optimal paths  
✅ **Task Allocation** - Prevents redundant work through item claiming mechanism  
✅ **Shared Knowledge** - Agents communicate visited cells and discovered items  
✅ **Performance Tracking** - Comprehensive metrics and visualizations  
✅ **Modular Design** - Clean, maintainable code structure

---

## 🎯 Problem Statement

**Environment:** A 5×5 grid containing:
- Empty cells (0) - free to traverse
- Obstacles (1) - cannot be crossed
- Items (2) - collectibles to be gathered

**Goal:** Multiple agents must cooperate to collect all items as efficiently as possible

**Challenges:**
- Obstacle avoidance
- Path optimization
- Workload distribution
- Communication overhead

---

## 🏗️ System Architecture

┌─────────────────────────────────────────────┐
│ SIMULATION ORCHESTRATION │
│ (main.py) │
└────────────────┬────────────────────────────┘
│
┌─────────┴─────────┐
│ │
┌──────▼──────┐ ┌───────▼────────┐
│ ENVIRONMENT │ │ VISUALIZATION │
│ - Grid mgmt │ │ - Charts │
│ - Obstacles │ │ - Paths │
│ - Items │ │ - Metrics │
└──────┬──────┘ └────────────────┘
│
│ Observes
│
┌──────▼─────────────────────────────────┐
│ AGENT LAYER │
│ ┌─────────┐ ┌─────────┐ │
│ │ Agent 0 │ │ Agent 1 │ │
│ │ - BFS │ │ - BFS │ │
│ │ - Move │ │ - Move │ │
│ └────┬────┘ └────┬────┘ │
└───────┼──────────────────┼────────────┘
│ │
└────────┬─────────┘
│ Share Knowledge
│
┌────────────▼──────────────┐
│ COOPERATION LAYER │
│ - Shared Memory │
│ - Task Allocation │
│ - Knowledge Sync │
└───────────────────────────┘


### Component Details

**Environment Module** (`environment.py`)
- Manages 5×5 grid world with obstacles and items
- Validates agent movements and position checks
- Handles item collection and state updates

**Agent Module** (`agent.py`)
- Implements autonomous agent behavior
- BFS pathfinding for optimal route calculation
- Intelligent target selection and movement execution

**Cooperation Module** (`cooperative_logic.py`)
- Shared knowledge base for all agents
- Item claiming mechanism to prevent conflicts
- Coordination and communication protocols

**Visualization Module** (`utils.py`)
- Real-time grid state visualization
- Performance metrics tracking
- Progress charts and summary reports

**Main Orchestrator** (`main.py`)
- Simulation initialization and execution
- Coordination between all components
- Results generation and reporting

---

## 📊 Performance Results

### Achieved Metrics

| Metric | Value | Rating |
|--------|-------|--------|
| **Total Steps to Complete** | 9 steps | Excellent |
| **Average Steps per Item** | 5.33 steps/item | Excellent |
| **Collection Rate** | 100% | Perfect |
| **Total Agent Moves** | 16 moves | Optimal |
| **System Efficiency** | Excellent | ✓ |

### Comparison with Baseline

| Strategy | Completion Steps | Efficiency | Improvement |
|----------|-----------------|------------|-------------|
| Random Movement (Baseline) | 49 steps | 16.3 steps/item | - |
| **BFS + Cooperation** | **9 steps** | **5.33 steps/item** | **82% faster** |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

Clone the repository
git clone <repository-url>
cd cooperative-multi-agent-system

Install required packages
pip install -r requirements.txt


### Quick Start

Run the simulation
python src/main.py


**Expected Output:**
- Console logs showing agent movements
- Three PNG images generated:
  - `grid_visualization.png` - Final grid state and paths
  - `collection_progress.png` - Items collected over time
  - `performance_metrics.png` - Efficiency analysis charts

---

## 📂 Project Structure

cooperative-multi-agent-system/
│
├── src/
│ ├── main.py # Main simulation script
│ ├── environment.py # Grid environment management
│ ├── agent.py # Agent class with BFS pathfinding
│ ├── cooperative_logic.py # Shared knowledge and coordination
│ └── utils.py # Visualization and reporting tools
│
├── screenshots/
│ ├── grid_visualization.png # Grid state and agent paths
│ ├── collection_progress.png # Collection progress chart
│ ├── performance_metrics.png # Performance metrics
│ └── console_output.png # Terminal output
│
├── README.md # This file
├── requirements.txt # Python dependencies
└── LICENSE # Project license


---

## 🧠 Algorithm Details

### Breadth-First Search (BFS)

**Why BFS?**
- Guarantees shortest path in unweighted graphs
- Time Complexity: O(V + E) where V=cells, E=edges
- Space Complexity: O(V)
- Optimal for small grid environments (5×5)

**Implementation Highlights:**
def bfs_path(self, start, goal, environment):
"""Find shortest path using BFS"""
queue = deque([(start, [start])])
visited = {start}

while queue:
    current, path = queue.popleft()
    if current == goal:
        return path
    
    for neighbor in environment.get_possible_moves(current):
        if neighbor not in visited:
            visited.add(neighbor)
            queue.append((neighbor, path + [neighbor]))

return []

### Cooperation Strategy

1. **Shared Knowledge Base**
   - All agents contribute to centralized memory
   - Tracks visited cells, known items, collected items

2. **Task Allocation**
   - Agents "claim" items before pursuing them
   - Prevents multiple agents from targeting same item
   - Ensures efficient workload distribution

3. **Decision Priority**
   - Follow planned path to claimed item
   - Find and claim nearest available item
   - Explore unvisited cells
   - Random movement (fallback)

---

## 🎨 Customization

### Modify Grid Environment

Edit `environment.py`:
self.grid = np.array(, # 0=Empty, 1=Obstacle, 2=Item​​
,​
,​
,​​
​
])

### Adjust Simulation Parameters

In `main.py`:
agents, environment, metrics = run_simulation(
num_steps=50, # Maximum simulation steps
num_agents=2, # Number of cooperative agents (2-3 recommended)
verbose=True # Enable detailed console output
)


### Change Agent Starting Positions

In `main.py`:
start_positions = [(0, 0), (4, 4), (2, 2)][:num_agents]



---

## 📈 Visualizations

The system generates three comprehensive visualizations:

### 1. Grid State and Agent Paths
- Shows final positions of all agents
- Complete movement trajectories color-coded by agent
- Obstacles and items clearly marked

### 2. Collection Progress Over Time
- Line chart showing cumulative items collected
- Separate lines for each agent
- Demonstrates collection efficiency

### 3. Performance Metrics Dashboard
- Items collected per agent (bar chart)
- Total moves per agent (bar chart)
- Efficiency: steps per item (bar chart)
- System-wide metrics summary (bar chart)

---

## 🔧 Future Enhancements

### Short-term
- [ ] Dynamic load balancing based on current workload
- [ ] Larger grid environments (10×10, 20×20)
- [ ] A* pathfinding with heuristics
- [ ] Partial observability (limited agent vision)

### Long-term
- [ ] Reinforcement learning for adaptive behavior
- [ ] Dynamic obstacles (moving barriers)
- [ ] Heterogeneous agents (different capabilities)
- [ ] Distributed coordination (message-passing)
- [ ] 3D environments
- [ ] Real-world robot deployment

---

## 📚 Educational Value

This project demonstrates:

✓ **Graph Algorithms** - BFS implementation and complexity analysis  
✓ **Multi-Agent Systems** - Coordination and cooperation strategies  
✓ **Software Engineering** - Modular, maintainable code architecture  
✓ **Data Visualization** - Performance analysis and reporting  
✓ **Algorithm Optimization** - Efficiency improvements through intelligent design

**Suitable for:**
- AI/ML coursework and assignments
- Algorithm study and implementation practice
- System design and architecture learning
- Portfolio projects for job applications

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional pathfinding algorithms (A*, Dijkstra)
- New cooperation strategies
- Enhanced visualizations
- Performance optimizations
- Documentation improvements
- Test coverage

**To contribute:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- BFS algorithm implementation inspired by classic graph theory
- Multi-agent coordination concepts from distributed AI research
- Visualization design using Matplotlib best practices

---

## 📧 Contact

For questions, issues, or suggestions:

- Open an issue on GitHub
- Email: [your-email@example.com]
- Project Link: [https://github.com/yourusername/cooperative-multi-agent-system](https://github.com/yourusername/repo)

---

## 📖 References

1. **Breadth-First Search**: Cormen, T. H., et al. "Introduction to Algorithms" (4th Edition)
2. **Multi-Agent Systems**: Wooldridge, M. "An Introduction to MultiAgent Systems" (2nd Edition)
3. **Cooperative AI**: Russell, S., Norvig, P. "Artificial Intelligence: A Modern Approach" (4th Edition)

---

**Built with ❤️ for learning and exploration in multi-agent systems**

