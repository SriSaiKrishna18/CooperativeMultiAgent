"""
Main Simulation Orchestrator for Cooperative Multi-Agent System.
Enhanced with benchmark mode, multiple algorithms, and comprehensive reporting.
"""
import os
import sys
import logging
from typing import List, Dict, Tuple, Optional, Any

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (Config, DEFAULT_CONFIG, SMALL_GRID_CONFIG, LARGE_GRID_CONFIG,
                   PathfindingAlgorithm, AgentRole, AgentStrategy, AllocationMethod,
                   GridConfig, AgentConfig, CooperationConfig, SimulationConfig)
from environment import Environment, CellType
from agent import Agent
from cooperative_logic import (CooperativeKnowledge, update_shared_knowledge,
                               create_auction_round)
from utils import (Visualizer, print_simulation_status, generate_summary_report,
                   setup_logging, get_timestamp)


logger = logging.getLogger(__name__)


def run_simulation(
    config: Config = None,
    verbose: bool = True
) -> Tuple[List[Agent], Environment, Dict[str, Any]]:
    """
    Run the enhanced cooperative multi-agent simulation.
    
    Args:
        config: Configuration object (uses DEFAULT_CONFIG if None)
        verbose: Enable detailed console output
        
    Returns:
        Tuple of (agents, environment, metrics)
    """
    config = config or DEFAULT_CONFIG
    
    # Setup logging
    setup_logging(config.simulation.logging_level)
    
    # Initialize environment
    env = Environment(config.grid)
    total_items = env.total_items
    
    print("=" * 70)
    print("  🤖 COOPERATIVE MULTI-AGENT SYSTEM SIMULATION")
    print("  Enhanced with A*/BFS/Dijkstra, Roles, and Advanced Cooperation")
    print("=" * 70)
    print(f"\n📊 Environment: {env.rows}x{env.cols} grid, {total_items} items, "
          f"{config.agent.num_agents} agents")
    print(f"🔧 Algorithm: {config.agent.default_algorithm.value}, "
          f"Allocation: {config.cooperation.allocation_method.value}")
    
    # Define agent starting positions
    num_agents = config.agent.num_agents
    if config.agent.start_positions:
        start_positions = config.agent.start_positions[:num_agents]
    else:
        # Default corners and edges
        corners = [
            (0, 0), 
            (env.rows - 1, env.cols - 1),
            (0, env.cols - 1),
            (env.rows - 1, 0),
            (env.rows // 2, 0),
            (env.rows // 2, env.cols - 1)
        ]
        start_positions = corners[:num_agents]
    
    # Create agents with diverse roles
    roles = [AgentRole.COLLECTOR, AgentRole.EXPLORER, AgentRole.COORDINATOR]
    agents = []
    for i, pos in enumerate(start_positions):
        agent = Agent(
            agent_id=i,
            start_position=pos,
            algorithm=config.agent.default_algorithm,
            role=roles[i % len(roles)],
            strategy=config.agent.default_strategy,
            max_energy=config.agent.max_energy,
            energy_per_move=config.agent.energy_per_move,
            vision_radius=config.agent.vision_radius
        )
        agents.append(agent)
    
    print(f"\n🚀 Deployed {num_agents} agents:")
    for agent in agents:
        print(f"   Agent {agent.id}: Position {agent.position}, "
              f"Role: {agent.role.value}, Algorithm: {agent.algorithm.value}")
    
    # Initialize shared knowledge
    shared_knowledge = CooperativeKnowledge(config.cooperation)
    
    # Setup zone-based allocation if configured
    if config.cooperation.zone_assignment:
        shared_knowledge.assign_zones(env, agents, method='vertical')
    
    # Initialize visualizer
    visualizer = Visualizer(
        output_dir=config.simulation.output_directory,
        dpi=config.visualization.figure_dpi,
        animation_fps=config.visualization.animation_fps
    )
    
    # Track collection progress
    collection_history = {agent.id: [] for agent in agents}
    
    # Capture initial frame
    if config.simulation.generate_animations:
        visualizer.capture_frame(0, env, agents)
    
    # Run simulation
    print(f"\n⏱️  Starting simulation (max {config.simulation.max_steps} steps)...\n")
    
    for step in range(config.simulation.max_steps):
        # Stop if all items are collected
        if env.items_remaining() == 0:
            print(f"\n🎉 Mission Complete! All items collected at step {step}!")
            break
        
        # Process auctions if using auction allocation
        if config.cooperation.allocation_method == AllocationMethod.AUCTION:
            available = shared_knowledge.get_available_items()
            if available:
                for item in available:
                    for agent in agents:
                        bid = agent.calculate_bid(item, env)
                        if bid < float('inf'):
                            shared_knowledge._start_auction(item, agent.id, bid)
                shared_knowledge.process_auctions(agents)
        
        # Deliver messages
        if config.cooperation.enable_communication:
            shared_knowledge.deliver_all_messages(agents)
        
        # Each agent takes a turn
        for agent in agents:
            # Agent chooses move
            new_position = agent.choose_move(env, shared_knowledge)
            
            # Move agent
            agent.move(new_position, step)
            
            # Check and collect item
            if env.collect_item(new_position, agent.id, step):
                agent.collect_item(step)
                shared_knowledge.mark_collected(new_position, agent.id)
                if verbose:
                    print(f"  ✅ Agent {agent.id} collected item at {new_position}")
            
            # Update shared knowledge
            update_shared_knowledge(agent, env, shared_knowledge)
        
        # Advance cooperation step
        shared_knowledge.step()
        
        # Record progress
        for agent in agents:
            collection_history[agent.id].append(agent.collected_items)
        
        # Capture animation frame
        if config.simulation.generate_animations:
            visualizer.capture_frame(step + 1, env, agents)
        
        # Print status periodically
        if verbose and step % max(1, config.simulation.max_steps // 10) == 0:
            print_simulation_status(step, agents, env)
    
    final_step = step + 1
    
    # Generate comprehensive report
    summary = generate_summary_report(agents, final_step, total_items, shared_knowledge)
    
    # Generate visualizations
    if config.simulation.save_visualizations:
        print("\n📊 Generating visualizations...")
        
        # Grid visualization
        visualizer.visualize_grid(env, agents, final_step)
        
        # Collection progress
        visualizer.plot_collection_progress(collection_history)
        
        # Performance metrics
        visualizer.plot_performance_metrics(agents, final_step, total_items)
        
        # Heatmap
        if config.visualization.show_heatmap:
            visualizer.plot_heatmap(env)
        
        # Animation
        if config.simulation.generate_animations and len(visualizer.frames) > 1:
            visualizer.create_animation(env)
        
        print(f"✅ Visualizations saved to '{config.simulation.output_directory}/' directory")
    
    # Compile metrics
    metrics = {
        'total_steps': final_step,
        'total_items': total_items,
        'items_collected': sum(a.collected_items for a in agents),
        'total_moves': sum(len(a.path) - 1 for a in agents),
        'collection_history': collection_history,
        'collection_rate': summary['collection_rate'],
        'avg_efficiency': summary['avg_efficiency'],
        'rating': summary['rating'],
        'cooperation_stats': shared_knowledge.get_statistics(),
        'environment_stats': env.get_statistics()
    }
    
    return agents, env, metrics


def run_benchmark(
    grid_config: GridConfig = None,
    num_agents: int = 2,
    algorithms: List[PathfindingAlgorithm] = None
) -> Dict[str, Dict[str, float]]:
    """
    Run benchmark comparing different algorithms.
    
    Args:
        grid_config: Grid configuration
        num_agents: Number of agents
        algorithms: List of algorithms to compare
        
    Returns:
        Comparison data for visualization
    """
    if algorithms is None:
        algorithms = [
            PathfindingAlgorithm.BFS,
            PathfindingAlgorithm.ASTAR,
            PathfindingAlgorithm.DIJKSTRA
        ]
    
    print("\n" + "=" * 70)
    print("  📊 ALGORITHM BENCHMARK MODE")
    print("=" * 70)
    
    results = {}
    
    for algo in algorithms:
        print(f"\n🔧 Testing {algo.value.upper()}...")
        
        config = Config(
            grid=grid_config or SMALL_GRID_CONFIG.grid,
            agent=AgentConfig(
                num_agents=num_agents,
                default_algorithm=algo
            ),
            simulation=SimulationConfig(
                verbose=False,
                save_visualizations=False,
                generate_animations=False
            )
        )
        
        try:
            agents, env, metrics = run_simulation(config, verbose=False)
            
            results[algo.value] = {
                'steps': metrics['total_steps'],
                'efficiency': metrics['items_collected'] / max(metrics['total_moves'], 1),
                'exploration': sum(len(a.cells_explored) for a in agents) / 
                              (env.rows * env.cols) * 100
            }
            
            print(f"   Steps: {metrics['total_steps']}, "
                  f"Efficiency: {results[algo.value]['efficiency']:.3f}")
        
        except Exception as e:
            logger.error(f"Benchmark failed for {algo.value}: {e}")
            results[algo.value] = {'steps': float('inf'), 'efficiency': 0, 'exploration': 0}
    
    # Generate comparison visualization
    visualizer = Visualizer()
    visualizer.plot_algorithm_comparison(results)
    
    print(f"\n✅ Benchmark complete! Results saved to 'images/algorithm_comparison.png'")
    
    return results


def main():
    """Main entry point with CLI-like interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Cooperative Multi-Agent System Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run with default settings
  python main.py --agents 3         # Run with 3 agents
  python main.py --grid 10          # Use 10x10 grid
  python main.py --algorithm astar  # Use A* pathfinding
  python main.py --benchmark        # Run algorithm benchmark
  python main.py --scenario maze    # Use predefined maze scenario
        """
    )
    
    parser.add_argument('--agents', type=int, default=2,
                       help='Number of agents (default: 2)')
    parser.add_argument('--grid', type=int, default=5,
                       help='Grid size NxN (default: 5)')
    parser.add_argument('--steps', type=int, default=50,
                       help='Maximum simulation steps (default: 50)')
    parser.add_argument('--algorithm', choices=['bfs', 'astar', 'dijkstra'],
                       default='astar', help='Pathfinding algorithm (default: astar)')
    parser.add_argument('--allocation', choices=['simple_claim', 'auction', 'priority'],
                       default='auction', help='Task allocation method (default: auction)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run algorithm benchmark')
    parser.add_argument('--scenario', choices=['simple', 'maze', 'open', 'clustered'],
                       help='Use predefined scenario')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    parser.add_argument('--no-animation', action='store_true',
                       help='Skip animation generation')
    
    args = parser.parse_args()
    
    if args.benchmark:
        grid_config = GridConfig(rows=args.grid, cols=args.grid)
        run_benchmark(grid_config, args.agents)
        return
    
    # Build configuration from arguments
    algo_map = {
        'bfs': PathfindingAlgorithm.BFS,
        'astar': PathfindingAlgorithm.ASTAR,
        'dijkstra': PathfindingAlgorithm.DIJKSTRA
    }
    
    alloc_map = {
        'simple_claim': AllocationMethod.SIMPLE_CLAIM,
        'auction': AllocationMethod.AUCTION,
        'priority': AllocationMethod.PRIORITY
    }
    
    if args.scenario:
        env = Environment.from_scenario(args.scenario)
        grid_config = GridConfig(
            rows=env.rows, cols=env.cols,
            use_predefined=True,
            predefined_layout=env.grid.tolist()
        )
    else:
        grid_config = GridConfig(rows=args.grid, cols=args.grid)
    
    config = Config(
        grid=grid_config,
        agent=AgentConfig(
            num_agents=args.agents,
            default_algorithm=algo_map[args.algorithm]
        ),
        cooperation=CooperationConfig(
            allocation_method=alloc_map[args.allocation]
        ),
        simulation=SimulationConfig(
            max_steps=args.steps,
            verbose=not args.quiet,
            generate_animations=not args.no_animation
        )
    )
    
    # Run simulation
    agents, env, metrics = run_simulation(config, verbose=not args.quiet)
    
    print(f"\n✅ Simulation complete!")
    print(f"   Final Score: {metrics['items_collected']}/{metrics['total_items']} items "
          f"in {metrics['total_steps']} steps")
    print(f"   Efficiency Rating: {metrics['rating']}")


if __name__ == "__main__":
    main()
