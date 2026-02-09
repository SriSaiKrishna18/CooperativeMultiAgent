from environment import Environment
from agent import Agent
from cooperative_logic import CooperativeKnowledge, update_shared_knowledge
from utils import (print_simulation_status, generate_summary_report, 
                   plot_collection_progress, visualize_grid, plot_performance_metrics)

def run_simulation(num_steps=50, num_agents=2, verbose=True):
    """
    Run the improved cooperative multi-agent simulation with BFS pathfinding
    and comprehensive performance tracking.
    
    Args:
        num_steps: Maximum number of simulation steps
        num_agents: Number of cooperative agents (2-3 recommended)
        verbose: Enable detailed step-by-step output
    
    Returns:
        agents: List of Agent objects
        env: Environment object
        metrics: Dictionary of performance metrics
    """
    # Initialize environment
    env = Environment()
    total_items = env.total_items
    
    print(f"Environment initialized with {total_items} items to collect\n")
    
    # Define agent starting positions
    start_positions = [(0, 0), (4, 4), (2, 2)][:num_agents]
    
    # Create agents
    agents = [Agent(agent_id=i, start_position=pos) 
              for i, pos in enumerate(start_positions)]
    
    print(f"Deployed {num_agents} cooperative agents")
    for agent in agents:
        print(f"  Agent {agent.id} starting at {agent.position}")
    
    # Initialize shared knowledge
    shared_knowledge = CooperativeKnowledge()
    
    # Track collection progress
    collection_history = {agent.id: [] for agent in agents}
    
    # Run simulation
    print(f"\nStarting simulation (max {num_steps} steps)...\n")
    
    for step in range(num_steps):
        # Stop if all items are collected
        if env.items_remaining() == 0:
            print(f"\n🎉 Mission Complete! All items collected at step {step}!")
            break
        
        for agent in agents:
            # Agent chooses intelligent move using BFS and shared knowledge
            new_position = agent.choose_move(env, shared_knowledge)
            
            # Move agent
            agent.move(new_position)
            
            # Check and collect item
            if env.collect_item(new_position):
                agent.collect_item()
                shared_knowledge.mark_collected(new_position)
                if verbose:
                    print(f"  ✓ Agent {agent.id} collected item at {new_position}")
            
            # Update shared knowledge
            update_shared_knowledge(agent, env, shared_knowledge)
        
        # Record progress
        for agent in agents:
            collection_history[agent.id].append(agent.collected_items)
        
        # Print status periodically
        if verbose and step % 10 == 0:
            print_simulation_status(step, agents, env)
    
    final_step = step + 1
    
    # Generate comprehensive report
    generate_summary_report(agents, final_step, total_items)
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # 1. Grid and path visualization
    visualize_grid(env, agents, step_number=final_step)
    
    # 2. Collection progress over time
    plot_collection_progress(collection_history)
    
    # 3. Performance metrics
    plot_performance_metrics(agents, final_step, total_items)
    
    # Return results
    metrics = {
        'total_steps': final_step,
        'total_items': total_items,
        'items_collected': sum(agent.collected_items for agent in agents),
        'total_moves': sum(len(agent.path) - 1 for agent in agents),
        'collection_history': collection_history
    }
    
    return agents, env, metrics


if __name__ == "__main__":
    print("="*60)
    print("COOPERATIVE MULTI-AGENT SYSTEM SIMULATION")
    print("Enhanced with BFS Pathfinding & Performance Tracking")
    print("="*60 + "\n")
    
    # Run simulation
    agents, environment, metrics = run_simulation(
        num_steps=50, 
        num_agents=2, 
        verbose=True
    )
    
    print("\nSimulation complete! All visualizations displayed.")
