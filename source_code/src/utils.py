import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def visualize_grid(environment, agents, step_number=None):
    """
    Visualize the grid with agents' positions and paths.
    """
    grid_visual = environment.grid.copy().astype(float)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left plot: Current state
    for i, agent in enumerate(agents):
        x, y = agent.position
        grid_visual[x, y] = 3 + i  # Agents marked as 3, 4, 5, etc.
    
    im1 = ax1.imshow(grid_visual, cmap='tab10', interpolation='nearest', vmin=0, vmax=5)
    ax1.set_title(f'Grid State' + (f' - Step {step_number}' if step_number else ''), 
                  fontsize=14, fontweight='bold')
    
    # Add grid lines
    for x in range(environment.rows + 1):
        ax1.axhline(x - 0.5, color='gray', linewidth=0.5)
    for y in range(environment.cols + 1):
        ax1.axvline(y - 0.5, color='gray', linewidth=0.5)
    
    ax1.set_xticks(range(environment.cols))
    ax1.set_yticks(range(environment.rows))
    
    # Add legend
    legend_labels = ['Empty', 'Obstacle', 'Item'] + [f'Agent {i}' for i in range(len(agents))]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(legend_labels))]
    ax1.legend(handles[:len(legend_labels)], legend_labels, 
              loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # Right plot: Agent paths
    colors_agents = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, agent in enumerate(agents):
        path_array = np.array(agent.path)
        if len(path_array) > 0:
            ax2.plot(path_array[:, 1], path_array[:, 0], 
                    color=colors_agents[i % len(colors_agents)], 
                    linewidth=2, marker='o', markersize=3, 
                    label=f'Agent {i} ({len(agent.path)} steps)')
    
    # Show obstacles
    for i in range(environment.rows):
        for j in range(environment.cols):
            if environment.grid[i, j] == 1:
                ax2.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                             color='gray', alpha=0.5))
    
    ax2.set_xlim(-0.5, environment.cols - 0.5)
    ax2.set_ylim(environment.rows - 0.5, -0.5)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Agent Movement Paths', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('grid_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: grid_visualization.png")
    plt.close()


def plot_collection_progress(collection_history):
    """
    Plot items collected over time for each agent.
    """
    plt.figure(figsize=(12, 6))
    
    for agent_id, history in collection_history.items():
        plt.plot(history, label=f'Agent {agent_id}', 
                marker='o', linewidth=2, markersize=4)
    
    plt.xlabel('Simulation Step', fontsize=12)
    plt.ylabel('Total Items Collected', fontsize=12)
    plt.title('Item Collection Progress Over Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('collection_progress.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: collection_progress.png")
    plt.close()


def plot_performance_metrics(agents, total_steps, total_items):
    """
    Display key performance metrics in a bar chart.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    agent_ids = [agent.id for agent in agents]
    items_collected = [agent.collected_items for agent in agents]
    path_lengths = [len(agent.path) - 1 for agent in agents]
    
    # Plot 1: Items Collected
    axes[0, 0].bar(agent_ids, items_collected, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Items Collected per Agent', fontweight='bold')
    axes[0, 0].set_xlabel('Agent ID')
    axes[0, 0].set_ylabel('Items Collected')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Total Moves
    axes[0, 1].bar(agent_ids, path_lengths, color='lightcoral', edgecolor='black')
    axes[0, 1].set_title('Total Moves per Agent', fontweight='bold')
    axes[0, 1].set_xlabel('Agent ID')
    axes[0, 1].set_ylabel('Number of Moves')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Efficiency (Steps per Item)
    efficiency = [path_lengths[i] / max(items_collected[i], 1) for i in range(len(agents))]
    axes[1, 0].bar(agent_ids, efficiency, color='lightgreen', edgecolor='black')
    axes[1, 0].set_title('Efficiency (Steps per Item)', fontweight='bold')
    axes[1, 0].set_xlabel('Agent ID')
    axes[1, 0].set_ylabel('Steps per Item Collected')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Overall System Metrics
    metrics = ['Total Steps', 'Total Items', 'Avg Steps/Item']
    values = [total_steps, total_items, total_steps / max(total_items, 1)]
    axes[1, 1].bar(metrics, values, color=['orange', 'purple', 'gold'], edgecolor='black')
    axes[1, 1].set_title('System-Wide Metrics', fontweight='bold')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: performance_metrics.png")
    plt.close()


def print_simulation_status(step, agents, environment):
    """
    Print status of all agents and environment.
    """
    print(f"\n--- Step {step} ---")
    for agent in agents:
        status = agent.get_status()
        target_str = f", Target: {status['current_target']}" if status['current_target'] else ""
        print(f"Agent {status['id']}: Position {status['position']}, "
              f"Items: {status['collected_items']}{target_str}")
    print(f"Items Remaining: {environment.items_remaining()}")


def generate_summary_report(agents, total_steps, total_items):
    """
    Generate comprehensive simulation summary with performance analysis.
    """
    print("\n" + "="*60)
    print("SIMULATION COMPLETE - PERFORMANCE REPORT")
    print("="*60)
    
    total_collected = sum(agent.collected_items for agent in agents)
    total_moves = sum(len(agent.path) - 1 for agent in agents)
    
    for agent in agents:
        status = agent.get_status()
        efficiency = status['path_length'] / max(status['collected_items'], 1)
        print(f"\nAgent {status['id']}:")
        print(f"  Final Position: {status['position']}")
        print(f"  Items Collected: {status['collected_items']}")
        print(f"  Total Moves: {status['path_length'] - 1}")
        print(f"  Efficiency: {efficiency:.2f} steps/item")
    
    print(f"\n{'SYSTEM METRICS':^60}")
    print("-" * 60)
    print(f"  Total Items in Environment: {total_items}")
    print(f"  Total Items Collected: {total_collected}")
    print(f"  Collection Rate: {(total_collected/total_items)*100:.1f}%")
    print(f"  Total Simulation Steps: {total_steps}")
    print(f"  Total Agent Moves: {total_moves}")
    print(f"  Average Steps per Item: {total_moves/max(total_collected, 1):.2f}")
    print(f"  System Efficiency: {'Excellent' if total_moves/max(total_collected, 1) < 10 else 'Good' if total_moves/max(total_collected, 1) < 20 else 'Needs Improvement'}")
    print("="*60)
