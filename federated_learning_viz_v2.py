import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def create_federated_learning_animation():
    # Create figure with dark theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#1B1C20')
    ax.set_facecolor('#1B1C20')

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

    # Create points for frontend, coordinator, and workers
    # Frontend at top
    frontend_pos = (0, 0.8)
    # Coordinator in middle
    coordinator_pos = (0, 0.2)
    # Workers in arc at bottom
    n_workers = 5
    worker_y = -0.4  # Fixed y position for workers
    worker_x_spread = 1.0  # How far to spread workers horizontally
    worker_x = np.linspace(-worker_x_spread/2, worker_x_spread/2, n_workers)
    worker_pos = list(zip(worker_x, [worker_y]*n_workers))

    # Initialize plots without labels (no legend entries)
    frontend = ax.plot([frontend_pos[0]], [frontend_pos[1]], 'o', 
                      color='#FF9966', markersize=20)[0]
    coordinator = ax.plot([coordinator_pos[0]], [coordinator_pos[1]], 'o', 
                         color='#4A90E2', markersize=20)[0]
    workers = ax.plot([p[0] for p in worker_pos], [p[1] for p in worker_pos], 'o', 
                     color='#50C878', markersize=15)[0]
    
    # Add static labels
    ax.text(frontend_pos[0], frontend_pos[1]+0.1, 'Front End', 
            ha='center', va='center', color='white', fontsize=12)
    ax.text(coordinator_pos[0], coordinator_pos[1]+0.1, 'Coordinator', 
            ha='center', va='center', color='white', fontsize=12)
    for i, (x, y) in enumerate(worker_pos):
        ax.text(x, y-0.1, f'Worker {i+1}', 
                ha='center', va='center', color='white', fontsize=10)

    # Initialize arrows
    arrows_to_coordinator = []
    arrows_from_coordinator = []
    frontend_arrows = []

    # Frontend to coordinator arrows (direction flipped - now coordinator to frontend)
    frontend_arrows.append(ax.arrow(coordinator_pos[0], coordinator_pos[1]+0.05,
                                  0, frontend_pos[1]-coordinator_pos[1]-0.1,
                                  head_width=0.05, head_length=0.05,
                                  fc='#FF9966', ec='#FF9966', alpha=0))

    # Worker to coordinator arrows
    for x, y in worker_pos:
        # Calculate arrow vectors
        dx = coordinator_pos[0] - x
        dy = coordinator_pos[1] - y
        # Scale arrows to not touch nodes directly
        scale = 0.7
        arrow_to = ax.arrow(x, y+0.05, dx*scale, dy*scale-0.05,
                          head_width=0.05, head_length=0.05,
                          fc='#FF6B6B', ec='#FF6B6B', alpha=0)
        arrow_from = ax.arrow(coordinator_pos[0], coordinator_pos[1]-0.05,
                           -dx*scale, -dy*scale+0.05,
                           head_width=0.05, head_length=0.05,
                           fc='#4A90E2', ec='#4A90E2', alpha=0)
        arrows_to_coordinator.append(arrow_to)
        arrows_from_coordinator.append(arrow_from)

    # Add legend with appropriately sized markers
    ax.plot([], [], 'o', color='#FF9966', markersize=8, label='Front End')
    ax.plot([], [], 'o', color='#4A90E2', markersize=8, label='Coordinator')
    ax.plot([], [], 'o', color='#50C878', markersize=8, label='Workers')
    ax.plot([], [], '->', color='#FF6B6B', markersize=8, label='Local Updates')
    ax.plot([], [], '->', color='#4A90E2', markersize=8, label='Global Model')
    legend = ax.legend(loc='upper right', facecolor='#1B1C20', edgecolor='white', markerscale=1)
    plt.setp(legend.get_texts(), color='white')

    # Add title
    plt.title('Federated Learning System in Action', color='white', pad=20, fontsize=14)

    # Add description
    description = """
    Federated Learning Process:
    1. Coordinator distributes global model
    2. Workers train on local data
    3. Workers send model updates
    4. Coordinator aggregates updates
    5. Process repeats
    """
    ax.text(-1.1, -1.1, description, color='white', fontsize=10)

    def animate(frame):
        frame = frame % 100
        
        # Reset all arrows
        for arrow in arrows_to_coordinator + arrows_from_coordinator + frontend_arrows:
            arrow.set_alpha(0)
            
        # Phase 1: Local training (0-30)
        if frame < 30:
            alpha = 0.3 + 0.7 * abs(np.sin(frame * 0.2))
            workers.set_alpha(max(0.3, min(1.0, alpha)))
            coordinator.set_alpha(1.0)
            frontend.set_alpha(1.0)
            
        # Phase 2: Send updates to coordinator (31-60)
        elif frame < 60:
            workers.set_alpha(1.0)
            coordinator.set_alpha(1.0)
            frontend.set_alpha(1.0)
            progress = min(1.0, max(0.0, (frame - 30) / 30))
            for arrow in arrows_to_coordinator:
                arrow.set_alpha(progress)
            
        # Phase 3: Coordinator aggregation (61-80)
        elif frame < 80:
            workers.set_alpha(1.0)
            frontend.set_alpha(1.0)
            alpha = 0.3 + 0.7 * abs(np.sin(frame * 0.2))
            coordinator.set_alpha(max(0.3, min(1.0, alpha)))
            for arrow in frontend_arrows:
                arrow.set_alpha(alpha)
            
        # Phase 4: Send back to workers (81-100)
        else:
            workers.set_alpha(1.0)
            coordinator.set_alpha(1.0)
            frontend.set_alpha(1.0)
            progress = min(1.0, max(0.0, (frame - 80) / 20))
            for arrow in arrows_from_coordinator:
                arrow.set_alpha(progress)

        return [frontend, coordinator, workers] + arrows_to_coordinator + arrows_from_coordinator + frontend_arrows

    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=100, interval=50, blit=True)
    
    # Save as GIF
    anim.save('federated_learning_system_v2.gif', 
              writer='pillow', 
              fps=30, 
              savefig_kwargs={'facecolor': '#1B1C20'})
    plt.close()

if __name__ == "__main__":
    create_federated_learning_animation()