"""
Diagram templates for ML code visualization - clean version
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import os

def create_ml_pipeline_diagram(title="ML Pipeline", save_path=None):
    """
    Create ML pipeline diagram with proper branching arrows
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Colors
    colors = {
        'input': '#E3F2FD',
        'process': '#F3E5F5',
        'model': '#E8F5E9',
        'metric': '#FFEBEE',
        'warning': '#FFFFCC',
    }
    
    # Box settings
    box_w = 2.2
    box_h = 1.0
    
    # TOP ROW
    # Data Input
    ax.add_patch(Rectangle((0.5, 5.5), box_w, box_h, 
                          facecolor=colors['input'], edgecolor='black', linewidth=1.5))
    ax.text(1.6, 6, 'DATA INPUT\n5000 samples', ha='center', va='center', fontsize=10, weight='bold')
    
    # Processing
    ax.add_patch(Rectangle((3.5, 5.5), box_w, box_h, 
                          facecolor=colors['process'], edgecolor='black', linewidth=1.5))
    ax.text(4.6, 6, 'PROCESSING\n80/20 Split', ha='center', va='center', fontsize=10, weight='bold')
    
    # Evaluation
    ax.add_patch(Rectangle((6.5, 5.5), box_w, box_h, 
                          facecolor=colors['metric'], edgecolor='black', linewidth=1.5))
    ax.text(7.6, 6, 'EVALUATION\nMetrics', ha='center', va='center', fontsize=10, weight='bold')
    
    # Top row arrows
    ax.annotate('', xy=(3.5, 6), xytext=(2.7, 6),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(6.5, 6), xytext=(5.7, 6),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # MIDDLE ROW - Models
    # Logistic Regression
    ax.add_patch(Rectangle((0.5, 3), box_w, box_h, 
                          facecolor=colors['model'], edgecolor='black', linewidth=1.5))
    ax.text(1.6, 3.5, 'Logistic Reg\nRecall: 0.00', ha='center', va='center', fontsize=9)
    
    # Random Forest
    ax.add_patch(Rectangle((3.5, 3), box_w, box_h, 
                          facecolor=colors['model'], edgecolor='black', linewidth=1.5))
    ax.text(4.6, 3.5, 'Random Forest\nRecall: 0.05', ha='center', va='center', fontsize=9)
    
    # XGBoost
    ax.add_patch(Rectangle((6.5, 3), box_w, box_h, 
                          facecolor=colors['model'], edgecolor='black', linewidth=1.5))
    ax.text(7.6, 3.5, 'XGBoost\nRecall: 0.13', ha='center', va='center', fontsize=9)
    
    # Three separate arrows from Processing to each model
    # Arrow to Logistic Regression
    ax.annotate('', xy=(1.6, 4), xytext=(4.6, 5.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Arrow to Random Forest (straight down)
    ax.annotate('', xy=(4.6, 4), xytext=(4.6, 5.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Arrow to XGBoost
    ax.annotate('', xy=(7.6, 4), xytext=(4.6, 5.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    ax.text(4.8, 4.7, 'Train', fontsize=8, style='italic')
    
    # BOTTOM - Result
    ax.add_patch(Rectangle((2.5, 0.8), 4, 0.8, 
                          facecolor=colors['warning'], edgecolor='red', 
                          linewidth=2, linestyle='--', fill=True))
    ax.text(4.5, 1.2, 'Target: 85% Recall ❌', ha='center', va='center', 
            fontsize=11, weight='bold', color='red')
    
    # Arrows from models to result
    ax.annotate('', xy=(3.5, 1.6), xytext=(1.6, 3),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='red', alpha=0.7))
    ax.annotate('', xy=(4.5, 1.6), xytext=(4.6, 3),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='red', alpha=0.7))
    ax.annotate('', xy=(5.5, 1.6), xytext=(7.6, 3),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='red', alpha=0.7))
    
    # Side boxes
    ax.text(0.2, 4.5, '📋 Requirements:\n• Recall ≥ 85%\n• Precision ≥ 70%', 
            fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    ax.text(8.5, 4.5, '💡 Next Steps:\n• Tune hyperparameters\n• Handle imbalance\n• Try ensemble', 
            fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    # Title
    ax.text(4.5, 7.2, title, ha='center', va='center', fontsize=14, weight='bold')
    
    # Clean up
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(0, 7.8)
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Diagram saved to: {save_path}")
    
    plt.show()
    return fig

def create_simple_flow_diagram(title="Simple ML Flow", save_path=None):
    """
    Create simple linear flow with proper spacing
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 3))
    
    # Colors for each step
    colors = ['#E3F2FD', '#F3E5F5', '#E8F5E9', '#FFEBEE', '#FFFFCC']
    labels = ['Data\n5000', 'Split\n80/20', 'Train\n3 Models', 'Evaluate', 'Result: ❌']
    
    box_w = 1.8
    box_h = 1.0
    spacing = 2.5
    y = 1.5
    
    # Draw boxes
    for i, (label, color) in enumerate(zip(labels, colors)):
        x = 1 + i * spacing
        
        # Last box is dashed
        if i == len(labels) - 1:
            ax.add_patch(Rectangle((x, y), box_w, box_h, 
                                 facecolor=color, edgecolor='red', 
                                 linewidth=2, linestyle='--'))
        else:
            ax.add_patch(Rectangle((x, y), box_w, box_h, 
                                 facecolor=color, edgecolor='black', linewidth=1.5))
        
        ax.text(x + box_w/2, y + box_h/2, label, ha='center', va='center', 
                fontsize=10, weight='bold')
        
        # Arrow to next box
        if i < len(labels) - 1:
            arrow_color = 'red' if i == len(labels) - 2 else 'black'
            arrow_style = '--' if i == len(labels) - 2 else '-'
            ax.annotate('', xy=(x + spacing, y + box_h/2), 
                       xytext=(x + box_w, y + box_h/2),
                       arrowprops=dict(arrowstyle='->', lw=2, 
                                     color=arrow_color, linestyle=arrow_style))
    
    # Model details below
    ax.text(7, 0.8, 'LR: 0% | RF: 5% | XGB: 13% (Recall)', 
            ha='center', fontsize=9, style='italic')
    
    # Title
    ax.text(7, 3.2, title, ha='center', fontsize=14, weight='bold')
    
    ax.set_xlim(0, 14)
    ax.set_ylim(0.5, 3.5)
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Simple diagram saved to: {save_path}")
    
    plt.show()
    return fig

def create_code_flow_diagram(code_blocks, title="Code Flow"):
    """
    Create vertical code flow diagram
    """
    n_blocks = len(code_blocks)
    fig, ax = plt.subplots(1, 1, figsize=(8, n_blocks * 1.2 + 2))
    
    box_w = 6
    box_h = 0.8
    spacing = 1.0
    x = 1
    start_y = n_blocks * spacing
    
    for i, (label, code) in enumerate(code_blocks):
        y = start_y - i * spacing
        
        # Color coding
        if 'Need tuning' in code or '❌' in code:
            color = '#FFEBEE'
        else:
            color = '#E8F5E9' if i % 2 == 0 else '#E3F2FD'
        
        # Draw box
        ax.add_patch(Rectangle((x, y), box_w, box_h, 
                             facecolor=color, edgecolor='black', linewidth=1))
        
        # Add text
        ax.text(x + 0.1, y + box_h/2, f"{i+1}. {label}:", 
                fontsize=9, weight='bold', va='center')
        ax.text(x + 2, y + box_h/2, code, 
                fontsize=8, family='monospace', va='center')
        
        # Arrow to next
        if i < n_blocks - 1:
            ax.annotate('', xy=(x + box_w/2, y - 0.1), 
                       xytext=(x + box_w/2, y),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Title
    ax.text(x + box_w/2, start_y + 1.2, title, ha='center', fontsize=12, weight='bold')
    
    ax.set_xlim(0, 8)
    ax.set_ylim(-0.5, start_y + 2)
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    return fig
