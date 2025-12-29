"""
Create a LeNet-style diagram showing the four model architectures:
Linear, MLP, Self-Attention, and Transformer
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# Set up the figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Architecture Comparison', fontsize=16, fontweight='bold', y=0.98)

# Colors
embed_color = '#E8F4F8'
linear_color = '#FFE5B4'
mlp_color = '#D4EDDA'
attn_color = '#FFF3CD'
output_color = '#F8D7DA'

def draw_box(ax, x, y, width, height, label, color='lightblue', text_size=9):
    """Draw a box with text"""
    box = FancyBboxPatch((x, y), width, height,
                        boxstyle="round,pad=0.02", 
                        edgecolor='black', 
                        facecolor=color,
                        linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, label, 
            ha='center', va='center', fontsize=text_size, fontweight='bold')
    return box

def draw_arrow(ax, x1, y1, x2, y2, color='black', style='->'):
    """Draw an arrow"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style, 
                           color=color, 
                           linewidth=1.5,
                           mutation_scale=20)
    ax.add_patch(arrow)

# ========== LINEAR MODEL ==========
ax = axes[0, 0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('Linear Model', fontsize=12, fontweight='bold', pad=10)

# Input embeddings
draw_box(ax, 1, 7, 8, 0.8, 'Input Embeddings\n(T×128)', embed_color)
draw_arrow(ax, 5, 7, 5, 6.2)

# Flatten
draw_box(ax, 1, 5.5, 8, 0.6, 'Flatten\n(T×128 → T×128)', linear_color, 8)
draw_arrow(ax, 5, 5.5, 5, 4.7)

# Linear layer
draw_box(ax, 2, 3.5, 6, 1, 'Linear Layer\n(T×128 → Vocab)', linear_color)
draw_arrow(ax, 5, 3.5, 5, 2.5)

# Output
draw_box(ax, 3, 1, 4, 0.8, 'Output Logits\n(Vocab)', output_color)

# ========== MLP MODEL ==========
ax = axes[0, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('MLP Model', fontsize=12, fontweight='bold', pad=10)

# Input embeddings
draw_box(ax, 1, 7, 8, 0.8, 'Input Embeddings\n(T×128)', embed_color)
draw_arrow(ax, 5, 7, 5, 6.2)

# Flatten
draw_box(ax, 1, 5.5, 8, 0.6, 'Flatten\n(T×128 → T×128)', linear_color, 8)
draw_arrow(ax, 5, 5.5, 5, 4.7)

# Hidden layer 1
draw_box(ax, 2, 4, 6, 0.7, 'Linear + ReLU\n(T×128 → 256)', mlp_color, 8)
draw_arrow(ax, 5, 4, 5, 3.3)

# Dropout
draw_box(ax, 3.5, 2.8, 3, 0.5, 'Dropout', mlp_color, 8)
draw_arrow(ax, 5, 2.8, 5, 2.3)

# Hidden layer 2
draw_box(ax, 2, 1.5, 6, 0.7, 'Linear + ReLU\n(256 → 256)', mlp_color, 8)
draw_arrow(ax, 5, 1.5, 5, 0.7)

# Output
draw_box(ax, 3, 0, 4, 0.6, 'Output\n(Vocab)', output_color, 8)

# ========== SELF-ATTENTION MODEL ==========
ax = axes[1, 0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('Self-Attention Model', fontsize=12, fontweight='bold', pad=10)

# Input embeddings
draw_box(ax, 1, 8.5, 8, 0.6, 'Token Embeddings\n(T×128)', embed_color, 8)
draw_arrow(ax, 5, 8.5, 5, 7.9)

# Positional embeddings
draw_box(ax, 1, 7.2, 8, 0.6, 'Positional Embeddings\n(T×128)', embed_color, 8)
draw_arrow(ax, 5, 7.2, 5, 6.6)

# Add
draw_box(ax, 3, 6, 4, 0.5, 'Add', attn_color, 8)
draw_arrow(ax, 5, 6, 5, 5.5)

# Self-attention block
draw_box(ax, 1.5, 4.5, 7, 1, 'Multi-Head Self-Attention\n(4 heads, causal mask)', attn_color, 8)
draw_arrow(ax, 5, 4.5, 5, 3.5)

# Feed-forward
draw_box(ax, 2, 2.5, 6, 0.8, 'Feed-Forward Network\n(128 → 256 → 128)', attn_color, 8)
draw_arrow(ax, 5, 2.5, 5, 1.7)

# Final time step
draw_box(ax, 3, 1, 4, 0.6, 'Final Time Step', attn_color, 8)
draw_arrow(ax, 5, 1, 5, 0.4)

# Output
draw_box(ax, 3.5, 0, 3, 0.3, 'Output\n(Vocab)', output_color, 7)

# ========== TRANSFORMER MODEL ==========
ax = axes[1, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('Transformer Model (3 Layers)', fontsize=12, fontweight='bold', pad=10)

# Input embeddings
draw_box(ax, 1, 8.5, 8, 0.5, 'Token + Positional Embeddings\n(T×128)', embed_color, 8)
draw_arrow(ax, 5, 8.5, 5, 8)

# Transformer Block 1
draw_box(ax, 1.5, 6.8, 7, 0.9, 'Transformer Block 1\n(Attention + FFN)', attn_color, 8)
draw_arrow(ax, 5, 6.8, 5, 5.9)

# Transformer Block 2
draw_box(ax, 1.5, 5, 7, 0.9, 'Transformer Block 2\n(Attention + FFN)', attn_color, 8)
draw_arrow(ax, 5, 5, 5, 4.1)

# Transformer Block 3
draw_box(ax, 1.5, 3.2, 7, 0.9, 'Transformer Block 3\n(Attention + FFN)', attn_color, 8)
draw_arrow(ax, 5, 3.2, 5, 2.3)

# Final time step
draw_box(ax, 3, 1.5, 4, 0.6, 'Final Time Step', attn_color, 8)
draw_arrow(ax, 5, 1.5, 5, 0.9)

# Output
draw_box(ax, 3.5, 0.2, 3, 0.6, 'Output\n(Vocab)', output_color, 8)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('images/model_architectures.png', dpi=300, bbox_inches='tight')
print("Architecture diagram saved to images/model_architectures.png")

