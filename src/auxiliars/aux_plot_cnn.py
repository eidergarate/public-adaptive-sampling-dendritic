# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:51:35 2024

@author: egarate
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def draw_box(ax, center, text, width=1.5, height=0.5, color='skyblue'):
    """Draw a box with text"""
    rect = FancyBboxPatch((center[0] - width / 2, center[1] - height / 2), width, height, 
                          boxstyle="round,pad=0.3", edgecolor='black', facecolor=color)
    ax.add_patch(rect)
    ax.text(center[0], center[1], text, ha="center", va="center", fontsize=10)

def plot_cnn_structure():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    
    # Input layers
    draw_box(ax, (0, 10), "Input: Temp Images", color='lightgreen')
    draw_box(ax, (2, 10), "Input: OP Images", color='lightgreen')
    draw_box(ax, (4, 10), "Input: Tabular Data", color='lightgreen')

    # Temp and OP image processing branches
    draw_box(ax, (0, 8), "Conv2D\nMaxPooling\n(Temp)", color='skyblue')
    draw_box(ax, (2, 8), "Conv2D\nMaxPooling\n(OP)", color='skyblue')

    # Concatenation and processing
    draw_box(ax, (1, 6.5), "Concat Temp & OP", color='orange')
    draw_box(ax, (1, 5), "Conv2D\nUpSampling\n(Combined Images)", color='skyblue')

    # Tabular data processing
    draw_box(ax, (4, 8), "Dense -> Dense\n(Tabular)", color='skyblue')
    draw_box(ax, (4, 6), "Reshape\nUpSampling\n(Tabular)", color='skyblue')

    # All data combination and processing
    draw_box(ax, (2.5, 3.5), "Concat All Data\n-> Conv2D", color='orange')

    # Output
    draw_box(ax, (2.5, 1.5), "Conv2DTranspose\n(Output)", color='lightcoral')

    # Connections
    ax.arrow(0, 9.5, 0, -0.8, head_width=0.1, head_length=0.2, fc='black', ec='black')
    ax.arrow(2, 9.5, 0, -0.8, head_width=0.1, head_length=0.2, fc='black', ec='black')
    ax.arrow(4, 9.5, 0, -0.8, head_width=0.1, head_length=0.2, fc='black', ec='black')

    ax.arrow(0, 7.5, 0.7, -0.4, head_width=0.1, head_length=0.2, fc='black', ec='black')
    ax.arrow(2, 7.5, -0.7, -0.4, head_width=0.1, head_length=0.2, fc='black', ec='black')
    ax.arrow(4, 7.5, 0, -0.8, head_width=0.1, head_length=0.2, fc='black', ec='black')

    ax.arrow(1, 6, 0, -0.3, head_width=0.1, head_length=0.2, fc='black', ec='black')
    ax.arrow(1, 4.4, 1.25, -0.35, head_width=0.1, head_length=0.2, fc='black', ec='black')
    ax.arrow(4, 5.5, -1.25, -1.3, head_width=0.1, head_length=0.2, fc='black', ec='black')

    ax.arrow(2.5, 3, 0, -0.8, head_width=0.1, head_length=0.2, fc='black', ec='black')

    plt.show()



# Call the function
plot_cnn_structure()
