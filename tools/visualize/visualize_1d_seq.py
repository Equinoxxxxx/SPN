import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def generate_colormap_legend(colormap=cv2.COLORMAP_JET, width=100, height=20):
    """
    生成一个示例图像, 表示colormap表示的数值范围, 归一化到[0, 1]区间内。
    
    Args:
        colormap: OpenCV colormap to use.
        width: Width of the generated image.
        height: Height of the generated image.
    
    Returns:
        img: Generated image as a numpy array.
    """
    # Create a gradient from 0 to 1
    gradient = np.linspace(0, 1, width).reshape(1, -1)
    gradient = np.vstack([gradient] * height)
    
    # Normalize gradient to range [0, 255]
    norm_gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply colormap
    colored_gradient = cv2.applyColorMap(norm_gradient, colormap)
    
    return colored_gradient


def _vis_1d_seq(seq, lim, path):
    plt.plot(seq, color='r', linewidth=2)
    plt.ylim(lim)
    plt.axhline(0, color='black', linewidth=.5)
    plt.savefig(path)
    canvas = FigureCanvas(plt.gcf())
    canvas.draw()
    w,h = canvas.get_width_height()
    img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(h,w,3)
    plt.close()
    return img

def apply_colormap(weights, colormap=cv2.COLORMAP_JET):
    # Normalize weights to range [0, 255]
    norm_weights = cv2.normalize(weights, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # seqlen,1
    norm_weights = norm_weights[:,0]  # seqlen,
    # Apply colormap
    colored_weights = cv2.applyColorMap(norm_weights, colormap) # seq_len, 1, 3
    colored_weights = colored_weights.transpose(1,0,2) # 1, seqlen, 3
    return colored_weights

def vis_1d_seq(seq, lim, path, weights=None, mode='below'):
    fig, ax = plt.subplots()
    if weights is not None and mode == 'background':
        # Apply colormap to weights
        colored_weights = apply_colormap(weights) # 1, seqlen, 3
        colored_weights = colored_weights[:,:,::-1]  # BGR -> RGB
        # Create a background image with the same size as the plot
        extent = [0, len(seq), lim[0], lim[1]]
        ax.imshow(colored_weights, aspect='auto', extent=extent, alpha=0.5)
    
    ax.plot(seq, color='r', linewidth=2)
    # ax.set_ylim(lim)
    ax.axhline(0, color='black', linewidth=.5)
    
    if weights is not None and mode == 'below':
        # Create a new figure for the weights
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [6, 1]}, figsize=(10, 6))
        ax1.plot(seq, color='r', linewidth=2)
        ax1.set_ylim(lim)
        ax1.axhline(0, color='black', linewidth=.5)
        
        # Apply colormap to weights
        colored_weights = apply_colormap(weights)
        colored_weights = colored_weights[:,:,::-1]  # BGR -> RGB
        # Create an image with the weights
        ax2.imshow(colored_weights, aspect='auto')
        ax2.set_yticks([])
        ax2.set_xticks([])
    
    plt.savefig(path)
    canvas = FigureCanvas(plt.gcf())
    canvas.draw()
    w, h = canvas.get_width_height()
    img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(h, w, 3)
    plt.close()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def vis_2_1d_seqs(seqs, tags, lims, path, weights=None, mode='below'):
    """
    Visualize up to 2 1D sequences in one figure with separate y-axes.
    
    Args:
        seqs: List of sequences to plot (max 2)
        tags: List of labels for each sequence
        lims: List of (min, max) y-limits for each sequence
        path: Path to save the figure
        weights: Optional attention weights to visualize
        mode: How to display weights ('below' or 'background')
    """
    assert len(seqs) <= 2, "Maximum 2 sequences allowed"
    assert len(seqs) == len(tags) == len(lims), \
        f"Input lengths must match: {len(seqs)}, {len(tags)}, {len(lims)}"
    
    colors = ['r', 'b']  # colors for the two possible curves
    
    if weights is not None and mode == 'below':
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [6, 1]}, figsize=(10, 6))
        ax_main = ax1
    else:
        fig, ax_main = plt.subplots()

    if weights is not None and mode == 'background':
        colored_weights = apply_colormap(weights)
        colored_weights = colored_weights[:,:,::-1]  # BGR -> RGB
        extent = [0, len(seqs[0]), lims[0][0], lims[0][1]]  # Use first sequence's limits
        ax_main.imshow(colored_weights, aspect='auto', extent=extent, alpha=0.5)
    
    # Plot first sequence on the left y-axis
    ax_left = ax_main
    ax_left.plot(seqs[0], color=colors[0], linewidth=2, label=tags[0])
    # ax_left.set_ylim(lims[0])
    ax_left.set_ylabel(tags[0], color=colors[0])
    ax_left.tick_params(axis='y', labelcolor=colors[0])
    
    if len(seqs) == 2:
        # Plot second sequence on the right y-axis
        ax_right = ax_main.twinx()
        ax_right.plot(seqs[1], color=colors[1], linewidth=2, label=tags[1])
        # ax_right.set_ylim(lims[1])
        ax_right.set_ylabel(tags[1], color=colors[1])
        ax_right.tick_params(axis='y', labelcolor=colors[1])
    
    ax_main.axhline(0, color='black', linewidth=.5)
    
    # Add legend
    lines_left = ax_left.get_lines()
    if len(seqs) == 2:
        lines_right = ax_right.get_lines()
        lines = lines_left + lines_right
        labels = [l.get_label() for l in lines]
        ax_main.legend(lines, labels, loc='upper right')
    
    if weights is not None and mode == 'below':
        colored_weights = apply_colormap(weights)
        colored_weights = colored_weights[:,:,::-1]  # BGR -> RGB
        ax2.imshow(colored_weights, aspect='auto')
        ax2.set_yticks([])
        ax2.set_xticks([])
    
    plt.savefig(path)
    canvas = FigureCanvas(plt.gcf())
    canvas.draw()
    w, h = canvas.get_width_height()
    img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(h, w, 3)
    plt.close()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img