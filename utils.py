import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


#shape (10, 11)

def plot_output(matrix):

    # Define color map
    cmap = plt.cm.colors.ListedColormap(['tab:blue', 'tab:orange'])

    # Plot the matrix
    plt.imshow(matrix, cmap=cmap, interpolation='nearest')

    # Add colorbar legend
    
    plt.legend()

    # Set x-axis labels and ticks
    plt.xticks(np.arange(0, 11), np.arange(0, 110, 10))
    plt.xlabel('Credence in deontology (%)')

    # Set y-axis labels and ticks
    plt.yticks(np.arange(0, 10), np.arange(10, 0, -1))
    plt.ylabel('Number on people on track (X)')

    # Set the title
    plt.title('MEC Utilitarianism vs Deontology')

    # Create custom proxy artists for the legend
    do_nothing_patch = mpatches.Patch(color='tab:blue', label='Do Nothing')
    switch_patch = mpatches.Patch(color='tab:orange', label='Switch')

    # Add the legend
    plt.legend(handles=[do_nothing_patch, switch_patch], loc='upper right')

    # Show the plot
    plt.show()
