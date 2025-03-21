import os
import matplotlib.pyplot as plt

# Define file-saving function
def save_plot(fig, filename):
    """
    Saves a matplotlib figure to a specified directory, overwriting the file if it already exists.

    This function saves the provided figure in a 'figures' directory (created if it doesn't exist),
    and ensures that each figure is saved with tight bounding boxes. If a file with the same name exists,
    it will be overwritten to keep only the latest version.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The matplotlib figure object to be saved.
    filename : str
        The name of the file (including extension, e.g., 'figure1.png') to save the figure as.

    Returns:
    --------
    None
        The function saves the figure and prints the path where it is stored.
    """
    
    filepath = os.path.join("figures", filename)  # Save in a 'figures' folder
    os.makedirs("figures", exist_ok=True)  # Create directory if it doesn't exist
    fig.savefig(filepath, bbox_inches="tight")
    print(f"Figure saved: {filepath}")