import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_model_diagram():
    fig, ax = plt.subplots(figsize=(16, 8))

    # Define the layers
    layers = [
        {"units": "Input", "position": (1, 7), "type": "Input Layer", "label": "Input"},
        {"units": "64", "position": (3, 7), "type": "Hidden Layer", "label": "Conv Layer"},
        {"units": "128", "position": (5, 7), "type": "Hidden Layer", "label": "Conv Layer"},
        {"units": "Pool", "position": (7, 7), "type": "Hidden Layer", "label": "Max Pooling"},
        {"units": "128", "position": (9, 7), "type": "Hidden Layer", "label": "LSTM"},
        {"units": "256", "position": (11, 7), "type": "Hidden Layer", "label": "Dense"},
        {"units": "128", "position": (13, 7), "type": "Hidden Layer", "label": "Dense"},
        {"units": "Softmax (4 outputs)", "position": (15, 7), "type": "Output Layer", "label": ""},
        {"units": "Sigmoid 1 (Auxiliary)", "position": (15, 8), "type": "Output Layer", "label": ""},
        {"units": "Sigmoid 4 (Auxiliary)", "position": (15, 6), "type": "Output Layer", "label": ""},
    ]

    # Draw circles for different layers
    for layer in layers:
        x, y = layer["position"]
        layer_type = layer["type"]
        if layer_type == "Input Layer":
            color = "lightgreen"
        elif layer_type == "Hidden Layer":
            color = "lightblue"
        elif layer_type == "Output Layer":
            color = "lightcoral"
        circle = plt.Circle((x, y), 0.5, edgecolor="black", facecolor=color, lw=2)
        ax.add_artist(circle)

    # Add connections between layers
    for i in range(len(layers) - 3):
        start_x, start_y = layers[i]["position"]
        end_x, end_y = layers[i + 1]["position"]

        # Ensure arrows do not enter the circles
        ax.annotate("", xy=(end_x - 0.5, end_y), xytext=(start_x + 0.5, start_y),
                    arrowprops=dict(arrowstyle="->", lw=1.5))

    # Connect last dense layer to all outputs
    dense_x, dense_y = layers[-4]["position"]
    for output_layer in layers[-3:]:
        output_x, output_y = output_layer["position"]
        # Ensure the arrow connects to the front of the oval
        ax.annotate("", xy=(output_x - 0.5, output_y), xytext=(dense_x + 0.5, dense_y),
                    arrowprops=dict(arrowstyle="->", lw=1.5))

    # Add unit text to circles and labels above
    for layer in layers:
        x, y = layer["position"]
        if layer["type"] == "Output Layer":
            # Place the label just to the right of the circle
            ax.text(x + 0.7, y, layer["units"], ha="left", va="center", fontsize=14, fontweight="bold")
        else:
            # Larger, bold text for units inside the ovals
            ax.text(x, y, layer["units"], ha="center", va="center", fontsize=14, fontweight="bold")

        # Larger, bold text for labels above the ovals
        if layer["label"]:
            ax.text(x, y + 0.8, layer["label"], ha="center", va="center", fontsize=12, fontweight="bold", color="black")

    # Set plot limits and labels
    ax.set_xlim(0, 17)
    ax.set_ylim(5, 9)
    ax.axis("off")

    # Add a legend for layer types
    legend_elements = [
        mpatches.Patch(color="lightgreen", label="Input Layer"),
        mpatches.Patch(color="lightblue", label="Hidden Layer"),
        mpatches.Patch(color="lightcoral", label="Output Layer"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=12)

    # Add a bold title
    plt.title("Model Architecture Diagram", fontsize=16, fontweight="bold")
    plt.show()

# Run the function
plot_model_diagram()
