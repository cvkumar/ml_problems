from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt


#  TODO: Make this graph points
def graph_with_num_clusters(
    title,
    y_label,
    y_data,
    x_data,
    seed=0,
    x_label="Number of Clusters",
    color="r",
    path="",
):
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x_data, y_data, color=color)
    file_name = f"{title}-seed_{seed}.png"
    plt.savefig(f"{path}{file_name}")
    plt.close()
