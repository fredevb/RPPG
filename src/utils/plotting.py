import matplotlib.pyplot as plt

def iter_grid_plt(x, grid_shape, fig_size=(12,12)):
    _, axs = plt.subplots(grid_shape[0], grid_shape[1], figsize=fig_size)
    axs = axs.flatten()
    for xi, ax in zip(x, axs):
        yield xi, ax

def plot_signals(t, signals, ax = None):
    rgb_index_color_map = {0 : 'r', 1 : 'g', 2 : 'b'}
    for idx, x in enumerate(signals):
        if ax:
            ax.plot(t, x, rgb_index_color_map[idx])
        else:
            plt.plot(t, x, rgb_index_color_map[idx])

def plt_relation(y_pred, y_true):
    plt.xlabel('Predition')
    plt.ylabel('Truth')
    plt.scatter(y_pred, y_true)