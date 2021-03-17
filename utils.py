import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_train_loss(model, ax=None, figsize=(8,6)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.plot(model.train_losses)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training loss')


def plot_density(model, data, density_grid=[-5, 5, -5, 5], axes=None,
    figsize=(16, 6)):

    if axes is None:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    
    # Evaluate density grid
    (x_range, y_range), density_eval = model.eval_density_grid(n_points=500,
        grid=density_grid)

    # Plot of data and contour lines of density
    axes[0].plot(data[:,0], data[:,1], 'b.', markersize=1)
    axes[0].contour(x_range, y_range, density_eval, colors='red')
    axes[0].set_xlabel('x_1')
    axes[0].set_ylabel('x_2')
    axes[0].set_title('Data and contour lines of density')

    # Color map of density
    cm = axes[1].pcolormesh(x_range, y_range, density_eval, cmap=plt.cm.RdBu_r,
        shading='auto')
    cbar = plt.colorbar(cm, ax=axes[1])
    axes[1].set_xlabel('x_1')
    axes[1].set_ylabel('x_2')
    axes[1].set_title('Heat map of density')


def plot_density_alt(model, data):
    (x_range, y_range), density_eval = model.eval_density_grid(n_points=500)
    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2,
                                    figsize=(20, 10))
    # Plot of data and contour lines of density
    ax1.plot(data[:,0], data[:,1], 'b.', markersize=1)
    ax1.contour(x_range, y_range, density_eval, colors='red')
    ax1.set_xlabel('x_1')
    ax1.set_ylabel('x_2')
    ax1.set_title('Data and contour lines of density')

    # Color map of density
    im = ax2.imshow(density_eval, interpolation='bicubic', cmap='winter', extent=[x_range[0], x_range[-1],
           y_range[0], y_range[-1]], origin='lower')

    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.1)
    #divider = make_axes_locatable(ax2)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar_ax = fig.add_axes([0.81, 0.175, 0.025, 0.65])
    cbar = fig.colorbar(im, cax=cbar_ax )

    ax2.set_xlabel('x_1')
    ax2.set_ylabel('x_2')
    ax2.set_title('Heat map of density')

    ax1.set_aspect(1)
    ax2.set_aspect(1)

    plt.show()
