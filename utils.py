import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_train_loss(model):
    fig = plt.figure(figsize=(8, 6))
    plt.plot(model.train_losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training loss')
    plt.show()

def plot_density(model, data):
    (x_range, y_range), density_eval = model.eval_density_grid(n_points=500)
    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    # Plot of data and contour lines of density
    ax1.plot(data[:,0], data[:,1], 'b.', markersize=1)
    ax1.contour(x_range, y_range, density_eval, colors='red')
    ax1.set_xlabel('x_1')
    ax1.set_ylabel('x_2')
    ax1.set_title('Data and contour lines of density')

    # Color map of density
    cm = ax2.pcolormesh(x_range, y_range, density_eval, cmap=plt.cm.RdBu_r, shading='auto')
    cbar = fig.colorbar(cm)
    ax2.set_xlabel('x_1')
    ax2.set_ylabel('x_2')
    ax2.set_title('Heat map of density')
    plt.show()

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
