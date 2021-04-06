import os
import pyro
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from models import GaussianMixtureModel, CPModel, TensorTrain

import numpy as np
from scipy import stats, linalg

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


def save_model(model, model_name):
    model_path = 'models/' + model_name

    # Make sure not to overwrite existing saved models
    if os.path.exists(model_path + '.pt'):
        model_count = 2
        model_path += '_' + str(model_count)

        while os.path.exists(model_path + '.pt'):
            model_count += 1
            model_path = model_path.rsplit('_', maxsplit=1)[0] + '_' + str(model_count)
    
    model_type = type(model).__name__
    model_kwargs = model.kwargs

    torch.save({
        'model_type': model_type,
        'model_kwargs': model_kwargs,
        'state_dict': model.state_dict(),
        'pyro_params': pyro.get_param_store().get_state(),
        'train_losses': model.train_losses
    }, model_path + '.pt')


def load_model(model_name, device='cpu'):
    pyro.clear_param_store()

    # Load model dict
    model_path = 'models/' + model_name
    model_dict = torch.load(model_path, map_location=device)

    # Set model
    model_dict['model_kwargs']['device'] = device
    model = eval(model_dict['model_type'])(**model_dict['model_kwargs'])
    model.load_state_dict(model_dict['state_dict'])
    model.train_losses = model_dict['train_losses']
    model.to(device)

    # Set pyro param store
    pyro.get_param_store().set_state(model_dict['pyro_params'])

    return model

def partial_corr(C):
    """
    FROM: https://gist.github.com/fabianp/9396204419c7b638d38f

    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling
    for the remaining variables in C.
    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable
    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """

    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot( beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)

            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr

    return P_corr

def order_variables_partial_correlation(data):
    P_og = np.abs(partial_corr(data))
    n = len(P_og[:, 0])
    n_sq = int(np.sqrt(n))
    P_og -= np.identity(n)

    # We start at first variable regardless
    import random
    import copy
    best_chain = []
    best_score = 0
    for i in range(100):
        P = copy.deepcopy(P_og)
        score = 0
        start_index = random.randint(0, n-1)
        start_index = int(n/2)
        chain = [start_index]
        P[start_index, :] = 0
        for i in range(n-1):
            best_var = np.argmax(P[:, chain[-1]])
            score += P[best_var, chain[-1]]
            chain.append(best_var)
            P[best_var, :] = 0
        chain = np.array(chain)
        #print(f'This ordering gave a score of {round(score, 2)}')
        if score > best_score:
            best_score = copy.deepcopy(score)
            best_chain = copy.deepcopy(chain)

    print(f'Best ordering gave {round(best_score, 2)} from the order {best_chain}')
    #img = np.zeros((n_sq, n_sq), dtype=np.uint8)
    #for i, pt in enumerate(best_chain):
    #    img[int(pt//n_sq), int(pt%n_sq)] = int(((i)/n)*255)
    #    plt.text(int(pt%n_sq), int(pt//n_sq), f"{i}")
    #plt.imshow(img)
    #plt.show()
    return best_chain