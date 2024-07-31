import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib import cm
from tqdm import tqdm
import matplotlib.gridspec as gridspec
from sys import path_importer_cache
import os

def eggholder(x):
    '''Evaluates Eggholder function for d-length vector x.'''
    assert feasible_check(x),"x out of bounds: {}".format(x)

    d = len(x)
    fx = 0
    for i in range(0, d-1):
        fx += -1*(x[i+1] + 47)*np.sin(np.sqrt(np.abs(x[i+1] + 0.5*x[i] + 47)))\
                -x[i]*np.sin(np.sqrt(np.abs(x[i] - x[i+1] - 47)))
    return fx

def eggholder_unconstrained(x, w_val=1e6, T=np.inf):
    '''Evaluates Eggholder function for d-length vector x.'''

    d = len(x)
    w = np.ones(d) * 1e6
    c = violations(x, d)
    fx = 0
    for i in range(0, d-1):
        fx += -1*(x[i+1] + 47)*np.sin(np.sqrt(np.abs(x[i+1] + 0.5*x[i] + 47)))\
                -x[i]*np.sin(np.sqrt(np.abs(x[i] - x[i+1] - 47)))
    fx += (1/T) * w@c
    return fx

def violations(x, d):
    c = np.zeros(d)
    for i in range(d):
        c[i] = np.max([0, (x[i] - 512), (-x[i] - 512)])
    return c


def plot_eggholder2D(archive=[], path=[], run_name = 'initial'):
    '''Plot the Eggholder Function for 2D vector x.
    inputs:
      - archive   {f(x): [x]}
      - path      {f(x): [x]}
    '''
    N = 1000 # for plotting
    x_range = np.linspace(-512,512,N)
    y_range = np.linspace(-512,512,N)
    xgrid, ygrid = np.meshgrid(x_range, y_range)
    zvals = np.zeros((N,N))

    # fill zvals according to whether file exists yet or not.
    if os.path.exists('zvals{}.npy'.format(N)):
        zvals = np.load('zvals{}.npy'.format(N))
    else:
        # create zvals array to represent the surface.
        for i in tqdm(range(N)):
            for j in range(N):
                zvals[j,i] = eggholder([x_range[i],y_range[j]])
        np.save('zvals{}.npy'.format(N), zvals)

    # form arrays of points to plot on the surface
    A = len(archive)
    P = len(path)

    x_archive = np.zeros(A)
    y_archive = np.zeros(A)
    z_archive = np.zeros(A)
    x_path = np.zeros(P)
    y_path = np.zeros(P)
    z_path = np.zeros(P)

    for index, key in enumerate(archive):
        z_archive[index] = key
        x_archive[index] = archive[key][0]
        y_archive[index] = archive[key][1]
    
    for index, key in enumerate(path):
        z_path[index] = key
        x_path[index] = path[key][0]
        y_path[index] = path[key][1]

    if path == []:
        # plot the surface of the 2D eggholder function

        fig = plt.figure(figsize=(12,9))
        ax = fig.add_subplot(1,1,1,projection='3d')
        surf = ax.plot_surface(xgrid, ygrid, zvals, rstride=5, cstride=5, linewidth=0, cmap=cm.plasma)

        ax.set_title('Surface Plot of the 2D Eggholder Function', fontsize=16)
        ax.set_xlabel('$x_1$', fontsize=14)
        ax.set_ylabel('$x_2$', fontsize=14)
        ax.set_zlabel('$f(\mathbf{x})$', fontsize=14)
        ax.scatter(x_archive, y_archive, z_archive, color = 'r')
        plt.tight_layout()
        # fig.savefig('eggholder_surface.eps', format = 'eps', transparent=True, bbox_inches='tight')

    # plot the archived solutions on the contour plot

    gs = gridspec.GridSpec(1,2)
    fig2 = plt.figure(figsize=(16, 6), dpi=80)
    ax1 = fig2.add_subplot(gs[0,0])
    ax2 = fig2.add_subplot(gs[0,1])

    # fig2 = plt.figure(figsize=(12,9))
    # ax = fig2.add_subplot(1,1,1)
    ax1.contour(xgrid, ygrid, zvals, 30, cmap=cm.plasma, zorder = -1)    
    if archive != []:
        ax1.set_title('(a) Archive', fontsize=18)
        ax1.scatter(x_archive, y_archive, zorder = 1, s=100, marker = 'x', color = 'lime')
    else:
        ax1.set_title('Contour Plot of the 2D Eggholder Function', fontsize=16)
    ax1.set_xlabel('$x_1$', fontsize=14)
    ax1.set_ylabel('$x_2$', fontsize=14)
    # cbar = fig2.colorbar(surf, aspect=15)
    # cbar.set_label('$f(\mathbf{x})$', rotation=0, fontsize=14)
    
    # z = [min(i) for i in zvals]

    # plot the path solutions on the contour plot

    if path != []:
        # fig3 = plt.figure(figsize=(12,9))
        # ax = fig2.add_subplot(1,1,1)
        ax2.contour(xgrid, ygrid, zvals, 30, cmap=cm.plasma, zorder = -1)    
        ax2.scatter(x_path, y_path, zorder = 1, s=0.8, marker = 'o', color = 'lime')

        ax2.set_title('(b) Path', fontsize=18)
        ax2.set_xlabel('$x_1$', fontsize=14)
        ax2.set_ylabel('$x_2$', fontsize=14)
        # cbar = fig2.colorbar(CS, aspect=15)
        # cbar.set_label('$f(\mathbf{x})$', rotation=0, fontsize=14)
        
        # fig4 = plt.figure(figsize=(12,9))
        # ax = fig4.add_subplot(1, 1, 1)
        # plt.plot(z_path)
        # ax.set_xlabel('Iteration', fontsize=14)
        # ax.set_ylabel('$f(\mathbf{x})$', fontsize=14)
        # ax.set_title('Objective Function value over Iterations for Minimization of 2D Eggholder', fontsize=16)

    fig2.savefig('Figures/contours_{}.pdf'.format(run_name), format = 'pdf', transparent=True, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
    return(z_path)

def plot_progress(fx_progress, z_path, d, run_name, f_evals_count_list = []):
    '''Plot the progress of best solution found across number of iterations'''

    gs = gridspec.GridSpec(1,2)
    fig2 = plt.figure(figsize=(16, 6), dpi=80)
    ax1 = fig2.add_subplot(gs[0,0])
    ax2 = fig2.add_subplot(gs[0,1])

    if f_evals_count_list == []:
        ax1.plot(fx_progress, label='Simulated Annealing')
    else:
        ax1.plot(f_evals_count_list,fx_progress, label='Simulated Annealing')
    ax1.set_title('(a) Best Solution Across Iterations', fontsize=18)
    ax1.set_xlabel('Function Evaluation Count', fontsize=14)
    ax1.set_ylabel('$f(\mathbf{x^*})$', fontsize=14)

    if d == 2:
        if f_evals_count_list == []:
            ax1.plot(np.ones(len(fx_progress))*-959.639453, '--', label='Global Minimum')
        else:
            ax1.plot(f_evals_count_list, np.ones(len(fx_progress))*-959.639453, '--', label='Global Minimum')
        ax1.legend()

    ax2.plot(z_path)
    ax2.set_title('(b) Objective Function Across Iterations', fontsize=18)
    ax2.set_xlabel('Iteration', fontsize=14)
    ax2.set_ylabel('$f(\mathbf{x^*})$', fontsize=14)

    fig2.savefig('Figures/progress_{}.pdf'.format(run_name), format = 'pdf', transparent=True, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

    # plt.figure()
    # plt.plot(fx_progress)
    # plt.xlabel('Number of Objective Evaluations', fontsize=14)
    # plt.ylabel('$f(\mathbf{x^*})$', fontsize=14)
    # plt.title('Best Solution Found Across Iterations', fontsize=14)
  
def plot_progress_multseed(fx_progress_list, d, title=None):
    '''Plot the progress of the best solution found across the number of iterations'''
    plt.figure()
    max_length = max(len(fx_progress_list[i]) for i in range(len(fx_progress_list)))
    for seed in range(len(fx_progress_list)):
        plt.plot(fx_progress_list[seed], label=seed)
    plt.xlabel('Function Evaluation Count', fontsize=14)
    plt.ylabel('$f(\mathbf{x^*})$', fontsize=14)

    if d == 2:
        plt.plot(np.ones(max_length)*-959.639453, '--', label='Global Minimum')

    if title == None:
        plt.title('Best Solution Found Across Iterations for Various Seeds', fontsize=14)
        title = 'General'
    else:
        plt.title(title, fontsize=14)
    plt.savefig('Figures/{}_multseed.pdf'.format(title), format = 'pdf', transparent=True, bbox_inches='tight')
    plt.show()
    
def plot_progress_multvar(fx_progress_dict, d, filename = 'General', count=False, f_evals_count_dict = None):
    '''Plot the progress of the best solution found across number of iterations for multiple runs.'''

    alphabet = ['a','b','c','d','e']
    num_graphs = len(fx_progress_dict)
    
    gs = gridspec.GridSpec(1,num_graphs)
    fig = plt.figure(figsize=(7*num_graphs, 5), dpi=80)

    for index, value in enumerate([*fx_progress_dict]):
        progress_list = fx_progress_dict[value]
        if count:
            f_evals_count_list = f_evals_count_dict[value]
        # max_length = max(len(progress_list[i]) for i in range(len(progress_list)))
        ax = fig.add_subplot(gs[0,index])
        final_vals = []
        for seed in range(len(progress_list)):
            if count:
                ax.plot(f_evals_count_list[seed], progress_list[seed], label = seed)
            else:
                ax.plot(progress_list[seed], label = seed)
            final_vals.append(progress_list[seed][-1])
            if d == 2:
                ax.plot(np.ones(15500)*-959.639453, '--', label='Global Minimum')
        ax.set_xlabel('Function Evaluation Count', fontsize=14)
        ax.set_ylabel('$f(\mathbf{x^*})$', fontsize=14)
        f_av = np.mean(final_vals)
        ax.set_title('({}) {}, f_av = {:.4f}'.format(alphabet[index], value, f_av), fontsize=18)

    plt.savefig('Figures/{}.pdf'.format(filename), format = 'pdf', transparent=True, bbox_inches='tight')
    plt.show()

def feasible_check(x):
    '''Checks if the vector x is in the feasible region.
    Returns True if feasible and False if not.
    '''
    if np.any(np.abs(x) > 512):
        return False
    else:
        return True