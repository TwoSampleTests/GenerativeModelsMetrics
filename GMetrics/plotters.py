import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions
tfb = tfp.bijectors
import matplotlib.pyplot as plt
from fpdf import FPDF
from PIL import Image
#import Trainer_2 as Trainer
#import Metrics_old as Metrics
from statistics import mean,median
import matplotlib.lines as mlines
from GMetrics import corner

def make_pdf_from_img(img):
    """Make pdf from image
    Used to circumvent bud in plot_model which does not allow to export pdf"""
    img_pdf = os.path.splitext(img)[0]+".pdf"
    cover = Image.open(img)
    width, height = cover.size
    pdf = FPDF(unit = "pt", format = [width, height])
    pdf.add_page()
    pdf.image(img, 0, 0)
    pdf.output(img_pdf, "F")

def sample_plotter(target_test_data,nf_dist,path_to_plots):

        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        f.suptitle('target vs nf')
        x=target_test_data
        ax1.plot(x[:,0],targ_dist.prob(x),'.')
        ax1.set_yscale('log')
        ax1.set_title('target')
        y=nf_dist.sample(target_test_data.shape[1])
        ax2.plot(y[:,0],nf_dist.prob(y),'.')
        ax2.set_yscale('log')
        ax2.set_title('nf')
        f.savefig(path_to_plots+'/sample_plot.pdf')
        ax1.cla()
        ax2.cla()
        f.clf()
        
        return
        
        
def train_plotter(t_losses,v_losses,path_to_plots,yscale='log'):
    plt.plot(t_losses,label='train')
    plt.plot(v_losses,label='validation')
    plt.legend()
    plt.title('history')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.yscale(yscale)
    plt.savefig(path_to_plots+'/loss_plot.pdf')
    plt.close()
    return
 
def cornerplotter(dist_1,
                  dist_2,
                  path_to_plots,
                  figure_name = "corner_plot.png",
                  max_points = 50_000,
                  max_dim = 32, 
                  n_bins = 50,
                  title = None,
                  corner_kwargs = {},
                  show = False,
                  save = True
                 ) -> None:
    try:
        tf.random.set_seed(0)
        np.random.seed(0)
        samp_1 = dist_1.sample(max_points).numpy()
    except:
        samp_1 = dist_1
    try:
        samp_2 = dist_2.sample(max_points).numpy()
    except:
        samp_2 = dist_2
    shape_1 = samp_1.shape
    shape_2 = samp_2.shape
    if shape_1 != shape_2:
        raise ValueError("The two samples have different shapes.")
    else:
        shape = shape_1
    samp_1_no_nans = samp_1[~np.isnan(samp_1).any(axis=1), :]
    samp_2_no_nans = samp_2[~np.isnan(samp_2).any(axis=1), :]
    if len(samp_1) != len(samp_1_no_nans):
        print("Points in sample_1 containing nan have been removed. The fraction of nans over the total samples was:", str((len(samp_1)-len(samp_1_no_nans))/len(samp_1)),".")
    if len(samp_2) != len(samp_2_no_nans):
        print("Points in sample_2 containing nan have been removed. The fraction of nans over the total samples was:", str((len(samp_2)-len(samp_2_no_nans))/len(samp_2)),".")
    samp_1 = samp_1_no_nans[:shape[0]]
    samp_2 = samp_2_no_nans[:shape[0]]
    labels = []
    for i in range(1,shape[1]+1):
        labels.append(r"$\mathbf{x}_{%d}$" % i)
        i = i+1
    thin = int(shape[1]/max_dim)+1
    if thin<=2:
        thin = 1
    samp_1 = samp_1[:, ::thin]
    samp_2 = samp_2[:, ::thin]
    ndims_eff = samp_1.shape[1]
    labels = list(np.array(labels)[::thin])
    red_line = mlines.Line2D([], [], color = 'red', label = '$\mathbf{X}_{1}$')
    blue_line = mlines.Line2D([], [], color = 'blue', label = '$\mathbf{X}_{2}$')
    figure = corner.corner(samp_1, 
                           color = 'red',
                           bins = n_bins,
                           labels = [r"%s" % s for s in labels],
                           normalize1d = True,
                           **corner_kwargs)
    corner.corner(samp_2,
                  color = 'blue',
                  bins = n_bins,
                  fig = figure, 
                  normalize1d = True)
    plt.legend(handles = [red_line, blue_line], 
               bbox_to_anchor = (-ndims_eff+1.8, ndims_eff+.3, ndims_eff-1, 0.) ,
               fontsize='xx-large')
    if title:
        figure.suptitle(title, fontsize=20)
    if save:
        figure_path = os.path.join(path_to_plots,figure_name)
        _, file_extension = os.path.splitext(figure_name)
        save_kwargs = {}
        if file_extension.lower() in ['.jpg', '.jpeg', '.png']:
            save_kwargs['pil_kwargs'] = {'quality': 50}
        plt.savefig(figure_path, **save_kwargs)
    if show:
        plt.show()
    plt.close()
    return

def marginal_plot(target_test_data,sample_nf,path_to_plots,ndims):
    n_bins=50
    if ndims<=4:
        fig, axs = plt.subplots(int(ndims/4), 4, tight_layout=True)
        for dim in range(ndims):
            column=int(dim%4)
            axs[column].hist(target_test_data[:,dim], bins=n_bins,density=True,histtype='step',color='red')
            axs[column].hist(sample_nf[:,dim], bins=n_bins,density=True,histtype='step',color='blue')
            x_axis = axs[column].axes.get_xaxis()
            x_axis.set_visible(False)
            y_axis = axs[column].axes.get_yaxis()
            y_axis.set_visible(False)
    elif ndims>=100:
        fig, axs = plt.subplots(int(ndims/10), 10, tight_layout=True)
        for dim in range(ndims):
            row=int(dim/10)
            column=int(dim%10)
            axs[row,column].hist(target_test_data[:,dim], bins=n_bins,density=True,histtype='step',color='red')
            axs[row,column].hist(sample_nf[:,dim], bins=n_bins,density=True,histtype='step',color='blue')
            x_axis = axs[row,column].axes.get_xaxis()
            x_axis.set_visible(False)
            y_axis = axs[row,column].axes.get_yaxis()
            y_axis.set_visible(False)
    else:
        fig, axs = plt.subplots(int(ndims/4), 4, tight_layout=True)
        for dim in range(ndims):
            row=int(dim/4)
            column=int(dim%4)
            axs[row,column].hist(target_test_data[:,dim], bins=n_bins,density=True,histtype='step',color='red')
            axs[row,column].hist(sample_nf[:,dim], bins=n_bins,density=True,histtype='step',color='blue')
            x_axis = axs[row,column].axes.get_xaxis()
            x_axis.set_visible(False)
            y_axis = axs[row,column].axes.get_yaxis()
            y_axis.set_visible(False)
    fig.savefig(path_to_plots+'/marginal_plot.pdf',dpi=300)
    fig.clf()
    return
   
def plot_corr_matrix(dist: np.ndarray,
                     path_to_plots,
                     figure_name = "corre_matrix_plot.pdf",
                     max_points = 10_000,
                     title = None,
                     show_labels = True,
                     show = False,
                     save = True
                    ) -> None:
    """
    """
    try:
        tf.random.set_seed(0)   
        np.random.seed(0)
        samp = dist.sample(max_points).numpy()
    except:
        samp = dist
    shape = samp.shape
    labels = []
    for i in range(1,shape[1]+1):
        labels.append(r"$\mathbf{x}_{%d}$" % i)
        i = i+1
    #df: pd.DataFrame = pd.DataFrame(samp, columns=labels)
    #f: plt.Figure = plt.figure(figsize=(18, 18))
    #plt.matshow(df.corr(), fignum=f.number)
    #cb = plt.colorbar()
    #plt.grid(False)

    df = pd.DataFrame(samp, columns=labels)

    # Create figure and plot correlation matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(df.corr(), interpolation='nearest')
    fig.colorbar(cax)

    # Set axis labels
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    if show_labels:
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels, rotation=90)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Turn off the grid
    ax.grid(False)
    
    if title:
        fig.suptitle(title, fontsize=20)
    
    if save:
        figure_path = os.path.join(path_to_plots,figure_name)
        _, file_extension = os.path.splitext(figure_name)
        save_kwargs = {}
        if file_extension.lower() in ['.jpg', '.jpeg', '.png']:
            save_kwargs['pil_kwargs'] = {'quality': 50}
        plt.savefig(figure_path, **save_kwargs)
    if show:
        plt.show()
    return

def plot_corr_matrix_side_by_side(dist_1: tfp.distributions.Distribution,
                                  dist_2: tfp.distributions.Distribution,
                                  path_to_plots,
                                  figure_name="corre_matrix_plot_side_by_side.pdf",
                                  max_points=10_000,
                                  title = None,
                                  show_labels=True,
                                  show=False,
                                  save=False) -> None:
    """
    Plots two correlation matrices side by side with a single colorbar in the middle.

    Parameters:
    - dist_1, dist_2: Distributions from which samples are drawn to compute the correlation matrices.
    - path_to_plots: Directory path to save the figure.
    - figure_name: Name of the file to save the figure.
    - max_points: Maximum number of points to sample from each distribution.
    - show_labels: Whether to show labels on the axes.
    - show: Whether to display the plot.
    - save: Whether to save the plot to a file.
    """
    try:
        tf.random.set_seed(0)
        np.random.seed(0)
        samp_1 = dist_1.sample(max_points).numpy()
    except:
        samp_1 = dist_1
    try:
        samp_2 = dist_2.sample(max_points).numpy()
    except:
        samp_2 = dist_2
    labels = [r"$\mathbf{x}_{%d}$" % i for i in range(1, samp_1.shape[1] + 1)]

    # Create DataFrames
    df1 = pd.DataFrame(samp_1, columns=labels)
    df2 = pd.DataFrame(samp_2, columns=labels)

    # Create figure and axes for subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.2})

    # Plot correlation matrices
    cax1 = ax1.matshow(df1.corr(), interpolation='nearest')
    cax2 = ax2.matshow(df2.corr(), interpolation='nearest')

    # Set axis labels
    for ax in (ax1, ax2):
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        if show_labels:
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels, rotation=90)
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    # Disable grid
    ax1.grid(False)
    ax2.grid(False)

    # Add one colorbar in the middle
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])  # Adjust these values to move and resize the colorbar
    fig.colorbar(cax1, cax=cbar_ax)
    
    if title:
        fig.suptitle(title, fontsize=20)

    # Saving the figure
    if save:
        figure_path = os.path.join(path_to_plots, figure_name)
        _, file_extension = os.path.splitext(figure_name)
        save_kwargs = {}
        if file_extension.lower() in ['.jpg', '.jpeg', '.png']:
            save_kwargs['pil_kwargs'] = {'quality': 50}
        plt.savefig(figure_path, **save_kwargs)

    # Show plot if requested
    if show:
        plt.show()

    plt.close()
    return
