
import os
import numpy as np
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
                  show = False,
                  save = True):
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
                           normalize1d = True)
    corner.corner(samp_2,
                  color = 'blue',
                  bins = n_bins,
                  fig = figure, 
                  normalize1d = True)
    plt.legend(handles = [red_line, blue_line], 
               bbox_to_anchor = (-ndims_eff+1.8, ndims_eff+.3, 1., 0.) ,
               fontsize='xx-large')
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

    
