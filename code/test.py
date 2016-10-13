"""
test.py

Author:		Barbara Rakitsch
Year:		2012
Group:		Machine Learning and Computational Biology Group (http://webdav.tuebingen.mpg.de/u/karsten/group/)
Institutes:	Max Planck Institute for Developmental Biology and Max Planck Institute for Intelligent Systems (72076 Tuebingen, Germany)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import csv
import scipy as SP
import pdb
import lmm_lasso
import matplotlib.pylab as PLT
import os

if __name__ == "__main__":
    # plot directory
    plots_dir = 'plots'

    # data directory
    data_dir = 'data'
    
    # load genotypes
    geno_filename = os.path.join(data_dir,'genotypes.csv')
    X = SP.genfromtxt(geno_filename)
    [n_s,n_f] = X.shape

    # simulate phenotype
    SP.random.seed(1)
    n_c = 5
    idx = SP.random.randint(0,n_f,n_c)
    w = 1./n_c * SP.ones((n_c,1))
    ypheno = SP.dot(X[:,idx],w)
    ypheno = (ypheno-ypheno.mean())/ypheno.std()
    pheno_filename = os.path.join(data_dir,'poppheno.csv')
    ypop = SP.genfromtxt(pheno_filename)
    ypop = SP.reshape(ypop,(n_s,1))
    y = 0.3*ypop + 0.5*ypheno + 0.2*SP.random.randn(n_s,1)
    y = (y-y.mean())/y.std()
    
    # init
    debug = False
    n_train = 150
    n_test = n_s - n_train
    n_reps = 100
    f_subset = 0.5
    mu = 10

    # split into training and testing
    train_idx = SP.random.permutation(SP.arange(n_s))
    test_idx = train_idx[n_train:]
    train_idx = train_idx[:n_train]

    # calculate kernel
    K = 1./n_f*SP.dot(X,X.T)
    
    # train
    res = lmm_lasso.train(X[train_idx],K[train_idx][:,train_idx],y[train_idx],mu,debug=debug)
    w = res['weights']
    print '... number of Nonzero Weights: %d'%(w!=0).sum()

    # predict
    ldelta0 = res['ldelta0']
    yhat = lmm_lasso.predict(y[train_idx],X[train_idx,:],X[test_idx,:],K[train_idx][:,train_idx],K[test_idx][:,train_idx],ldelta0,w)
    corr = 1./n_test * ((yhat-yhat.mean())*(y[test_idx]-y[test_idx].mean())).sum()/(yhat.std()*y[test_idx].std())
    print '... corr(Yhat,Ytrue): %.2f (in percent)'%(corr)


    # stability selection
    ss = lmm_lasso.stability_selection(X,K,y,mu,n_reps,f_subset)

    # create plot folder
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        
    # plot kernel
    fig = PLT.figure()
    fig.add_subplot(111)
    PLT.imshow(K,interpolation='nearest')
    PLT.xlabel('samples')
    PLT.ylabel('samples')
    PLT.title('Population Kernel')
    fn_out = os.path.join(plots_dir,'kernel.pdf')
    PLT.savefig(fn_out)
    PLT.close()

    # plot negative log likelihood of the null model
    monitor = res['monitor_nm']
    fig = PLT.figure()
    fig.add_subplot(111)
    PLT.plot(monitor['ldeltagrid'],monitor['nllgrid'],'b-')
    PLT.plot(monitor['ldeltaopt'],monitor['nllopt'],'r*')
    PLT.xlabel('ldelta')
    PLT.ylabel('negative log likelihood')
    PLT.title('nLL on the null model')
    fn_out = os.path.join(plots_dir, 'nLL.pdf')
    PLT.savefig(fn_out)
    PLT.close()
        
    # plot Lasso convergence
    monitor = res['monitor_lasso']
    fig = PLT.figure()
    fig.add_subplot(311)
    PLT.plot(monitor['objval'])
    PLT.title('Lasso convergence')
    PLT.ylabel('objective')
    fig.add_subplot(312)
    PLT.plot(monitor['r_norm'],'b-',label='r norm')
    PLT.plot(monitor['eps_pri'],'k--',label='eps pri')
    PLT.ylabel('r norm')
    fig.add_subplot(313)
    PLT.plot(monitor['s_norm'],'b-',label='s norm')
    PLT.plot(monitor['eps_dual'],'k--',label='eps dual')
    PLT.ylabel('s norm')
    PLT.xlabel('iteration')
    fn_out = os.path.join(plots_dir,'lasso_convergence.pdf')
    PLT.savefig(fn_out)
    PLT.close()

    # plot weights
    fig = PLT.figure()
    fig.add_subplot(111)
    PLT.title('Weight vector')
    PLT.plot(w,'b',alpha=0.7)
    for i in range(idx.shape[0]):
        PLT.axvline(idx[i],linestyle='--',color='k')
    fn_out = os.path.join(plots_dir,'weights.pdf')
    PLT.savefig(fn_out)
    PLT.close()

    # plot stability selection
    fig = PLT.figure()
    fig.add_subplot(111)
    PLT.title('Stability Selection')
    PLT.plot(ss,'b',alpha=0.7)
    for i in range(idx.shape[0]):
        PLT.axvline(idx[i],linestyle='--',color='k')
    PLT.axhline(0.5,color='r')
    fn_out = os.path.join(plots_dir,'ss_frequency.pdf')
    PLT.savefig(fn_out)
    PLT.close()

    # plot predictions
    fig = PLT.figure()
    fig.add_subplot(111)
    PLT.title('prediction')
    PLT.plot(y[test_idx],yhat, 'bx')
    PLT.plot(y[test_idx],y[test_idx],'k')
    PLT.xlabel('y(true)')
    PLT.ylabel('y(predicted)')
    PLT.xlabel('SNPs')
    PLT.ylabel('weights')
    fn_out = os.path.join(plots_dir,'predictions.pdf')
    PLT.savefig(fn_out)
    PLT.close()
        
        

