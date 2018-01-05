import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import linalg
from numpy import linalg
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def load_data(dataset_str):
    """Load data."""
    # if dataset_str in ['USPS-Fea', 'CIFAR-Fea', 'Cifar_10000_fea', 'Cifar_R10000_fea']:
    #     data = sio.loadmat('data/{}.mat'.format(dataset_str))
    #     labels = data['labels']
    #     labels = np.zeros([data['labels'].shape[0],np.max(data['labels'])+1])
    #     labels[np.arange(data['labels'].shape[0]),data['labels'].astype(np.int16).flatten()] = 1
    #     features = sp.lil_matrix(data['X']+1)
    #     adj = data['G']

    def parse_index_file(filename):
        """Parse index file."""
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    g = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    # features = sp.lil_matrix(allx)

    labels = np.vstack((ally, ty))
    return features, labels, adj

def preprocess_features(features, feature_type):
    # """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    # d_inv_sqrt = np.power(rowsum, -0.5)
    # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # return adj*d_inv_sqrt*d_inv_sqrt.flatten()
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def laplacian_smooth(features, adj, lam, mu, repeat=1):
    n = adj.shape[0]
    smoothor = sp.diags(np.ones(n)) * (1-lam) + lam * adj
    inflator = sp.diags(np.ones(n)) * (1-mu) + mu * adj
    transformor = smoothor*inflator
    for i in range(repeat):
        features = transformor * features
    return features

if __name__ == '__main__':
    dataset = 'cora'
    features, labels, adj = load_data(dataset)
    y=np.argmax(labels, axis=1)
    n = adj.shape[0]
    adj_I = adj + sp.diags(np.ones(n))
    # d_I = np.diag(np.array(adj.sum(1)))

    adj_rw = normalize(adj, norm='l1', axis=1)
    adj_I_rw = normalize(adj, norm='l1', axis=1)
    adj_I_sym = normalize_adj(adj_I) #
    d_I = np.array(adj_I.sum(1)).reshape(-1)
    d_I_inv_sqrt = np.power(d_I, -0.5)
    d_I = np.diag(d_I)
    d_I_inv_sqrt = np.diag(d_I_inv_sqrt)

    # vals, vecs = linalg.eig(adj_rw.toarray())
    # vals = vals.astype(np.float64)
    # print(np.min(vals),np.max(vals))
    # plt.hist(vals, histtype='step')
    # vals, vecs = linalg.eig(adj_I_rw.toarray())
    # vals = vals.astype(np.float64)
    # print(np.min(vals),np.max(vals))
    # plt.hist(vals, histtype='step')
    # plt.show()


    try:
        vals, vecs, vec_inv = np.load(dataset+'_vals.npy'), \
                              np.load(dataset+'_vecs.npy'), \
                              np.load(dataset + '_vec_inv.npy')
        print('load vals, vecs, vec_inv from files')
    except:
        vals, vecs = linalg.eigh(adj_I_sym.toarray())
        vecs = d_I_inv_sqrt.dot(vecs)
        vec_inv = np.linalg.inv(vecs)

        # vals = vals.astype(np.float64)
        # vec_inv = vec_inv.astype(np.float64)
        np.save(dataset+'_vals.npy', vals)
        np.save(dataset+'_vecs.npy', vecs)
        np.save(dataset+'_vec_inv.npy', vec_inv)

    #unsmoothed
    ax = plt.subplot(321)
    plt.title('unsmoothed')
    # ax.set_yscale("log")
    # plt.axis([-0.1, 1.6, 0.1, 1e7])
    transformed = vec_inv*features
    # transformed = transformed.sum(1).reshape([n,1])
    # for i in range(transformed.shape[1]):
    #     plt.scatter(-(vals-1), np.abs(transformed[:,i]), s=2)

    # #graph convolution
    # ax = plt.subplot(322)
    # plt.title('smoothed by graph convolution')
    # # ax.set_yscale("log")
    # # plt.axis([-0.1, 1.6, 0.1, 1e7])
    # transformed_gc = transformed*vals.reshape([-1,1])*vals.reshape([-1,1])
    # for i in range(transformed_gc.shape[1]):
    #     plt.scatter(-(vals-1), np.abs(transformed_gc[:,i]), s=2)
    #
    # #lp
    # ax = plt.subplot(323)
    # plt.title('smoothed by absorption probability')
    # # ax.set_yscale("log")
    # # plt.axis([-0.1, 1.6, 0.1, 1e7])
    # transformed_lp = transformed*np.divide(0.2, 1.2-vals.reshape([-1,1]))
    # for i in range(transformed_lp.shape[1]):
    #     plt.scatter(-(vals-1), np.abs(transformed_lp[:,i]), s=2)
    #
    # # laplacian smooth
    # ax = plt.subplot(324)
    # plt.title('laplacian smooth')
    # # ax.set_yscale("log")
    # # plt.axis([-0.1, 1.6, 0.1, 1e7])
    # vals_ls = (0.3*vals.reshape([-1, 1]) + 0.7)*(-0.31*vals.reshape([-1, 1]) + 1.31)
    # vals_ls = np.power(vals_ls, 100)
    # transformed_ls = transformed * vals_ls
    # for i in range(transformed_ls.shape[1]):
    #     plt.scatter(-(vals - 1), np.abs(transformed_ls[:, i]), s=2)

    # label
    ax = plt.subplot(325)
    plt.title('Decomposition of Y')
    # ax.set_yscale("log")
    # plt.axis([-0.1, 1.6, 0.1, 1e7])
    transformed_y = vec_inv.dot(labels)
    # transformed_y = transformed_y.sum(1).reshape([n,1])
    # for i in range(transformed_y.shape[1]):
    #     plt.scatter(-(vals - 1), np.abs(transformed_y[:, i]), s=2)

    # label
    ax = plt.subplot(326)
    plt.title('Y/X')
    # ax.set_yscale("log")
    # plt.axis([-0.1, 1.6, 0.1, 1e7])
    # transformed_y = vec_inv.dot(labels)
    # transformed_y = transformed_y.sum(1).reshape([n,1])
    for i in range(transformed_y.shape[1]):
        ax = plt.subplot(111)
        plt.title('Y/X')
        accumulation_x = np.zeros(transformed[:, 0].shape)
        for j in range(transformed.shape[1]):
            accumulation_x+=transformed[:, j]*transformed[:, j]
        plt.scatter(-(vals - 1), np.log1p(np.abs(transformed_y[:, i]/np.sqrt(accumulation_x))), s=2)
        # plt.show()

    plt.subplots_adjust(hspace=0.4)
    plt.show()

    # lam = 0.3
    # mu = -0.31
    # repeat = 20
    # projector = TSNE(n_components=2, init='pca')
    # itr = 5
    #
    # smoothed = features
    # for i in range(itr):
    #     emb = projector.fit_transform(smoothed.toarray())
    #     # emb = smoothed.toarray()
    #     plt.subplot(4,itr,i+1)
    #     plt.scatter(emb[:,0], emb[:,1], c=y, cmap='Spectral', s=4)
    #     smoothed = laplacian_smooth(smoothed, adj_rw, lam, mu=0, repeat=repeat)
    #
    # smoothed = features
    # for i in range(itr):
    #     emb = projector.fit_transform(smoothed.toarray())
    #     # emb = smoothed.toarray()
    #     plt.subplot(4,itr,i+1+itr)
    #     plt.scatter(emb[:,0], emb[:,1], c=y, cmap='Spectral', s=4)
    #     smoothed = laplacian_smooth(smoothed, adj_I_rw, lam, mu=0, repeat=repeat)
    #
    # smoothed = features
    # for i in range(itr):
    #     emb = projector.fit_transform(smoothed.toarray())
    #     # emb = smoothed.toarray()
    #     plt.subplot(4,itr,i+1+2*itr)
    #     plt.scatter(emb[:,0], emb[:,1], c=y, cmap='Spectral', s=4)
    #     smoothed = laplacian_smooth(smoothed, adj_rw, lam, mu, repeat=repeat)
    #
    # smoothed = features
    # for i in range(itr):
    #     emb = projector.fit_transform(smoothed.toarray())
    #     # emb = smoothed.toarray()
    #     plt.subplot(4,itr,i+1+3*itr)
    #     plt.scatter(emb[:,0], emb[:,1], c=y, cmap='Spectral', s=4)
    #     smoothed = laplacian_smooth(smoothed, adj_I_rw, lam, mu, repeat=repeat)
    # plt.show()