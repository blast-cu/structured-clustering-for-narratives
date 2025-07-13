import argparse
import collections
import math
import pickle
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from pyhocon import ConfigFactory
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from torch import optim
from torch.nn import Parameter
from tqdm import tqdm

from utils.constraint_flat_db import ConstraintFlatDB
from clustering.metrics.purity import Purity
from models.regression import RegressionModel


class MSELoss(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()

    def forward(self, input, target):
        return torch.mean((input - target) ** 2)


def build_network(layers, activation="relu", dropout=0):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i - 1], layers[i]))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)


class DCC(nn.Module):
    def __init__(self, config, input_dim=784, z_dim=10, n_clusters=10,
                 encodeLayer=[400], decodeLayer=[400], activation="relu", dropout=0, alpha=1., gamma=0.1):
        super(self.__class__, self).__init__()
        
        self.config = config

        seed = self.config["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.z_dim = z_dim
        self.layers = [input_dim] + encodeLayer + [z_dim]
        self.activation = activation
        self.dropout = dropout
        self.encoder = build_network([input_dim] + encodeLayer, activation=activation, dropout=dropout)
        self.decoder = build_network([z_dim] + decodeLayer, activation=activation, dropout=dropout)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._dec = nn.Linear(decodeLayer[-1], input_dim)

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.gamma = gamma
        self.mu = Parameter(torch.Tensor(n_clusters, z_dim))

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, x):
        h = self.encoder(x)
        z = self._enc_mu(h)
        h = self.decoder(z)
        xrecon = self._dec(h)
        # compute q -> NxK
        q = self.soft_assign(z)
        return z, q, xrecon

    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha)
        q = q ** (self.alpha + 1.0) / 2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

    def encode_batch(self, X, batch_size=256):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        encoded = []
        self.eval()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))
        for batch_idx in range(num_batch):
            inputs = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
            z, _, _ = self.forward(inputs)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded

    def cluster_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=1))

        kldloss = kld(p, q)
        return self.gamma * kldloss

    @staticmethod
    def recon_loss(x, xrecon):
        recon_loss = torch.mean((xrecon - x) ** 2)
        return recon_loss

    @staticmethod
    def pairwise_loss(p1, p2):
        cl_loss = torch.mean(-torch.log(1.0 - torch.sum(p1 * p2, dim=1)))
        return cl_loss

    @staticmethod
    def target_distribution(q):
        p = q ** 2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    @staticmethod
    def satisfied_constraints(cl_ind1, cl_ind2, y_pred):

        if cl_ind1.size == 0 or cl_ind2.size == 0:
            return 1.1

        count = 0
        satisfied = 0
        for (i, j) in zip(cl_ind1, cl_ind2):
            count += 1
            if y_pred[i] != y_pred[j]:
                satisfied += 1

        return float(satisfied) / count

    @staticmethod
    def get_constraint_pairs(cl_constraints: List[Tuple[int, int]]):
        print("Computing cannot-link constraint indices...", flush=True)
        cl_ind1 = []
        cl_ind2 = []
        for (i, j) in cl_constraints:
            cl_ind1.append(i)
            cl_ind2.append(j)

        return np.array(cl_ind1), np.array(cl_ind2)

    def fit(self, anchor=None, positive=None, negative=None,
            cl_ind1=None, cl_ind2=None, cl_penalty=None, X=None, lr=0.001, batch_size=256,
            num_epochs=10, update_interval=1, tol=1e-3, use_kmeans=True,
            clustering_loss_weight=1):

        '''X: tensor data'''
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        X = X.to(self.device)
        print("=====Training DCC=======", flush=True)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

        if use_kmeans:
            print("Initializing cluster centers with kmeans...", flush=True)
            kmeans = KMeans(self.n_clusters, n_init=20)
            data = self.encode_batch(X)
            y_pred = kmeans.fit_predict(data.data.cpu().numpy())
            y_pred_last = y_pred
            self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))
        else:
            # use kmeans to randomly initialize cluster ceters
            print("Randomly initializing cluster centers...", flush=True)
            kmeans = KMeans(self.n_clusters, n_init=1, max_iter=1)
            data = self.encode_batch(X)
            y_pred = kmeans.fit_predict(data.data.cpu().numpy())
            y_pred_last = y_pred
            self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))

        self.train()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))
        cl_num_batch = int(math.ceil(1.0 * cl_ind1.shape[0] / batch_size))
        cl_num = cl_ind1.shape[0]

        update_cl = 1
        
        for epoch in range(num_epochs):
            if epoch % update_interval == 0:
                # update the targe distribution p
                latent = self.encode_batch(X)
                q = self.soft_assign(latent)
                p = self.target_distribution(q).data

                y_pred = torch.argmax(q, dim=1).data.cpu().numpy()

                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / num
                y_pred_last = y_pred
                if epoch > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol, flush=True)
                    print("Reached tolerance threshold. Stopping training.", flush=True)

                    break

            # train 1 epoch for clustering and reconstruction loss
            train_loss = 0.0
            recon_loss_val = 0.0
            cluster_loss_val = 0.0

            print("Computing Clustering and Reconstruction Loss...", flush=True)
            for batch_idx in tqdm(range(num_batch)):
                inputs = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
                target = p[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
                optimizer.zero_grad()

                z, qbatch, xrecon = self.forward(inputs)

                cluster_loss = self.cluster_loss(target, qbatch)
                recon_loss = self.recon_loss(inputs, xrecon)

                loss = cluster_loss + recon_loss
                loss.backward()
                optimizer.step()

                cluster_loss_val += cluster_loss.data * len(inputs)
                recon_loss_val += recon_loss.data * len(inputs)
                train_loss = clustering_loss_weight * cluster_loss_val + recon_loss_val

            print("#Epoch %3d: Total: %.4f Clustering Loss: %.4f Reconstruction Loss: %.4f" % (
                epoch + 1, train_loss / num, cluster_loss_val / num, recon_loss_val / num), flush=True)

            # train 1 epoch for pairwise cannot link constraint loss
            print("Computing Pairwise Loss...", flush=True)
            cl_loss = 0.0
            if epoch % update_cl == 0 and cl_penalty > 0:
                for cl_batch_idx in tqdm(range(cl_num_batch)):
                    inputs1 = X[cl_ind1[cl_batch_idx * batch_size: min(cl_num, (cl_batch_idx + 1) * batch_size)]]
                    inputs2 = X[cl_ind2[cl_batch_idx * batch_size: min(cl_num, (cl_batch_idx + 1) * batch_size)]]
                    optimizer.zero_grad()
                    z1, q1, xr1 = self.forward(inputs1)
                    z2, q2, xr2 = self.forward(inputs2)
                    loss = cl_penalty * self.pairwise_loss(q1, q2)
                    cl_loss += loss.data
                    loss.backward()
                    optimizer.step()

            if cl_penalty > 0:
                print("CL loss:", float(cl_loss.cpu()), flush=True)

        return y_pred

    def save(self, config, y_pred, weight_pairwise):
        """Save the model to a file."""

        print("Saving clustering results...", flush=True)

        output = {
            "number_cluster": self.n_clusters,
            "w_cl": weight_pairwise,
            "cluster_centers": self.mu.data.cpu().numpy(),
            "labels": y_pred
        }

        with open(config["clusters_path"] + f"dcc_clusters_{self.n_clusters}_{weight_pairwise}.pickle", 'wb') as f:
            pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

    def predict(self, X):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        latent = self.encode_batch(X)
        q = self.soft_assign(latent)

        y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
        return y_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Constrained Clustering')
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see config.conf)')
    parser.add_argument('-k', metavar='N_CLUSTERS', default=5, type=int, help='number of clusters')
    parser.add_argument('--epochs', metavar='EPOCHS', default=100, type=int, help='maximum number of epochs')
    parser.add_argument('--weight_pairwise', metavar='WEIGHT_PAIRWISE', default=0.5, type=float,
                        help='weight for cannot-link constraints')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')

    args = parser.parse_args()
    config = ConfigFactory.parse_file('./config.conf')[args.c]

    print("N_CLUSTERS: " + str(args.k), flush=True)
    print("WEIGHT_PAIRWISE: " + str(args.weight_pairwise), flush=True)
    print("EPOCHS: " + str(args.epochs), flush=True)
    print("BATCH_SIZE: " + str(args.batch_size), flush=True)

    print("Loading data for clustering...", flush=True)

    # Load data
    with open(config["processed_chains_path"], 'rb') as f:
        data = pickle.load(f)

    print("Generating embeddings for sentences...", flush=True)
    # Initialize embeddings and constraints
    sbert_model = SentenceTransformer(config["cluster_model"])
    X = sbert_model.encode(
        data["chain_sents"], batch_size=32, show_progress_bar=True, normalize_embeddings=True
    )

    dcc = DCC(config, input_dim=384, z_dim=100, n_clusters=args.k, encodeLayer=[192],
              decodeLayer=[192], activation="relu", dropout=0)

    print("Load constraints...", flush=True)
    sorted_constraints = ConstraintFlatDB(config["constraints_flat_path"]).get_all_tuples_as_list()
    cl_ind1, cl_ind2 = DCC.get_constraint_pairs(sorted_constraints)

    y_pred = dcc.fit(cl_ind1=cl_ind1, cl_ind2=cl_ind2, X=X, cl_penalty=args.weight_pairwise, num_epochs=args.epochs,
                     batch_size=args.batch_size)

    # Compute and print purity results after clustering is complete
    print("\n=== Purity Results ===", flush=True)
    clustering_data = {
        "number_cluster": dcc.n_clusters,
        "labels": y_pred,
        "embeddings": X,
        "cluster_centers": dcc.mu.data.cpu().numpy()
    }
    purity_calculator = Purity(config["processed_chains_path"], clustering_data)
    purity_calculator.compute_purity()
    purity_calculator.print_results()
    print("======================\n", flush=True)

    # Run regression model after purity computation
    print("\n=== Regression Results ===", flush=True)
    regression_model = RegressionModel(config)
    data = regression_model.create_dataset(config, clustering_data)
    test_accuracy, f1_score = regression_model.regression(config, data)
    print("==========================\n", flush=True)

    dcc.save(config, y_pred, args.weight_pairwise)