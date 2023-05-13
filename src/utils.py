import copy
import numpy as np
import torch

from sklearn.cluster import DBSCAN, OPTICS
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

def get_latent_code(model, latent_size, loss_func, X):
    latent_code = {}
    def get_latent_code(name):
        def hook(model, input, output):
            latent_code[name] = input[0]
        return hook

    model.decoder_1.register_forward_hook(get_latent_code('decoder_1'))

    model.eval()
    with torch.no_grad():
        latent_codes = np.array([[0] * latent_size])
        losses = []
        for _, data in enumerate(torch.utils.data.DataLoader( X, batch_size=1, shuffle=False)):
            output = model(data.float())
            loss = torch.sqrt(loss_func(output, data.float()))
            latent_codes = np.append(latent_codes, np.array(latent_code['decoder_1'].tolist()), axis=0)
            losses.append(loss.item())
    return latent_codes[1:], losses

def get_anomalies_loss(loss, size, time_shift=10, sigma_threshold=1):
    # Removing seasonality
    loss = np.array(loss) - np.array([*loss[time_shift:], *([0]*time_shift)])

    loss_mean  = np.mean(loss)
    loss_sigma = np.std(loss)

    anomaly_idx = np.where(loss - loss_mean > sigma_threshold * loss_sigma)[0]

    anomaly = np.array([-1] * size)
    anomaly[anomaly_idx] = 1

    return anomaly

def get_anomalies_dbscan(latent_codes):
    db = OPTICS(cluster_method='dbscan')
    db.fit(latent_codes)

    anomaly = np.array([-1] * len(latent_codes))
    anomaly[db.labels_ == -1] = 1
    
    return anomaly

def get_anomalies_forest(latent_codes):
    forest = IsolationForest(random_state=0)
    anomaly = forest.fit_predict(latent_codes) * -1
    return anomaly

def train(
    model, optimizer, loss_func, dataloader, epochs, 
    early_stopping_patience, early_stopping_delta, 
    verbose=False, visualize_latent=False, X_test=[], latent_size=None):
    
    losses = []
    effective_dimensions = []

    best_model = None
    best_loss = 1e4

    for epoch in range(epochs):
        epoch_loss = []

        # Training
        model.train()
        for _, data in enumerate(dataloader):
            optimizer.zero_grad()
            # Forward pass
            output = model(data.float())
            # Calculating loss
            loss = torch.sqrt(loss_func(output, data.float()))
            # Backward pass
            loss.backward()
            # Optimize the weights
            optimizer.step()
            
            epoch_loss.append(loss.item())

        total_loss = sum(epoch_loss)
        losses.append(total_loss)

        if best_loss > total_loss:
            best_loss = total_loss
            best_model = copy.deepcopy(model)

        # Early stopping
        if epoch > early_stopping_patience and \
            np.abs(np.diff(losses[-early_stopping_patience:])).sum() <= early_stopping_patience * early_stopping_delta:
            print('Early stopping')
            break

        # Show information
        if verbose:
            current_progress = int(epoch / epochs * 100)
            if current_progress % 10 == 0:
                print('Training [{:>3}%]  Loss: {:.5}'.format(current_progress, total_loss))
                
                if visualize_latent:
                    latent_codes, _ = get_latent_code(model, latent_size, loss_func, X_test)
                    
                    all_unique_tupes = [(i, j) for i in range(0, latent_size + 1) for j in range(i, latent_size) if i != j]
                    n_rows = 2
                    n_columns = int((len(all_unique_tupes) + 1) / 2)

                    fig, axes = plt.subplots(nrows=n_rows, ncols=n_columns, figsize=(20, 7))
                    fig.suptitle('Latent encoding of testing data', fontsize=16)

                    for index, (i, j) in enumerate(all_unique_tupes):
                        row = int(index > n_columns - 1)
                        column = index - row * n_columns

                        axes[row, column].scatter(x=latent_codes[:,i], y=latent_codes[:,j], alpha=0.3)
                        axes[row, column].set_title("Components " + str((i, j)))
                    
                    plt.show()
    return best_model, losses