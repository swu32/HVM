# Compare the sequence parsing complexity of a neural network to the complexity learned by the HCM and HVM.
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out


def slicer(seq, size):
    """Divide the sequence into chunks of the given size."""
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def plot_learning_comparison(datann, sz = 5, savename = 'modelcomparisonall.png'):
    import matplotlib.pyplot as plt
    import numpy as np

    datahcm = np.load('./data/hcm'+ ' sz = ' + str(sz) + '.npy')
    datahvm = np.load('./data/hvm'+ ' sz = ' + str(sz) + '.npy')

    # both are three dimensional arrays

    titles = ['parsing length', 'representation complexity', 'explanatory volume', 'sequence complexity',
              'representation entropy', 'n chunks', 'n variables', 'storage cost']

    units = ['n chunk', 'bits', 'l', 'bits', 'bits', 'n chunk', 'n variable', 'bits']
    # Create a figure and subplots with 2 rows and 3 columns
    fig, axs = plt.subplots(2, 4, figsize=(10, 6))
    x = np.cumsum(datahcm[0,:, 0])

    for i, ax in enumerate(axs.flat):
        if i >= 8:
            break
        hcm_mean = np.mean(datahcm[:, :, i + 1], axis=0)
        hvm_mean = np.mean(datahvm[:, :, i + 1], axis=0)
        nn_mean = np.mean(datann[:, :, 0], axis=0)

        ax.plot(x, hcm_mean, label='HCM', color='orange', linewidth=4, alpha=0.3)
        ax.plot(x, hvm_mean, label='HVM', color='blue', linewidth=4, alpha=0.3)
        if i == 3:
            ax.plot(x,nn_mean, label='NN', color='green', linewidth=4, alpha=0.3)
            for j in range(0, datann.shape[0]):
                ax.plot(x, datann[j, :, 0], color='green', linewidth=1, alpha=0.3)

        for j in range(0, datahcm.shape[0]):
            ax.plot(x, datahcm[j, :, i + 1], color='orange', linewidth=1, alpha=0.3)
            ax.plot(x, datahvm[j, :, i + 1], color='blue', linewidth=1, alpha=0.3)

        ax.set_title(titles[i])
        ax.set_ylabel(units[i])
        ax.set_xlabel('Sequence Length')
    # Adjust spacing between subplots
    fig.tight_layout()
    # Show the figure
    plt.legend()
    plt.show()
    # save the figure
    fig.savefig(savename)

    return


def train_nn_alphabet_increase():
    # Define hyperparameters and model
    input_size = 1
    hidden_size = 50
    output_size = 10

    size_increment = [5, 10, 15, 20, 25, 30, 35, 40]
    for sz in size_increment:
        openpath = './generative_sequences/random_abstract_sequence' + ' d = ' + str(sz) + '.npy'
        with open(openpath, 'rb') as f:
            fullseq = np.load(f)
        slice_sz = 1000
        n_measure = 1 # just measure the complexity
        n_iter = int(len(fullseq)/slice_sz)
        n_epoch = 14
        datann = np.empty((n_iter, n_epoch, n_measure))
        i = 0 # in each iteration, use the same data for training 14 number of epoches
        for seq in slicer(fullseq, slice_sz): # the same sequence as in
            data = torch.tensor(seq, dtype=torch.int)
            # Prepare inputs and targets
            X = data[:-1].view(1, -1, 1).float()
            y = data[1:].view(1, -1).long()
            output_size = sz + 1 # the number of unique elements in the sequence
            # initialize a model
            model = LSTM(input_size, hidden_size, output_size)
            criterion = nn.CrossEntropyLoss(reduction = 'sum') # need to evaluate sequence complexity sum
            optimizer = optim.Adam(model.parameters(), lr=0.01)

            # Train the model
            for epoch in range(n_epoch):
                outputs = model(X)
                loss = criterion(outputs.view(-1, output_size), y.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_value = criterion(outputs.view(-1, output_size), y.view(-1))
                print(f"Sequence Complexity (using cross-entropy): {loss_value.item()}")
                datann[i, epoch, 0] = loss_value.item()
            i = i + 1
        np.save('./data/nn' + ' sz = ' + str(sz) + '.npy', datann)
        plot_learning_comparison(datann, sz=sz, savename='./data/plusNN' + ' sz = ' + str(sz) + '.png')
    return


def train_nn_depth_increase():
    # Define hyperparameters and model
    input_size = 1
    hidden_size = 50
    output_size = 10

    size_increment = [5, 10, 15, 20, 25, 30, 35, 40]
    for sz in size_increment:
        openpath = './generative_sequences/random_abstract_sequence_fixed_support_set' + ' d = ' + str(sz) + '.npy'
        with open(openpath, 'rb') as f:
            fullseq = np.load(f)
        slice_sz = 1000
        n_measure = 1 # just measure the complexity
        n_iter = int(len(fullseq)/slice_sz)
        n_epoch = 14
        datann = np.empty((n_iter, n_epoch, n_measure))
        i = 0 # in each iteration, use the same data for training 14 number of epoches
        for seq in slicer(fullseq, slice_sz): # the same sequence as in
            data = torch.tensor(seq, dtype=torch.int)
            # Prepare inputs and targets
            X = data[:-1].view(1, -1, 1).float()
            y = data[1:].view(1, -1).long()
            output_size = sz + 1 # the number of unique elements in the sequence
            # initialize a model
            model = LSTM(input_size, hidden_size, output_size)
            criterion = nn.CrossEntropyLoss(reduction = 'sum') # need to evaluate sequence complexity sum
            optimizer = optim.Adam(model.parameters(), lr=0.01)

            # Train the model
            for epoch in range(n_epoch):
                outputs = model(X)
                loss = criterion(outputs.view(-1, output_size), y.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_value = criterion(outputs.view(-1, output_size), y.view(-1))
                print(f"Sequence Complexity (using cross-entropy): {loss_value.item()}")
                datann[i, epoch, 0] = loss_value.item()
            i = i + 1
        np.save('./data/nn_fixed_support_set' + ' d = ' + str(sz) + '.npy', datann)
    return



train_nn_depth_increase()


#train_nn_alphabet_increase()



