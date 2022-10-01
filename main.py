# %%
import random
import torch as T
import torch.nn as N
import torch.optim as optim
import numpy as np
from Bio import SeqIO
import torch.nn.functional as F
import torch.utils.data as D
from torchsummary import summary


# %%
length = 128
sample = 100
base_dict = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
file_list = ['GCF_000018685.1_ASM1868v1_genomic.fna',
             'GCF_000019145.1_ASM1914v1_genomic.fna',
             'GCF_000016965.1_ASM1696v1_genomic.fna']

# %%
sequence_matrix = np.zeros(shape=(len(file_list) * sample, length), dtype='int64')
sequence_label = list()
index = 0
for filename in file_list:
    print(filename)
    fasta_iterator = SeqIO.parse(filename, 'fasta')
    fasta_sequence = next(fasta_iterator)
    name, sequence = fasta_sequence.description, str(fasta_sequence.seq)
    print(name, len(sequence))
    # print(count_kmers(sequence, 4))
    sequence = [base_dict[i] if i in base_dict else 5 for i in sequence]
    for i in range(0, sample):
        rand_number = random.randint(0, len(sequence) - length)
        sub_sequence = sequence[rand_number:rand_number + length]
        sequence_matrix[i + index * sample, :] = sub_sequence
        sequence_label.append(index)
    index = index + 1


# %%
class AE(N.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_1 = N.Conv1d(in_channels=5, out_channels=128, kernel_size=5, stride=1,padding=2)
        self.encoder_2 = N.MaxPool1d(kernel_size=4, stride=4, return_indices=True)
        self.encoder_3 = N.Conv1d(in_channels=128, out_channels=32, kernel_size=5, stride=1,padding=2)
        self.encoder_4 = N.MaxPool1d(kernel_size=4, stride=4, return_indices=True)
        self.encoder_5 = N.Conv1d(in_channels=32, out_channels=8, kernel_size=5, stride=1,padding=2)
        self.encoder_6 = N.MaxPool1d(kernel_size=4, stride=4, return_indices=True)
        self.encoder_7 = N.Conv1d(in_channels=8, out_channels=2, kernel_size=5, stride=1,padding=2)

        # self.encoder_s = N.AvgPool1d(kernel_size=2, stride=2)

        self.decoder_1 = N.ConvTranspose1d(in_channels=2, out_channels=8, kernel_size=5, stride=1,padding=2)
        self.decoder_2 = N.MaxUnpool1d(kernel_size=4, stride=4)
        self.decoder_3 = N.ConvTranspose1d(in_channels=8, out_channels=32, kernel_size=5, stride=1,padding=2)
        self.decoder_4 = N.MaxUnpool1d(kernel_size=4, stride=4)
        self.decoder_5 = N.ConvTranspose1d(in_channels=32, out_channels=128, kernel_size=5, stride=1,padding=2)
        self.decoder_6 = N.MaxUnpool1d(kernel_size=4, stride=4)
        self.decoder_7 = N.ConvTranspose1d(in_channels=128, out_channels=5, kernel_size=5, stride=1,padding=2)

    def forward(self, x0):
        x1 = T.relu(self.encoder_1(x0))
        x2, i2 = self.encoder_2(x1)
        x3 = T.relu(self.encoder_3(x2))
        x4, i4 = self.encoder_4(x3)
        x5 = T.relu(self.encoder_5(x4))
        x6, i6 = self.encoder_6(x5)
        x7 = T.relu(self.encoder_7(x6))

        # encoded = T.relu(self.encoder_8(x7))

        x8 = T.relu(self.decoder_1(x7))
        x9 = self.decoder_2(x8, i6)
        x10 = T.relu(self.decoder_3(x9))
        x11 = self.decoder_4(x10, i4)
        x12 = T.relu(self.decoder_5(x11))
        x13 = self.decoder_6(x12, i2)
        x14 = T.relu(self.decoder_7(x13))

        return x14


# %%
#  use gpu if available
device = T.device("mps" if T.has_mps else "cpu")
#device = 'cpu'

# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
full_model = AE(input_shape=(length, 5)).to(device)
summary(AE(input_shape=(length, 5)).to('cpu'), input_size=(5, 128))

# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = optim.Adam(full_model.parameters(), lr=1e-3)

# mean-squared error loss
criterion = N.MSELoss()

# %%
sequence_tensor = T.from_numpy(sequence_matrix).to(device)

class SequenceDataset(D.Dataset):
    def __init__(self):
        self.sequence_tensor = sequence_tensor
        self.sequence_one_hot = F.one_hot(self.sequence_tensor, num_classes=5)
        self.sequence_one_hot = self.sequence_one_hot.float()

    def __len__(self):
        return len(self.sequence_one_hot)

    def __getitem__(self, idx):
        return T.transpose(self.sequence_one_hot[idx, ...], 0, 1)


# %%
train_dataloader = D.DataLoader(SequenceDataset(), batch_size=64, shuffle=True)

epochs = 100
for epoch in range(epochs):
    loss = 0
    for inputs in train_dataloader:
        optimizer.zero_grad()

        outputs = full_model(inputs)

        train_loss = criterion(outputs, inputs)

        train_loss.backward()

        # perform parameter update based on current gradients
        optimizer.step()

        loss += train_loss.item()

    # compute the epoch training loss
    loss = loss / len(train_dataloader)

    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
