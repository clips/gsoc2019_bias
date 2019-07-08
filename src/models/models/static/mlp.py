import pandas
import torch
import torch.utils.data as data_utils
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import Adam

from src.models.data.idf_dataset import IDFDataset
from src.models.models.model import Classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class OneNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, learning_rate):
        super(OneNet, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size, bias=True)
        self.relu = nn.modules.activation.LeakyReLU()
        self.output_layer = nn.Linear(hidden_size, num_classes, bias=True)

        self.learning_rate = learning_rate

    def forward(self, x):
        out = self.layer_1(x)
        out = self.relu(out)
        out = self.output_layer(out)
        return out

    def get_net_crit_opt(self):
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return self, criterion, optimizer

class TwoNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, learning_rate):
        super(TwoNet, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size, bias=True)
        self.layer_2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.relu = nn.modules.activation.LeakyReLU()
        self.output_layer = nn.Linear(hidden_size, num_classes, bias=True)

        self.learning_rate = learning_rate

    def forward(self, x):
        out = self.layer_1(x)
        out = self.relu(out)
        out = self.layer_2(out)
        out = self.relu(out)
        out = self.output_layer(out)
        return out

    def get_net_crit_opt(self):
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return self, criterion, optimizer

class MLPWrapper(Classifier):
    def __init__(self, model, hidden_size = 1000, learning_rate = 0.0002, epochs = 5, batch_size = 500):
        self.model = model
        self.hidden_size = hidden_size
        self.learning_rate= learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, idf : IDFDataset):
        train = idf.get_train_dataset_torch()

        tokenizer = idf.tokenizer

        train_loader = data_utils.DataLoader(train, batch_size=self.batch_size)

        input_size = len(tokenizer.vocabulary_)
        num_classes = idf.num_labels

        self.network, self.criterion, optimizer = self.model(input_size, self.hidden_size, num_classes, self.learning_rate).get_net_crit_opt()

        train_loss = []
        for epoch in range(self.epochs):
            print(epoch)
            curr_loss = None
            for i, data in enumerate(train_loader):
                inputs, labels = data

                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = self.network(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if(curr_loss is None):
                    curr_loss = loss.item()
                else:
                    curr_loss = (curr_loss + loss.item())/2

            train_loss.append(curr_loss)

        return train_loss

    def predict(self, x_test):
        self.network.eval()
        with torch.no_grad():
            input = torch.from_numpy(x_test)
            outputs = self.network(input.float())
            _, outputs = torch.max(outputs.data, 1)
        self.network.train()
        return outputs.numpy()

    def predict_one(self, x_single):
        self.network.eval()
        output = self.network(x_single)
        self.network.train()
        return output

    def load_model(self, filename: str):
        pass

    def save_model(self, filename: str):
        pass

    def score(self, dataset):
        self.network.eval()

        accuracy, loss = None, 0

        test_loader = data_utils.DataLoader(dataset, batch_size=self.batch_size)

        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self.network(inputs)
                if(loss == 0):
                    loss = self.criterion(outputs, labels).item()
                else:
                    loss = (loss + self.criterion(outputs, labels).item())/2
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        self.network.train()
        accuracy = correct / total
        return accuracy, loss

if __name__ == "__main__":
    print("Loading file")
    frame = pandas.read_csv("hatespeech-data.csv", sep="\t", header=None, names=["texts", "labels"], usecols=(0, 1))

    print("Tokenizing data")
    dataset = IDFDataset()
    dataset.load_data(frame, 0.25)

    print("Training classifier")
    mlp = MLPWrapper(OneNet)
    mlp.train(dataset)
    print("Accuracy {}".format(mlp.score(dataset.get_test_dataset_torch())[0]))
    predicted = mlp.predict(dataset.get_test_dataset()[0].todense())
    print(len(predicted))
    print(len(dataset.get_test_dataset()[1]))
    print("F1 Score {}".format(f1_score(dataset.get_test_dataset()[1], predicted, average='weighted')))

    # n = OneNet
    # _, test_accuracy, test_loss, train_accuracy, train_loss = train_nn(OneNet)
    # plotter.plot_loss_graph(train_loss, test_loss)

