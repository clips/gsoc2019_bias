import torch
from torch import nn
from torch.optim import Adam, SGD
import torch.utils.data as data_utils

from models.src.data.token_data_loader import get_token_dataset
from models.src.utils import plotter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class One_Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, learning_rate):
        super(One_Net, self).__init__()
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

class Two_Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, learning_rate):
        super(Two_Net, self).__init__()
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

def train_nn(model, hidden_size = 1000, learning_rate = 0.0001, epochs = 20):
    BATCH = 200

    train, test = get_token_dataset()
    train.make_tensor_set()
    test.make_tensor_set()

    tokenizer = train.tokenizer

    train_loader = data_utils.DataLoader(train, batch_size=BATCH)
    test_loader = data_utils.DataLoader(test, batch_size=BATCH)

    print(train.num_labels)
    input_size = len(tokenizer.vocabulary_)
    num_classes = 4

    network, criterion, optimizer = model(input_size, hidden_size, num_classes, learning_rate).get_net_crit_opt()


    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []


    accuracy, loss = test_nn_classifier(network, criterion, train_loader, tokenizer, train_data=True)
    train_accuracy.append(accuracy)
    train_loss.append(loss)
    accuracy, loss = test_nn_classifier(network, criterion, test_loader, tokenizer)
    test_accuracy.append(accuracy)
    test_loss.append(loss)

    best_acc = accuracy

    print()

    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        accuracy, loss = test_nn_classifier(network, criterion, train_loader, tokenizer, train_data=True)
        train_accuracy.append(accuracy)
        train_loss.append(loss)
        accuracy, loss = test_nn_classifier(network, criterion, test_loader, tokenizer)
        test_accuracy.append(accuracy)
        test_loss.append(loss)

        if(accuracy > best_acc):
            print("New best model found, epoch: " + str(epoch))
            best_acc = accuracy

        print()

    print('Training finished')

    test_nn_classifier(network, criterion, test_loader, tokenizer, plot = True)

    return model, test_accuracy, test_loss, train_accuracy, train_loss

def test_nn_classifier(network, criterion, loader, vectorizer, train_data = False, plot = False):
    network.eval()

    accuracy, loss = None, 0

    correct = 0
    total = 0
    confusion_matrix = torch.zeros(4, 4)
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = network(inputs)
            if(loss == 0):
                loss = criterion(outputs, labels).item()
            else:
                loss = (loss + criterion(outputs, labels).item())/2
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for t, p in zip(labels.view(-1), predicted.view(-1)):
               confusion_matrix[t.long(), p.long()] += 1

    network.train()

    plotter.plot_confusion_matrix(confusion_matrix.numpy())

    accuracy = correct / total

    if(train_data):
        print("Training Accuracy: %.2f, Training Loss: %2f" % (accuracy*100, loss))
    else:
        print("Test Accuracy: %.2f, Test Loss: %2f" % (accuracy*100, loss))

    return accuracy, loss

if __name__ == "__main__":
    n = One_Net
    _, test_accuracy, test_loss, train_accuracy, train_loss = train_nn(One_Net)
    plotter.plot_loss_graph(train_loss, test_loss)

