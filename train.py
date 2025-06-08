import torch
import utils
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, sampler
from model import MLP, ResNet18, CNN


# boilerplate mainly copy-pasted from A4's PyTorch.ipynb
training_loss = []
training_accs = []
validation_accs = []
save_model = False
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print('using device', device)


def train_model(model, optimizer, epochs=15, print_every=100):
    model = model.to(device)
    for e in range(epochs):
        model.train()
        for it, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # keep track of training loss per iter
            training_loss.append(loss.item())

            if it % print_every == 0:
                print('epoch %d, iteration %d, loss = %.4f' % (e, it, loss.item()))

        # after each epoch check and store training and validation accuracies
        print(f'epoch {e} accuracies:')
        print(end='  ')
        check_accuracy(train_loader, model, train=True)
        print(end='  ')
        check_accuracy(val_loader, model, train=False)


def check_accuracy(loader, model, train: bool | None = None):
    split = 'test'
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        if train is not None:
            if train:
                split = 'train'
                training_accs.append(acc)
            else:
                split = 'val'
                validation_accs.append(acc)
        print(f'{split} accuracy: {100 * acc:.2f}% got {num_correct} / {num_samples} correct')


def plot_curves():
    iters = list(range(len(training_loss)))
    epochs = list(range(len(training_accs)))

    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.scatter(iters, training_loss, label='Training Loss', color='dodgerblue')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, training_accs, label='Training Accuracy', color='dodgerblue', marker='o')
    plt.plot(epochs, validation_accs, label='Validation Accuracy', color='darkorange', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy (per epoch)')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # model = MLP(hidden_dims=[2048, 1024, 256, 128])
    # model = ResNet18()
    model = CNN()

    # lr = 0.001 was too small and it was taking forever probably due to dropout
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # X_train, X_test, y_train, y_test = utils.load_mlp_data()
    X_train, X_test, y_train, y_test = utils.load_sampled_cnn_data()
    # X_train, X_test, y_train, y_test = utils.load_full_cnn_data()

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # 80/20 split for training and validation sets
    num_train = int(0.8 * len(train_dataset) // 1)
    size = len(train_dataset)
    print(f'train set: {num_train}, val set: {size - num_train}, test set: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler.SubsetRandomSampler(range(num_train)))
    val_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler.SubsetRandomSampler(range(num_train, size)))
    test_loader = DataLoader(test_dataset, batch_size=64)

    train_model(model, optimizer, epochs=10)
    check_accuracy(test_loader, model)
    plot_curves()

    # save the model at the end if needed
    if save_model:
        torch.save(model.state_dict(), f'models/{type(model).__name__.lower()}.pth')
