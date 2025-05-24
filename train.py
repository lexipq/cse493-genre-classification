import torch
import torch.optim as optim
import torch.nn.functional as F
from model import ResNet18

# boilerplate mainly copy-pasted from A4's PyTorch.ipynb
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print('using device', device)


def train_model(model, optimizer, epochs=1):
    model = model.to(device=device)
    for _ in range(epochs):
        for t, (x, y) in enumerate(train_loader):
            model.train()
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % 100 == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy(train_loader, model)
                check_accuracy(test_loader, model)
                print()


def check_accuracy(loader, model):
    split = 'val' if loader.dataset.train else 'test'
    print('Checking accuracy on the %s set' % split)
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)  # move to device, e.g. GPU
            y = y.to(device=device)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


if __name__ == '__main__':
    model = ResNet18()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    train_model(model, optimizer)
    check_accuracy(test_loader, model)
