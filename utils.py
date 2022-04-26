import torch

def accuracy(model, testdl, device):
    model.eval()
    
    correct, total = 0, 0
    for images, labels in testdl:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        _, preds = torch.max(output, 1)
        correct += (preds == labels).sum().item()
        total += len(labels)
    accuracy_percent = correct / total

    return accuracy_percent


def evaluate(model, criterion, valid_dl, device):

    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for images, labels in valid_dl:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            epoch_loss += loss.item()

    return epoch_loss

