import torch
import torch.nn.functional as F

def train_network(model, train_data_loader, loss_fn, optimizer, hr_predict=None, validation_set = None, transform=None):
    size = len(train_data_loader.dataset)
    l = []
    c = []
    v = []
    if validation_set:
        v_X, v_y = validation_set
        v_y = hr_predict(v_y)
    for batch, (X, y) in enumerate(train_data_loader):
        model.train()
        if hr_predict is not None:
            y = hr_predict(y)

        if transform:
            X = transform(X)

        yhat = model(X)
        loss = loss_fn(yhat, y)


        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1 == 0:
            model.eval()
            if validation_set:
                v.append(loss_fn(model(v_X), v_y).item())
            loss, current = loss.item(), (batch + 1) * len(X)
            l.append(loss)
            c.append(batch)
            print(f"loss: {loss:>7f} (bpm: {loss*60})  [{current:>5d}/{size:>5d}]")
    return c, l, v