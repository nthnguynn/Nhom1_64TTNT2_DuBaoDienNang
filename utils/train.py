def train(model, loader, criterion, optimizer, epochs, device):
    model.to(device)
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.6f}")