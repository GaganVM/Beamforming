# main.py
from models import *
from loss import *
from dataloader import create_data_loaders
import torch.optim as optim
import os

train_loader, val_loader, _ = create_data_loaders(input_df, output_df, batch_size=10)

model = Model1()
criterion = rmse_loss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 100
best_val_loss = float('inf')
checkpoint_dir = 'checkpoints'

# Create directory for checkpoints if it does not exist
os.makedirs(checkpoint_dir, exist_ok=True)

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f'Training Epoch [{epoch + 1}/{epochs}], Loss: {average_loss}')

    # Validation
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_targets)
            total_val_loss += val_loss.item()

    average_val_loss = total_val_loss / len(val_loader)
    print(f'Validation Epoch [{epoch + 1}/{epochs}], Loss: {average_val_loss}')

    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        checkpoint_path = os.path.join(checkpoint_dir, f'model_checkpoint_epoch_{epoch + 1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': average_loss,
            'val_loss': average_val_loss,
        }, checkpoint_path)

    model.train()
