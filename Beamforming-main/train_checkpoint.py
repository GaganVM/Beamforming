from models import *
from loss import *
from dataloader_gpu import create_data_loaders
import torch
import torch.optim as optim
import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = Model2().to(device)
criterion = RMSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# loading checkpoint
checkpoint=torch.load('C:\\Users\\admin\\Desktop\\iot\\Beamforming-main\\autoencoder_checkpoints\\azimuthal\\model2\\model_checkpoint_epoch_100.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

start_epoch=100
print("Loading data..")
train_loader, val_loader, _ = create_data_loaders('C:\\Users\\admin\\Desktop\\iot\\input_signal_data.csv','C:\\Users\\admin\\Desktop\\iot\\output_weight_parameters_data.csv', batch_size=1024, device=device)

epochs = 200
best_val_loss = np.inf
checkpoint_dir = 'C:\\Users\\admin\\Desktop\\iot\\Beamforming-main\\autoencoder_checkpoints\\position\\model23'
weights_dir = 'C:\\Users\\admin\\Desktop\\iot\\Beamforming-main\\autoencoder_weights\\position\\model23'  

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)

print("Start training..")
start = time.time()
for epoch in range(start_epoch,epochs):
    model.train()
    total_loss = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
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
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_targets)
            total_val_loss += val_loss.item()

    average_val_loss = total_val_loss / len(val_loader)
    print(f'Validation Epoch [{epoch + 1}/{epochs}], Loss: {average_val_loss}')
    checkpoint_path = os.path.join(checkpoint_dir, f'model_checkpoint_epoch_{epoch + 1}.pt')
    torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': average_loss,
            'val_loss': average_val_loss,
        }, checkpoint_path)
    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        best_epoch = epoch + 1
        best_model_weights = model.state_dict()
end = time.time()
print(str((end-start)//60) + " minutes")
weights_path = os.path.join(weights_dir, f'model_weights_best_{best_epoch}.pt')  
torch.save(best_model_weights, weights_path)
