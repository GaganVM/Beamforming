import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

def create_data_loaders(input_df, output_df, batch_size=10, test_size1=0.1, val_size=0.1, random_state=42):
    input_columns = input_df.iloc[:, [0, 1, 5, 6]]
    output_columns = output_df.iloc[:, :]

    input_tensor = torch.tensor(input_columns.values, dtype=torch.float32)
    output_tensor = torch.tensor(output_columns.values, dtype=torch.float32)

    input_array = input_tensor.numpy()
    output_array = output_tensor.numpy()

    input_train, input_temp, output_train, output_temp = train_test_split(
        input_array, output_array, test_size=(val_size + test_size1), random_state=random_state
    )

    input_val, input_test, output_val, output_test = train_test_split(
        input_temp, output_temp, test_size=test_size1/(val_size + test_size1), random_state=random_state
    )

    input_train = torch.tensor(input_train, dtype=torch.float32)
    input_val = torch.tensor(input_val, dtype=torch.float32)
    input_test = torch.tensor(input_test, dtype=torch.float32)
    output_train = torch.tensor(output_train, dtype=torch.float32)
    output_val = torch.tensor(output_val, dtype=torch.float32)
    output_test = torch.tensor(output_test, dtype=torch.float32)

    train_dataset = TensorDataset(input_train, output_train)
    val_dataset = TensorDataset(input_val, output_val)
    test_dataset = TensorDataset(input_test, output_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
