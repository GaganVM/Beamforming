import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def create_data_loaders(input_df, 
                        output_df, 
                        batch_size=10, 
                        test_size1=0.1, 
                        val_size=0.1, 
                        random_state=42, 
                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        ):
    input_df=pd.read_csv(input_df) 
    output_df=pd.read_csv(output_df)   
    # input_columns = input_df.iloc[:,[2,3,4,7,8,9]]
    input_columns = input_df.iloc[:,:]

    output_columns = output_df.iloc[:, :]

    input_array=input_columns.values
    output_array=output_columns.values
    
    scaler_input = MinMaxScaler()
    scaler_output = MinMaxScaler()
    
    input_array=scaler_input.fit_transform(input_array)
    output_array=scaler_output.fit_transform(output_array)
    
    print("Device for Dataloder:",device)
    
    input_tensor = torch.tensor(input_array, dtype=torch.float32, device=device)
    output_tensor = torch.tensor(output_array, dtype=torch.float32, device=device)

    input_array = input_tensor.cpu().numpy() 
    output_array = output_tensor.cpu().numpy()

    input_train, input_temp, output_train, output_temp = train_test_split(
        input_array, output_array, test_size=(val_size + test_size1), random_state=random_state
    )

    input_val, input_test, output_val, output_test = train_test_split(
        input_temp, output_temp, test_size=test_size1/(val_size + test_size1), random_state=random_state
    )

    input_train = torch.tensor(input_train, dtype=torch.float32, device=device)
    input_val = torch.tensor(input_val, dtype=torch.float32, device=device)
    input_test = torch.tensor(input_test, dtype=torch.float32, device=device)
    output_train = torch.tensor(output_train, dtype=torch.float32, device=device)
    output_val = torch.tensor(output_val, dtype=torch.float32, device=device)
    output_test = torch.tensor(output_test, dtype=torch.float32, device=device)

    train_dataset = TensorDataset(input_train, output_train)
    val_dataset = TensorDataset(input_val, output_val)
    test_dataset = TensorDataset(input_test, output_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader
