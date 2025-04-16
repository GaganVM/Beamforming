import os
import numpy as np
import torch
from models import LSTMModel

# model prediction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm = LSTMModel(input_size=5,
                 hidden_size=576,
                 num_layers=4,
                 output_size=128).to(device=device)

state = torch.load(os.path.join(os.getcwd(),'model_states','model_weights_best_198.pt'), 
                   map_location=torch.device(device))
lstm.load_state_dict(state)

def complex_weight(arr):
    n = len(arr)    
    imag = arr[:n//2]
    real = arr[n//2:]
    return np.vectorize(complex)(real, imag)

# antenna weight prediction
def array_weight_vector(ris_vectors=[],
                        ue_vectors=[],
                        ris_angles=[],
                        ue_angles=[],
                        vector=True,
                        angle=True,
                        model= None,
                        neural_network=False
                        ):
        if vector and angle:
                input_array = ris_angles[0]+ris_vectors[0]+ue_angles[0]+ue_vectors[0]
        elif not vector and angle:
                input_array = ris_angles[0]+ue_angles[0]
        elif vector and not angle:
                input_array = ris_vectors[0]+ue_vectors[0]
        else:
                raise Exception("Neither angles nor vectors are selected for weight prediction!!")
        if not neural_network:
               input_array = np.array(input_array).reshape(1, 2, len(input_array)//2)
        input_tensor = torch.tensor(input_array, dtype=torch.float32, device=device)
        try:
            with torch.no_grad():
                    if not model:
                            # default made with the best performing model, but can be changed
                            output = lstm(input_tensor)                
                    else:
                            output = model(input_tensor)
            if neural_network:
                weights = complex_weight(output.cpu().numpy())
            else:
                weights = complex_weight(output.cpu().numpy()[0])
            
            return weights
        except:
                raise Exception('mismatch of shape of input and model input shape')