import numpy as np

def compute_path_diff(ris_pos: list, ue_pos: list):
    ris_pos = np.array([ris_pos]).reshape(3,1)#.dtype(float) # shape: (3,1)
    ue_pos = np.array([ue_pos]).reshape(3,1)#.dtype(float) # shape: (3,1)

    diff = ue_pos-ris_pos
    abs_diff = np.linalg.norm(diff)
    abs_ris_pos = np.linalg.norm(ris_pos)
    abs_ue_pos = np.linalg.norm(ue_pos)
    path_diff = abs_ris_pos + abs_diff - abs_ue_pos

    return path_diff # shape: ()

def path_derivative(ris_pos: list, ue_pos: list):
    ris_pos = np.array([ris_pos]).reshape(3,1)#.dtype(float) # shape: (3,1)
    ue_pos = np.array([ue_pos]).reshape(3,1)#.dtype(float) # shape: (3,1)

    diff = ue_pos-ris_pos
    abs_diff = np.linalg.norm(diff)
    abs_ue_pos = np.linalg.norm(ue_pos)
    path_diff_derivative = np.abs(((ue_pos-ris_pos)/abs_diff) - (ue_pos/abs_ue_pos)) # shape: (3,1)
    
    return path_diff_derivative # shape: (3,1)

def fisher_info(ris_pos:list, ue_pos:list, lr:float, w:float, k:float, c:float):
    """
    Compute fisher information matrix
    Input:
        ris_pos: position of RIS
        ue_pos: position of UE
        lr: localization error
        w: omega
        k: SD of noise
        c: speed of wave
    Output: 
        fisher_matrix: fisher information matrix
    """
    path_diff_derivative = path_derivative(ris_pos, ue_pos) # shape: (3,1)
    path_diff = compute_path_diff(ris_pos, ue_pos)
    fisher_matrix = path_diff_derivative @ path_diff_derivative.T # shape: (3,3)

    coff = ((1/(np.sqrt(2*np.pi)*path_diff))-((lr**2*w**2*path_diff)/(c**2*k**2)))**2
    # fisher_matrix = fisher_matrix*coff # shape: (3,3)
    return fisher_matrix, coff # shape: (3,3), ()

def cramer_limit(ris_pos, ue_pos, lr, w, k, c):
    fisher_matrix, coff = fisher_info(ris_pos, ue_pos, lr, w, k, c) # shape: (3,3)
    cramer_limit = (1/coff)*np.linalg.inv(fisher_matrix) # shape: (3,3)
    limit_x, limit_y, limit_z = cramer_limit[0][0], cramer_limit[1][1], cramer_limit[2][2]
    return limit_x, limit_y, limit_z

# if __name__=='__main__':
    # ris_pos = []
    # ue_pos = []
    # lr = [6.14]
    # w = 
    # k = np.sqrt(2)
    # c = 3*10e8


