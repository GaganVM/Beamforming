import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import pandas as pd
import scipy.signal as ss
import torch
import os
import warnings
import argparse
from tqdm import tqdm
warnings.filterwarnings("ignore")

from weight import array_weight_vector
from utils import *
from doa import music
from models import Model1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

repo_dir = os.getcwd()
tm = Time()

# np.random.seed(6)
study = 'lr_vs_sinr'
P,Q = 8,8 # rows and columns of antenna array
M = P*Q
lamda = 1 # wavelength

def generate_locations(n_antenna, n_source, ue, factor):
    #print('\nsimulation sample data\n----------------------')
    array = np.linspace(0,(n_antenna-1)/2,n_antenna)

    # angle for RIS would be added
    p_thetas = (np.pi/3)*(np.random.rand(n_source))   # random source polar directions
    az_thetas = (2*np.pi)*(np.random.rand(n_source))   # random source azimuthal directions

    # power from RIS would be added
    ue_alphas = np.sqrt(1/2)*(np.random.randn(ue) + np.random.randn(ue)*1j) # random UE powers
    ue_alphas = ue_alphas*factor
    noise_ = -1.59237022-2.38010103j # a random noise
    ris_alphas = ue_alphas*noise_

    #print('random signal direction #polar:',p_thetas*180/np.pi)
    #print('random signal direction #azimuthal:',az_thetas*180/np.pi)

    ue_vectors, ris_vectors = [],[]
    ue_angles, ris_angles = [], []
    for polar,az in zip(p_thetas[:ue],az_thetas[:ue]):
        ue_vectors.append(find_position_vector(polar,az))
        ue_angles.append([polar*180/np.pi,az*180/np.pi])

    for polar,az in zip(p_thetas[ue:],az_thetas[ue:]):
        ris_vectors.append(find_position_vector(polar,az))
        ris_angles.append([polar*180/np.pi,az*180/np.pi])

    bs_vector = [0,0,0]
    #print('\nbase station locations:',bs_vector)
    #print(f'\nRIS \ndirection vectors: {ris_vectors}\ndirection angles: {ris_angles}\npower: {ris_alphas}')
    #print(f'\nuser equipment \ndirection vectors: {ue_vectors}\ndirection angles: {ue_angles}\npower: {ue_alphas}')
    return {
        'ris': [ris_vectors, ris_angles, ris_alphas],
        'ue': [ue_vectors, ue_angles, ue_alphas]
    }

def get_covariance_matrix(n_antenna, numrealization, snr, ris_vectors, ris_angles, ris_alphas, ue_vectors, ue_angles, ue_alphas, model=None):
    if method=='dl':
        tm.start(message='preparing covariance matrix by DL')
        H = np.zeros((n_antenna,numrealization)) + 1j*np.zeros((n_antenna,numrealization))
        #print('H matrix shape(n_antenna,time_instance):',H.shape)
        antenna_weight = array_weight_vector(ris_vectors,
                                            ue_vectors,
                                            ris_angles,
                                            ue_angles,
                                            vector=True,
                                            angle=True,)
    elif method=='nn':
        tm.start(message='preparing covariance matrix by NN')
        H = np.zeros((n_antenna,numrealization)) + 1j*np.zeros((n_antenna,numrealization))
        #print('H matrix shape(n_antenna,time_instance):',H.shape)
        antenna_weight = array_weight_vector(ris_vectors,
                                     ue_vectors,
                                     ris_angles,
                                     ue_angles,
                                     vector=False,
                                     angle=True,
                                     model=model,
                                     neural_network=True) 
    elif method=='nsb':
        tm.start(message='preparing covariance matrix by NSB')
        H = np.zeros((n_antenna,numrealization)) + 1j*np.zeros((n_antenna,numrealization))
        #print('H matrix shape(n_antenna,time_instance):',H.shape)
        antenna_weight = nsb(P,Q, wavelength=lamda, pos_angles=ris_angles+ue_angles).reshape(-1,)
        
    elif method=='capon':
        M = P*Q # number of antenna elements (sensors)
        pos_angles = ris_angles+ue_angles
        Rss = np.eye((2))# correlation matrix of the information symbols
        Rnn = 0.1*np.eye((M))# correlation matrix of additive noise
        A = find_steering_matrix(wavelength=lamda, pos_angles=pos_angles, planar_antenna_shape=(P,Q)) # steering vectors
        R = A @ Rss @ np.conj(A).T + Rnn # correlation matrix
        g = np.array([1,1]) # gate array: both DoAs are "switched on"        
        tm.start(message='preparing covariance matrix by Capon')
        H = np.zeros((n_antenna,numrealization)) + 1j*np.zeros((n_antenna,numrealization))
        #print('H matrix shape(n_antenna,time_instance):',H.shape)
        antenna_weight = capon(A,R,g)

    elif method=='bartlett':
        M = P*Q # number of antenna elements (sensors)
        pos_angles = ris_angles+ue_angles
        A = find_steering_matrix(wavelength=lamda, pos_angles=pos_angles, planar_antenna_shape=(P,Q)) # steering vectors
        tm.start(message='preparing covariance matrix by Bartlett')
        H = np.zeros((n_antenna,numrealization)) + 1j*np.zeros((n_antenna,numrealization))
        #print('H matrix shape(n_antenna,time_instance):',H.shape)
        antenna_weight = np.average(A,axis=1)/M

    for iter in range(numrealization):
        # random distortions due to propagation medium
        #ris
        distortion_ris = np.exp(1j*2*np.pi*np.random.rand(1)) 
        recieved_power_ris = distortion_ris*ris_alphas*antenna_weight
        #ue
        distortion_ue = np.exp(1j*2*np.pi*np.random.rand(1))
        recieved_power_ue = distortion_ue*ue_alphas*antenna_weight
        
        net_recieved_power = recieved_power_ris+recieved_power_ue
        noise = np.sqrt(0.5/snr)*(np.random.randn(n_antenna)+np.random.randn(n_antenna)*1j)
        H[:,iter] = net_recieved_power+noise
    CovMat = H@H.conj().transpose()
    return CovMat

def get_doa(method, CovMat, n_source, n_antenna, general_angles, ris_angles, ris_vectors, model=None):
    dir = os.path.join(repo_dir,'data',study)
    os.makedirs(name=dir, exist_ok=True)
    # MUSIC algorithm
    path = os.path.join(dir,f'{method}:psindB.pkl')
    if not os.path.exists(path):
        if method=='nn':
            # MUSIC algorithm
            DoAsMUSIC, psindB = music(CovMat,
                                    L = n_source,
                                    N = n_antenna,
                                    angles = general_angles,
                                    ris_data = ris_angles+ris_vectors,
                                    model = model,
                                    vector_ = False,
                                    angle_ = True,
                                    neural_network=True)
        elif method in ['nsb', 'capon', 'bartlett', 'dl']:
            DoAsMUSIC, psindB = music(CovMat,
                                    L = n_source,
                                    N = n_antenna,
                                    angles = general_angles,
                                    ris_data = ris_angles+ris_vectors,
                                    height=None,
                                    method=method)
        with open(path, 'wb') as file:
            pkl.dump(psindB, file)
    else:   
        with open(path, 'rb') as file:
            psindB = pkl.load(file)
    return psindB

def compute_lr(samples, sinr, frequency, speed, n_antenna, pos_data, model=None, ris_angles=None, ris_vectors=None, noiseSD=1):
    ue_angle = samples[np.argmax(sinr)].tolist()
    specified_vector = find_position_vector(ue_angle[0],ue_angle[1])
    ris_data=pos_data['ris'][1]+pos_data['ris'][0]
    if method=='dl':
        weights = array_weight_vector(ris_vectors=[ris_data[1]],
                                    ue_vectors=[specified_vector],
                                    ris_angles=[ris_data[0]],
                                    ue_angles=[ue_angle],
                                    vector=True,
                                    angle=True
                                    )
    elif method=='nn':
        weights = array_weight_vector(ris_vectors=[ris_data[1]],
                              ue_vectors=[specified_vector],
                              ris_angles=[ris_data[0]],
                              ue_angles=[ue_angle],
                              vector=False,
                              angle=True,
                              model=model,
                              neural_network=True)     
    elif method=='nsb':
        ue_angle = samples[np.argmax(sinr)].tolist()
        ris_data = ris_angles+ris_vectors
        weights = nsb(P,Q, wavelength=lamda, pos_angles=[ris_data[0]]+[ue_angle]).reshape(-1,)    
    elif method=='capon':
        # frequency = 3e8  
        # speed= frequency*lamda
        Rss = np.eye((2))# correlation matrix of the information symbols
        Rnn = 0.1*np.eye((M))# correlation matrix of additive noise
        ue_angle = samples[np.argmax(sinr)].tolist()
        ris_data=ris_angles+ris_vectors
        pos_angles = [ris_data[0]]+[ue_angle]
        A = find_steering_matrix(wavelength=lamda, pos_angles=pos_angles, planar_antenna_shape=(P,Q)) # steering vectors
        R = A @ Rss @ np.conj(A).T + Rnn # correlation matrix
        g = np.array([1,1]) # gate array: both DoAs are "switched on"
        weights = capon(A,R,g)    
    elif method=='bartlett':
        ue_angle = samples[np.argmax(sinr)].tolist()
        ris_data=ris_angles+ris_vectors

        pos_angles = [ris_data[0]]+[ue_angle]
        A = find_steering_matrix(wavelength=lamda, pos_angles=pos_angles, planar_antenna_shape=(P,Q)) # steering vectors
        weights = np.average(A,axis=1)/M
    #ris
    # print(np.random.normal(loc=0.0, scale=noiseSD, size=1))
    # distortion_ris = np.exp(1j*2*np.pi*np.random.default_rng(seed=42).random(1)) 
    distortion_ris = np.exp(1j*np.random.normal(loc=0.0, scale=noiseSD, size=1)) # gussian noise to phase
    recieved_power_ris = np.sum(distortion_ris*pos_data['ris'][2]*weights)/n_antenna
    # ris_time, ris_sinusoidal_wave = complex_to_sinusoidal(recieved_power_ris, 
    #                                                     frequency, 
    #                                                     duration,
    #                                                     sampling_rate=sampling_rate)

    #ue
    # distortion_ue = np.exp(1j*2*np.pi*np.random.default_rng(seed=21+np.argmax(sinr)).random(1))
    distortion_ue = np.exp(1j*np.random.normal(loc=0.0, scale=noiseSD, size=1))  # gussian noise to phase
    recieved_power_ue = np.sum(distortion_ue*pos_data['ue'][2]*weights)/n_antenna
    # ue_time, ue_sinusoidal_wave = complex_to_sinusoidal(recieved_power_ue, 
    #                                                     frequency, 
    #                                                     duration,
    #                                                     sampling_rate=sampling_rate)

    #print('RIS\n---')
    #print('amplitude:',np.abs(recieved_power_ris))
    phase_ris = np.angle(recieved_power_ris)*180/np.pi
    #print('phase:',phase_ris)

    #print('\nUser Equipement\n---------------')
    #print('amplitude:',np.abs(recieved_power_ue))
    phase_ue = np.angle(recieved_power_ue)*180/np.pi
    #print('phase:',phase_ue)

    phase_diff = abs(phase_ris-phase_ue)
    omega = np.deg2rad(360)*frequency
    del_t = phase_diff/omega
    del_d = speed*del_t

    #print(f'\ntime delay: {abs(del_t)} seconds')
    #print(f'∇d: {del_d} meters')
    return del_d, del_t

def simulation(factor, method, noiseSD):
    P,Q = 8,8 # rows and columns of antenna array
    lamda = 1 # wavelength
    ue = 1  # number of user equipments
    ris = 1  # number of RIS
    n_source = ue+ris
    n_antenna = P*Q  # number of antenna elements 
    snr = 10
    frequency = 3e8
    # duration = 5
    # sampling_rate=50 
    speed= frequency*lamda
    numAngles = 360
    numrealization = 100 # number of time samples collected at antenna array
    signal_samples = 10
    model = Model1(input_size=4) if method=='nn' else None

    pos_data = generate_locations(n_antenna, n_source, ue, factor)
    ris_vectors, ris_angles, ris_alphas = pos_data['ris'][0], pos_data['ris'][1], pos_data['ris'][2]
    ue_vectors, ue_angles, ue_alphas = pos_data['ue'][0], pos_data['ue'][1], pos_data['ue'][2]

    p_angles = np.linspace(0,np.pi/3,numAngles)*180/np.pi
    az_angles = np.linspace(0,2*np.pi,numAngles)*180/np.pi
    general_angles = np.array(np.meshgrid(p_angles,az_angles)).T.reshape(-1, 2).tolist()
    #print('Total sample location collected:', len(general_angles))

    sim_tm = Time()
    sim_tm.start(message='>> simulation time')
    CovMat = get_covariance_matrix(n_antenna, numrealization, snr, 
                                   pos_data['ris'][0], pos_data['ris'][1], pos_data['ris'][2], 
                                   pos_data['ue'][0], pos_data['ue'][1], pos_data['ue'][2], model=model)
    #print('covariance matrix shape:',CovMat.shape)
    tm.end()

    psindB = get_doa(method, CovMat, n_source, n_antenna, general_angles, pos_data['ris'][1], pos_data['ris'][0], model=model)

    DoAsMUSIC,_= ss.find_peaks(psindB)
    indies = np.argsort(psindB[DoAsMUSIC])[-1*signal_samples:]
    samples = np.array(general_angles)[DoAsMUSIC][indies]

    if method=='dl':
        sinr = SINR(ris_power = ris_alphas,
                    ue_power = ue_alphas,
                    ue_angles = samples,
                    ris_data = ris_angles+ris_vectors,
                    snr = 100)
    elif method=='nn':
        # model = Model1(input_size=4)
        input_ = 'azimuthal'
        model_name = 'model1'
        weight_file = os.listdir(os.path.join(repo_dir,'model_states',input_, model_name))[0]
        state = torch.load(os.path.join(repo_dir,'model_states',input_,model_name,weight_file), 
                        map_location=torch.device(device))

        model.load_state_dict(state)
        #print(model.eval())
        sinr = SINR(ris_power = ris_alphas,
                    ue_power = ue_alphas,
                    ue_angles = samples,
                    ris_data = ris_angles+ris_vectors,
                    model = model,
                    vector = False,
                    angle = True,
                    snr = 10,
                    neural_network=True)
    elif method in ['nsb', 'capon', 'bartlett']:
        sinr = SINR(ris_power = ris_alphas,
                    ue_power = ue_alphas,
                    ue_angles = samples,
                    ris_data = ris_angles+ris_vectors,
                    snr = 10)    

    detected_ue_angles = samples[np.argmax(sinr)].tolist()
    #print('detected angles for user equipment:',detected_ue_angles)
    #print('actual UE direction:',ue_angles)
    #print('with maximum SINR:',np.max(sinr))
    #print()

    del_d, del_t = compute_lr(samples, sinr, frequency, speed, n_antenna, pos_data, model, ris_angles, ris_vectors, noiseSD)
    sim_tm.end()

    return del_d, sinr

def parse_args():
    parser = argparse.ArgumentParser(description="A simple example parser")

    # Positional argument for the filename
    parser.add_argument('method', type=str, help='select the method from dl, nsb', default='dl')
    return parser.parse_args()

if __name__=='__main__':
    factors = [0.01, 0.1, 1, 2, 5, 10, 20, 50, 75, 100, 500, 1000]
    # args = parse_args()
    # method = 'bartlett'
    # data = []
    factor = 1
    # for factor in factors:
    #     del_d, sinr = simulation(factor, method)
    #     data.append([factor, del_d, np.max(sinr)])
    # df = pd.DataFrame(data=data, columns=['factor', 'lr', 'sinr'])
    # df.to_csv(os.path.join(os.getcwd(), method+'.csv'), index=False)
    # #print(df)
    stds = list(range(1,10)) #[1, 3, 5, 8, 10]

    methods = ['nn', 'dl', 'nsb', 'capon', 'bartlett']
    # std_df = pd.DataFrame(stds, columns=['noise_std'])
    # sinr_df = pd.DataFrame(stds, columns=['noise_std'])
    n_exprt = 1000
    avgSTD = np.zeros((len(methods), len(stds)))
    avgSINR = np.zeros((len(methods), len(stds)))
    for i in tqdm(list(range(n_exprt)), desc='experiment'):
        std_df = [] #np.array([]) #pd.DataFrame()
        sinr_df = [] #np.array([]) #pd.DataFrame()
        for method in methods:
            stdData, sinrData = [], []
            for std in stds:
                del_d, sinr = simulation(factor, method, noiseSD=std)
                stdData.append(del_d)
                sinrData.append(np.max(sinr))
            # std_df = np.append(std_df, np.array(stdData))
            # sinr_df = np.append(sinr_df, np.array(sinrData))
            std_df.append(stdData)
            sinr_df.append(sinrData)
            # std_df[method] = stdData
            # sinr_df[method] = sinrData
        std_df = np.array(std_df)
        sinr_df = np.array(sinr_df)
        avgSTD += std_df/n_exprt
        avgSINR += sinr_df/n_exprt

    std_df = pd.DataFrame(avgSTD, columns=stds)
    sinr_df = pd.DataFrame(avgSINR, columns=stds)
    std_df['method'] = methods
    sinr_df['method'] = methods
    # df = pd.DataFrame(data=data, columns=['std', 'lr', 'sinr'])
    std_df.to_csv(os.path.join(os.getcwd(), 'std'+'.csv'), index=False)
    sinr_df.to_csv(os.path.join(os.getcwd(), 'sinr'+'.csv'), index=False)
