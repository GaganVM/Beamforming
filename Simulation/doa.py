import numpy as np
import scipy.linalg as LA
import scipy.signal as ss
from utils import find_position_vector
from weight import array_weight_vector
from tqdm import tqdm
from utils import find_steering_matrix, nsb, capon

def array_response_vector(array,theta):
        N = array.shape
        v = np.exp(1j*2*np.pi*array*np.sin(theta))
        return v/np.sqrt(N)

def music1(CovMat,L,N,array,Angles):
        # CovMat is the signal covariance matrix, L is the number of sources, N is the number of antennas
        # array holds the positions of antenna elements
        # Angles are the grid of directions in the azimuth angular domain
        _,V = LA.eig(CovMat)
        Qn  = V[:,L:N]
        numAngles = Angles.size
        pspectrum = np.zeros(numAngles)
        for i in range(numAngles):
                av = array_response_vector(array,Angles[i])
                pspectrum[i] = 1/LA.norm((Qn.conj().transpose()@av))
        psindB = np.log10(10*pspectrum/pspectrum.min())
        DoAsMUSIC,_= ss.find_peaks(psindB,height=1.35, distance=1.5)
        return DoAsMUSIC,pspectrum

def music(CovMat,L,N,
          angles,
          ris_data, 
          model=None,
          vector_=True, 
          angle_=True,
          height=[7,9], 
          method='dl', 
          P=8, 
          Q=8, 
          lamda=1,
          neural_network=False
          ):
        # CovMat is the signal covariance matrix, L is the number of sources, N is the number of antennas
        # Angles are the grid of directions in the azimuth angular domain
        _,V = LA.eig(CovMat)
        Qn  = V[:,L:N]
        pspectrum = np.zeros(np.array(angles).size)
        methods = ['dl','nsb','capon', 'bartlett']
        if method not in methods:
                raise Exception(f'Invalid weight prediction method: {method}, must be in {methods}.')
        with tqdm(total=len(angles), desc="Processing Angles") as progress_bar:
                if method=='dl':
                        for i,angle in enumerate(angles):
                                specified_vector = find_position_vector(angle[0],angle[1])
                                weights = array_weight_vector(ris_vectors=[ris_data[1]],
                                                                ue_vectors=[specified_vector],
                                                                ris_angles=[ris_data[0]],
                                                                ue_angles=[angle],
                                                                vector=vector_,
                                                                angle=angle_,
                                                                model=model,
                                                                neural_network=neural_network)
                                pspectrum[i] = 1/LA.norm((Qn.conj().transpose()@weights))
                                progress_bar.update(1)
                elif method=='nsb':
                        for i,angle in enumerate(angles):
                                weights = nsb(P,Q, wavelength=lamda, pos_angles=[ris_data[0]]+[angle]).reshape(-1,)
                                pspectrum[i] = 1/LA.norm((Qn.conj().transpose()@weights))
                                progress_bar.update(1)
                elif method=='capon':
                        Rss = np.eye((2))
                        Rnn = 0.1*np.eye((P*Q))
                        for i,angle in enumerate(angles):
                                A = find_steering_matrix(wavelength=lamda, 
                                                        pos_angles=[ris_data[0]]+[angle], 
                                                        planar_antenna_shape=(P,Q))
                                R = A @ Rss @ np.conj(A).T + Rnn
                                g = np.array([1,1])
                                weights = capon(A,R,g)
                                pspectrum[i] = 1/LA.norm((Qn.conj().transpose()@weights))
                                progress_bar.update(1)
                else:
                        M = P*Q # number of antenna elements (sensors)
                        for i,angle in enumerate(angles):
                                A = find_steering_matrix(wavelength=lamda, 
                                                        pos_angles=[ris_data[0]]+[angle], 
                                                        planar_antenna_shape=(P,Q))
                                weights = np.average(A,axis=1)/M
                                pspectrum[i] = 1/LA.norm((Qn.conj().transpose()@weights))
                                progress_bar.update(1)
                        
        progress_bar.close()
        psindB = np.log10(10*pspectrum/(pspectrum.min()+1e-10))
        DoAsMUSIC,_= ss.find_peaks(psindB, height=height, distance=1.5)
        return DoAsMUSIC,pspectrum

def esprit(CovMat,L,N):
        # CovMat is the signal covariance matrix, L is the number of sources, N is the number of antennas
        _,U = LA.eig(CovMat)
        S = U[:,0:L]
        Phi = LA.pinv(S[0:N-1]) @ S[1:N] # the original array is divided into two subarrays [0,1,...,N-2] and [1,2,...,N-1]
        eigs,_ = LA.eig(Phi)
        DoAsESPRIT = np.arcsin(np.angle(eigs)/np.pi)
        return DoAsESPRIT