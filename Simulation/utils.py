from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
import scipy.linalg as LA
import scipy.signal as ss
import scipy
import math
import numpy as np
import time, datetime
from weight import array_weight_vector

class Time():
    def __init__(self):
        self.begin = 0
        self.final = 0
    def now(self):
        return datetime.datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
    def reset(self):
        self.begin = time.time()
        self.final = time.time()        
    def start(self, message=None):
        if message:
            self.message = message
        self.begin = time.time()
    def end(self):
        self.final = time.time()
        # if self.message:
        #     print('\n>> {}: Done!! Time taken: {:.4f} sec'.format(self.message, float(self.final-self.begin)))
        # else:
        #     print('\n>> Done!! Time taken: {:.4f} sec'.format(float(self.final-self.begin)))
        self.message = None

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
        
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)
    

def visualize_simulation(coordinates, 
                         color, 
                         name, 
                         show_path=True, 
                         precision=1, 
                         anno=True,
                         shift=0,
                         legend_loc=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # locating equipments placed
    bs,ris,ue = coordinates['bs'],coordinates['ris'],coordinates['ue']
    ax.scatter(bs[0], bs[1], bs[2], color=color['bs'], label=name['bs'], marker="s", s=50)
    ax.scatter(ris[0], ris[1], ris[2], color=color['ris'], label=name['ris'], marker='*',s=40)
    ax.scatter(ue[0], ue[1], ue[2], color=color['ue'], label=name['ue'])

    # adding signal flow path
    if show_path:
        for equip in ['ris','ue']:
            loc = coordinates[equip]
            for x,y,z in zip(loc[0],loc[1],loc[2]):
                for xbs,ybs,zbs in zip(bs[0],bs[1],bs[2]):
                    a = Arrow3D([xbs,x],[ybs,y],[zbs,z], 
                                mutation_scale=20, 
                                lw=1, 
                                arrowstyle="<|-", 
                                color=color[equip],
                                alpha=0.5
                               )
                    ax.add_artist(a)

                if equip=='ue':
                    for xris,yris,zris in zip(ris[0],ris[1],ris[2]):
                        b = Arrow3D([xris,x],[yris,y],[zris,z], 
                                    mutation_scale=20, 
                                    lw=1, 
                                    arrowstyle="<|-", 
                                    color=color['ris'],
                                    alpha=0.5
                                   )
                        ax.add_artist(b)
    if anno:
        for loc in [bs,ris,ue]:
            for x,y,z in zip(loc[0],loc[1],loc[2]):
                ax.text(x,y,z+shift,'({},{},{})'.format(round(x,precision),round(y,precision),round(z,precision)))

    ax.set_title("Simulation Setup")
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')

    plt.legend(loc=legend_loc)
    plt.show()
        
def find_position_angles(unit_vector):
    polar_angle = (np.arccos(unit_vector[2])*180)/np.pi
    azimuthal_angle = (np.arctan2(unit_vector[1],unit_vector[0])*180)/np.pi

    return [polar_angle, azimuthal_angle]

def generate_vectors(specified_vector, 
                     num_vectors, 
                     lower_rotation_limit=6, 
                     return_angle=True, 
                     upper_rotation_limit=60):
    result = []
    rotation_angle = math.radians(lower_rotation_limit)
    upper_rotation_angle = math.radians(upper_rotation_limit)
    rotation_angles = np.linspace(rotation_angle, upper_rotation_angle, num=num_vectors)
    for i in range(num_vectors):
        np.random.seed()  # This will set a new random seed based on the system time
#         print(specified_vector)
        random_vector = np.random.rand(3)
        random_vector /= np.linalg.norm(random_vector)

        rotation_axis = np.cross(specified_vector, random_vector)
        rotation_axis /= np.linalg.norm(rotation_axis)

        # Performing the rotation using Rodrigues' rotation formula
        cos_theta = math.cos(rotation_angles[i])
        sin_theta = math.sin(rotation_angles[i])
        rotated_vector = specified_vector * cos_theta + \
                         np.cross(rotation_axis, specified_vector) * sin_theta + \
                         rotation_axis * np.dot(rotation_axis, specified_vector) * (1 - cos_theta)
        rotated_vector /= np.linalg.norm(rotated_vector)
        rotated_angle = find_position_angles(rotated_vector)
        if not return_angle:
            rotated_angle.extend(rotated_vector.tolist())
        result.append(rotated_angle)
            
    return result

def array_response_vector(array,theta):
    N = array.shape
    v = np.exp(1j*2*np.pi*array*np.sin(theta))
    return v/np.sqrt(N)


def esprit(CovMat,L,N):
    # CovMat is the signal covariance matrix, L is the number of sources, N is the number of antennas
    _,U = LA.eig(CovMat)
    S = U[:,0:L]
    Phi = LA.pinv(S[0:N-1]) @ S[1:N] # the original array is divided into two subarrays [0,1,...,N-2] and [1,2,...,N-1]
    eigs,_ = LA.eig(Phi)
    DoAsESPRIT = np.arcsin(np.angle(eigs)/np.pi)
    return DoAsESPRIT

def find_position_vector(polar, azimuth):
    z = np.cos(polar*np.pi/180)
    x = np.sin(polar*np.pi/180)*np.cos(azimuth*np.pi/180)
    y = np.sin(polar*np.pi/180)*np.sin(azimuth*np.pi/180)
    return [x,y,z] #shape: (3,)

def find_steering_vector(wavelength, position, planar_antenna_shape):
    (P,Q) = planar_antenna_shape
    beta = (2*np.pi)/wavelength
    steering_vector = []
    for q in range(Q): # locating anatenas as (y,x)
        for p in range(P):
            r = [p*wavelength/2, q*wavelength/2, 0] #distance between each antena is taken as wavelenght of the signals
            steering_vector.append(np.exp(complex(0,beta*np.dot(r,position))))
    return steering_vector #shape: (PQ,)

def find_steering_matrix(wavelength, pos_angles, planar_antenna_shape):
    steering_matrix = []
    for angle in pos_angles:
        position = find_position_vector(angle[0], angle[1])
        steering_vector = find_steering_vector(wavelength, position, planar_antenna_shape)
        steering_matrix.append(steering_vector)
    steering_matrix = np.array(steering_matrix).T
    return steering_matrix #shape: (PQ, N)

def corelation(complex1, complex2):
    if len(complex1)!=len(complex2):
        raise Exception(f"Mismatch in length of two complex array, i.e. {len(complex1)}!={len(complex2)}")
    abs1 = complex1/np.abs(complex1)
    abs2 = complex2.conj()/np.abs(complex2.conj())
    return np.abs(np.inner(abs1,abs2))/len(complex1)

def array_response_vector(array, polar, azimuthal):
    N = array.shape
    v = np.exp(1j*2*np.pi*array*np.sin(polar))
    return v/np.sqrt(N)

def nsb(P, Q, wavelength, pos_angles):
    e1 = np.zeros(len(pos_angles),)
    e1[0] = 1.0
    e1 = e1[np.newaxis].T # shape: (N,1)

    A = find_steering_matrix(wavelength, pos_angles, (P,Q))
    w = A.dot(np.linalg.inv((np.dot(np.conj(A).T, A)))).dot(e1)
#     w = A.dot(np.linalg.inv((np.dot(A.T, A)))).dot(e1)
    return w

def capon(A, R, g):
    """ 
    Capon's method (MVDR)
    w_Capon = R^(-1) * A * (A^H * R^(-1) * A)^(-1) * g 
    A: steering vector 
    R: coorelation matrix
    """
    w = (np.linalg.inv(R) @ A @ np.linalg.inv( np.conj(A).T @ np.linalg.inv(R) @ A ) @ g).T
    return w

def complex_to_sinusoidal(power_complex, frequency, duration, sampling_rate=1000):
    amplitude = np.abs(power_complex)
    phase = np.angle(power_complex)
    
    time = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    sinusoidal_wave = amplitude * np.cos(2 * np.pi * frequency * time + phase)
    
    return time, sinusoidal_wave

def cross_correlation(signal1, signal2):
#     return np.correlate(signal1, signal2, mode='full')
    return scipy.signal.correlate(signal1, signal2, mode='full')

def find_time_delay(wave1, wave2, sampling_rate):
#     cross_correlation = np.correlate(wave1, wave2, mode='full')
    cross_correlation = scipy.signal.correlate(wave1, wave2, mode='full')
    time_lags = np.arange(-len(wave1) + 1, len(wave1)) / sampling_rate
    time_delay = time_lags[np.argmax(cross_correlation)]
    return time_delay

def power_recieved(ris_power,
                   ue_power,
                   ue_angle,
                   ris_data,
                   model=None,
                   vector=True, 
                   angle=True, 
                   ue_seed=21,
                   ris_seed=42, 
                   neural_network=False
                   ):
    specified_vector = find_position_vector(ue_angle[0],ue_angle[1])
    weights = array_weight_vector(ris_vectors=[ris_data[1]],
                                  ue_vectors=[specified_vector],
                                  ris_angles=[ris_data[0]],
                                  ue_angles=[ue_angle],
                                  vector=vector,
                                  angle=angle,
                                  model=model,
                                  neural_network=neural_network
                                 )
    distortion_ris = np.exp(1j*2*np.pi*np.random.default_rng(seed=ris_seed).random(1)) 
#     distortion_ris = np.exp(1j*2*np.pi*np.random.rand(1)) 
    recieved_power_ris = distortion_ris*ris_power*weights
    distortion_ue = np.exp(1j*2*np.pi*np.random.default_rng(seed=ue_seed).random(1))
#     distortion_ue = np.exp(1j*2*np.pi*np.random.rand(1))
    recieved_power_ue = distortion_ue*ue_power*weights

    net_recieved_power = recieved_power_ris+recieved_power_ue
    return net_recieved_power

def SINR(ris_power,
         ue_power,
         ris_data,
         ue_angles,
         model=None,
         vector=True, 
         angle=True,
         snr=10,
         return_net_power=False,
         n_antenna=64,
         neural_network=False
         ):
    powers,sinr = [],[]
    for i,ue_angle in enumerate(ue_angles.tolist()):
        powers.append(power_recieved(ris_power,
                                     ue_power,
                                     ue_angle,
                                     ris_data,
                                     model=model,
                                     vector=vector,
                                     angle=angle,
                                     ue_seed=i+21,
                                     ris_seed=42,
                                     neural_network=neural_network).sum()
                     ) #/n_antenna)

    noise = np.sqrt(0.5/snr)                                             \
            *((np.random.default_rng(seed=42).random(n_antenna)           \
              +np.random.default_rng(seed=42).random(n_antenna)*1j).sum()\
               /n_antenna)
    try:
        total_power = np.array(powers).sum()
        for i,power in enumerate(powers):
            interference = np.abs(np.average(total_power-power)/np.abs(np.average(total_power-power)))
            denom = np.abs(noise)+interference
            sinr_ = 10*np.log10(np.abs(power)/denom)
            sinr.append(sinr_)
    except ZeroDivisionError:
        print("Error: Division by zero. Make sure interference_power + noise_power is not zero.")
    if return_net_power:
        return total_power,sinr
    else:
        return sinr