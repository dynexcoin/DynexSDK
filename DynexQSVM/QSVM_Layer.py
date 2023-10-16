import os
import numpy as np
import torch
import dimod
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import logging
from itertools import product
import matplotlib
import matplotlib.pyplot as plt
import dynex
import dimod
import neal


logging.basicConfig(filename="QSVM.log", level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class QSVM_Layer(nn.Module):

    """
        This function defines the class of quantum support vector machine.

        :Parameters:
        
        - B, K, C, gamma, xi: SVM model parameters
        - dataset: dataset for train and test
        - train_percent: the percentage of dataset for training 
        - sampler_type: sampler type
                        "DNX" The Dynex Neuromorphic sampler
                        "EXACT" A brute force exact solver which tries all combinations. Very limited problem size
                        "QPU" D-Wave Quantum Processor (QPU) based D-Wave sampler
                        "HQPU" D-Wave Advantage Hybrid Solver
                        "SA" Simulated Annealing using the SimulatedAnnealerSampler from the D-Wave Ocean SDK
        - mainnet: use mainnet or not
        - num_reads: the number of reads for sampler
        - annealing_time: annealing time on the DYNEX platform

        :Example:

        .. code-block:: Pyton

            import math
            import torch
            import torch.nn as nn
            import numpy as np
            import matplotlib.pyplot as plt
            from sklearn.metrics import classification_report
            from torchvision.transforms import ToTensor
            from torch.utils.data import random_split
            from torch.utils.data import Dataset, DataLoader
            from DynexQSVM.QSVM_Layer import QSVM_Layer

            class BankDataset(Dataset):
                def __init__(self, data_file):
                    training_data = np.loadtxt('./datasets/{}'.format(data_file), delimiter=',')
                    for i in range(len(training_data)):
                        if(training_data[i][-1] == 0):
                            training_data[i][-1] = -1
                    data = training_data[:, :2]
                    t = training_data[:, -1]
                    x_min, x_max = 1000, 0
                    y_min, y_max = 1000, 0
                    # rescalling data
                    for i in range(len(training_data)):
                        x_min = min(data[i][0], x_min)
                        x_max = max(data[i][0], x_max)
                        y_min = min(data[i][1], y_min)
                        y_max = max(data[i][1], y_max)
                    for i in range(len(training_data)):
                        data[i][0] = (data[i][0] - x_min)/(x_max - x_min)
                        data[i][1] = (data[i][1] - y_min)/(y_max - y_min)
                    
                    self.data = data
                    self.target = t 

                def __len__(self):
                    return len(self.data)

                def __getitem__(self, idx):
                    
                    d = self.data[idx]
                    t = self.target[idx]
                    return d, t

            def plot_figure(SVM,dataset,train_percent,sampler_type, img):
                plt.figure()
                cm = plt.cm.RdBu
                data = dataset.data
                t = dataset.target
                N = int(len(dataset)*train_percent)
                xx, yy = np.meshgrid(np.linspace(0.0, 1.0, 80), np.linspace(0.0, 1.0, 80))
                Z = []
                for row in range(len(xx)):
                    Z_row = []
                    for col in range(len(xx[row])):
                        target = np.array([xx[row][col], yy[row][col]])
                        Z_row.append(SVM(target))
                    Z.append(Z_row)
                
                cnt = plt.contourf(xx, yy, Z, levels=np.arange(-1, 1.1, 0.1), cmap=cm, alpha=0.8, extend="both")
                plt.contour(xx, yy, Z, levels=[0.0], colors=("black",), linestyles=("--",), linewidths=(0.8,))
                plt.colorbar(cnt, ticks=[-1, 0, 1])

                red_sv = []
                blue_sv = []
                red_pts = []
                blue_pts = []

                for i in range(N):
                    if(SVM.dnxlayer.alpha[i]):
                        if(t[i] == 1):
                            blue_sv.append(data[i, :2])
                        else:
                            red_sv.append(data[i, :2])
                    else:
                        if(t[i] == 1):
                            blue_pts.append(data[i, :2])
                        else:
                            red_pts.append(data[i, :2])

                plt.scatter([el[0] for el in blue_sv],
                            [el[1] for el in blue_sv], color='b', marker='^', edgecolors='k', label="Type 1 SV")

                plt.scatter([el[0] for el in red_sv],
                            [el[1] for el in red_sv], color='r', marker='^', edgecolors='k', label="Type -1 SV")

                plt.scatter([el[0] for el in blue_pts],
                            [el[1] for el in blue_pts], color='b', marker='o', edgecolors='k', label="Type 1 Train")

                plt.scatter([el[0] for el in red_pts],
                            [el[1] for el in red_pts], color='r', marker='o', edgecolors='k', label="Type -1 Train")    
                plt.legend(loc='lower right', fontsize='x-small')
                plt.savefig(f'{img}.jpg')

            # initialize the train, validation, and test data loaders
            bank_dataset = BankDataset(data_file='banknote_1.txt')
            train_percent = 0.8
            train_size = int(len(bank_dataset) * train_percent)
            test_size = len(bank_dataset) - train_size; 
            train_dataset, test_dataset = torch.utils.data.random_split(bank_dataset, [train_size, test_size])

            train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)

            class QSVMModel(nn.Module):
                def __init__(self, B,K,C,gamma,xi,dataset,train_percent,spl,mainnet,num_reads,annealing_time):
                    super().__init__();
                    # Dynex Neuromporphic layer
                    self.dnxlayer = QSVM_Layer(B,K,C,gamma,xi,dataset,train_percent,spl,mainnet,num_reads,annealing_time); 

                def forward(self, x):
                    x = self.dnxlayer(x);
                    return x;

            B = 2;
            K = 2;
            C = 3;
            gamma = 16;
            xi = 0.001;
            spl = "SA";     
            device = "cpu" # no GPU used for Dynex only
            mainnet = True
            num_reads=100 
            annealing_time = 500

            ## load a trained model to predict the test dataset
            model = QSVMModel(B,K,C,gamma,xi,bank_dataset,train_percent,spl,mainnet,num_reads,annealing_time)
            predict(model, './models/QSVM.pth', test_loader)

            ### train a new model on the train dataset
            EPOCHS = 1
            for e in range(0, EPOCHS):
                print("training a new model...")
                print('EPOCH',e+1,'of',EPOCHS);
                tp, fp, tn, fn = 0, 0, 0, 0
                # set the model in training mode
                model.train()
                print("training end")
                # loop over the training set
                for (x, y) in test_loader:
                    # send the input to the device
                    (x, y) = (x.to(device), y.to(device))
                    # perform a forward pass and calculate the training loss
                    pred = model(x);
                    if(y == 1):
                        if(pred > 0):
                            tp += 1
                        else:
                            fp += 1
                    else:
                        if(pred < 0):
                            tn += 1
                        else:
                            fn += 1
                print("test dataset result:")               
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f_score = tp/(tp + 1/2*(fp+fn))
                accuracy = (tp + tn)/(tp+tn+fp+fn)
                print(f"{precision=} {recall=} {f_score=} {accuracy=}")

            plot_figure(model,bank_dataset,train_percent,spl,"img")
        """
    
    def __init__(self, B:int,K:int,C:int,gamma:int,xi:float,dataset,train_percent,sampler_type,mainnet,num_reads,annealing_time):
        
        super(QSVM_Layer, self).__init__()
        self.B = B
        self.K = K 
        self.C = C 
        self.gamma = gamma
        self.xi = xi 
        self.N = int(len(dataset)*train_percent)
        self.sampler_type = sampler_type
        self.data = dataset.data 
        self.t = dataset.target
        self.mainnet = mainnet
        self.num_reads = num_reads
        self.annealing_time = annealing_time

        if(sampler_type == 'HQPU'):
            self.sampler = LeapHybridSampler()
        if(sampler_type == 'SA'):
            self.sampler = neal.SimulatedAnnealingSampler()
        if(sampler_type == 'QPU'):
            self.sampler = EmbeddingComposite(DWaveSampler())
        if(sampler_type == 'DNX'):
            self.sampler = ''
        if(sampler_type == 'EXACT'):
            self.sampler = dimod.ExactSolver()
  
        self.debugging = False
        self.logging = False

        # Log the initialization
        logger.info("Initialized QSVM")
    
    def delta(self, i, j):
        if i == j:
            return 1
        else:
            return 0

    def kernel(self,x, y):
        if self.gamma == -1:
            k = np.dot(x, y)
        elif self.gamma >= 0:
            k = np.exp(-self.gamma*(np.linalg.norm(x-y, ord=2)))
        return k

    def forward(self,x):
        if torch.is_tensor(x):
            x = x.numpy()
        N = len(self.alpha)
        f = sum([self.alpha[n]*self.t[n]*self.kernel(self.data[n], x) for n in range(self.N)]) + self.b
        logger.debug("Completed forward pass")
        return f
    
        
    def train(self,save_model=True, save_path='./models'):
        """
        train the SVM model.

        :Parameters:

            - save_model: save the model's state after training.
            - save_path: the path of the model saved.
        """
        Q_tilde = np.zeros((self.K*self.N, self.K*self.N))
        for n in range(self.N):
            for m in range(self.N):
                for k in range(self.K):
                    for j in range(self.K):
                        Q_tilde[(self.K*n+k, self.K*m+j)] = 0.5*(self.B**(k+j))*self.t[n]*self.t[m]*(self.kernel(self.data[n], self.data[m])+self.xi)-(self.delta(n, m)*self.delta(k, j)*(self.B**k))

        Q = np.zeros((self.K*self.N, self.K*self.N))
        for j in range(self.K*self.N):
            Q[(j, j)] = Q_tilde[(j, j)]
            for i in range(self.K*self.N):
                if i < j:
                    Q[(i, j)] = Q_tilde[(i, j)] + Q_tilde[(j, i)]

        size_of_q = Q.shape[0]
        qubo = {(i, j): Q[i, j] for i, j in product(range(size_of_q), range(size_of_q))}

        if(self.sampler_type == 'HQPU'):
            response = self.sampler.sample_qubo(qubo)
        if(self.sampler_type == 'SA'):
            response = self.sampler.sample_qubo(qubo, num_reads=self.num_reads)
        if(self.sampler_type == 'QPU'):
            response = self.sampler.sample_qubo(qubo, num_reads=self.num_reads)
        if(self.sampler_type == 'EXACT'):
            response = self.sampler.sample_qubo(qubo)
        if(self.sampler_type == 'DNX'):
            response = dynex.sample_qubo(qubo, mainnet=self.mainnet, description='Dynex QSVM (PyTorch)', num_reads=self.num_reads, annealing_time=self.annealing_time)
            
        a = response.first.sample

        self.alpha = []
        for n in range(self.N):
            self.alpha.append(sum([(self.B**k)*a[self.K*n+k] for k in range(self.K)]))

        self.b = sum([self.alpha[n]*(self.C-self.alpha[n])*(self.t[n]-(sum([self.alpha[m]*self.t[m]*self.kernel(self.data[m], self.data[n])
                                                    for m in range(self.N)]))) for n in range(self.N)])/sum([self.alpha[n]*(self.C-self.alpha[n]) for n in range(self.N)])

        
        # Saving the model if specified
        if save_model:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            model_save_path = os.path.join(save_path, f'QSVM.pth')
            self.save_model(model_save_path)
            logger.info(f"Model saved at {model_save_path}")
                
        logger.info("Training completed")
        
        
    def save_model(self, path):
        """
        Save the trained model.

        :Parameters:
        
            - path (str): Path to save the model's state.
        """
        torch.save({'B':self.B,'K':self.K,'C':self.C,'gamma':self.gamma,'xi':self.xi,'alpha': self.alpha, 'b': self.b}, path)
        
    def load_model(self, path):
        """
        Load the model from a saved state.
        
        :Parameters:
        
            - path (str): Path from where to load the model's state.
        """
        checkpoint = torch.load(path)
        self.B = checkpoint['B']
        self.K = checkpoint['K'] 
        self.C = checkpoint['C'] 
        self.gamma = checkpoint['gamma']
        self.xi = checkpoint['xi'] 
        self.alpha = checkpoint['alpha']
        self.b = checkpoint['b']