import copy
import operator
import random
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

# for sampler:
#from pyqubo import Binary
from dimod import BinaryQuadraticModel, SimulatedAnnealingSampler, BINARY
import dynex

def sigmoid(x):
    result = 1 / (1 + np.exp(-x))
    return result

class DYNEX_QRBM:
    def __init__(self,
                 n_visible,
                 n_hidden,
                 num_reads = 1000, 
                 annealing_time = 300, 
                 clones = 1,
                 mainnet=False, 
                 minimum_stepsize = 0.00000006, 
                 logging=False, 
                 debugging=False,
                 rnd = None,
                 seed = None,
                 description = 'Parallel QRBM'):

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.num_reads = num_reads
        self.annealing_time = annealing_time
        self.clones = clones
        self.mainnet = mainnet
        self.minimum_stepsize = minimum_stepsize
        self.logging = logging
        self.debugging = debugging
        if seed != None:
            np.random.seed(seed);
            print('SEED SET TO',seed)
        self.w = (np.random.rand(self.n_visible, self.n_hidden) * 2 - 1) * 1
        self.visible_bias = (np.random.rand(self.n_visible) * 2 - 1) * 1
        self.hidden_bias = (np.random.rand(self.n_hidden) * 2 - 1) * 1
        self.n_epoch = 0
        self.mse = [];
        self.num_features = 0;
        self.num_labels = 0;
        if rnd is None:
            rnd = np.random.RandomState();
        self.rnd = rnd;
        self.description = description;
        
    def sample_opposite_layer_pyqubo(self, v, layer, weights, opposite_layer):
        """
        Generates Qubo model based on two layers and samples them 
        on the Dynex platform
        
        Parameters
        ----------
        v : numpy.ndarray
            The first layer. Shape (num_features+num_labels).
        """
        # initialize Hamiltonian
        #H = 0
        H_vars = []
        bqm = BinaryQuadraticModel.empty(vartype=BINARY)

        # initialize all variables (one for each node in the opposite layer)
        for j in range(len(opposite_layer)):
            #H_vars.append(Binary('DNX'+str(j)))
            H_vars.append('DNX'+str(j))

        for i, bias in enumerate(layer):
            # filter only chosen nodes in the first layer
            if not v[i]:
                continue

            # add reward to every connection
            for j, opp_bias in enumerate(opposite_layer):
                #H += -1 * weights[i][j] * H_vars[j]
                bqm.add_linear(H_vars[j], -1 * weights[i][j])

        for j, opp_bias in enumerate(opposite_layer):
            #H += -1 * opp_bias * H_vars[j]
            bqm.add_linear(H_vars[j],  -1 * opp_bias)

        #model = H.compile()
        #bqm = model.to_bqm()
        
        # use the dynex sampler:
        dnxmodel = dynex.BQM(bqm, logging=self.logging);
        dnxsampler = dynex.DynexSampler(dnxmodel, logging = self.logging, mainnet = self.mainnet, description = self.description);
        sampleset = dnxsampler.sample(num_reads = self.num_reads, 
                                      annealing_time = self.annealing_time, 
                                      clones = self.clones,
                                      debugging = self.debugging, 
                                      minimum_stepsize = self.minimum_stepsize);
        
        solution1 = sampleset.first.sample;
        solution1_list = [(k[3:], v) for k, v in solution1.items()]
        solution1_list.sort(key=lambda tup: int(tup[0]))  # sorts in place
        solution1_list_final = [v for (k, v) in solution1_list]
        return solution1_list_final

    def sample_opposite_layer_pyqubo_batch(self, v_batch, layer, weights, opposite_layer):
        """
        Generates Qubo model based on a batch of layers and samples them 
        on the Dynex platform
        
        Parameters
        ----------
        v : numpy.ndarray
            The first layer. Shape (batch_size, num_features+num_labels).
        """

        # initialize all variables (one for each node per batch in the opposite layer)
        H_vars = [];
        for batch in range(0, len(v_batch)):
            H_vars_item = [];
            for j in range(len(opposite_layer)):
                binstring = 'DNX_batch_'+str(batch)+'_';
                #H_vars_item.append(Binary(binstring+str(j)));
                H_vars_item.append(binstring+str(j));
            H_vars.append(H_vars_item);

        # initialize Hamiltonian
        #H = 0
        bqm = BinaryQuadraticModel.empty(vartype=BINARY)
        for batch in range(0, len(v_batch)):
        #for batch in tqdm(range(len(v_batch))):
            v = v_batch[batch];
            
            for i, bias in enumerate(layer):
                # filter only chosen nodes in the first layer
                if not v[i]:
                    continue
    
                # add reward to every connection
                for j, opp_bias in enumerate(opposite_layer):
                    #H += -1 * weights[i][j] * H_vars[batch][j]
                    bqm.add_linear(H_vars[batch][j], -1 * weights[i][j])
    
            for j, opp_bias in enumerate(opposite_layer):
                #H += -1 * opp_bias * H_vars[batch][j]
                bqm.add_linear(H_vars[batch][j],  -1 * opp_bias)

        #model = H.compile()
        #bqm = model.to_bqm()
        
        # use the dynex sampler:
        dnxmodel = dynex.BQM(bqm, logging=self.logging);
        dnxsampler = dynex.DynexSampler(dnxmodel, logging = self.logging, mainnet = self.mainnet, description = self.description);
        sampleset = dnxsampler.sample(num_reads = self.num_reads, 
                                      annealing_time = self.annealing_time, 
                                      clones = self.clones,
                                      debugging = self.debugging, 
                                      minimum_stepsize = self.minimum_stepsize);

        solution1 = sampleset.first.sample;
        solutions = [];
        for batch in range(0, len(v_batch)):
            filter_string = 'DNX_batch_'+str(batch)+'_';
            solution1_batch = {k:v for (k,v) in solution1.items() if filter_string in k}; # filter vars of current batch
            solution1_list = [(k[len(filter_string):], v) for k, v in solution1_batch.items()] # remove binstring
            solution1_list.sort(key=lambda tup: int(tup[0]))  # sorts in place
            solution1_list_final = [v for (k, v) in solution1_list]
            solutions.append(solution1_list_final);
        return solutions
    
    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def infer(self, visible: npt.NDArray):
        """
        Infer the hidden layer given the visible layer.

        Parameters
        ----------
        visible : numpy.ndarray
            The visible layer. Shape (num_samples, num_visible).

        Returns
        -------
        hidden : numpy.ndarray
            Binary hidden layer values sampled from their probability
            distribution. Shape (num_samples, num_hidden).
        prob_hidden : numpy.ndarray
            The probability for each unit of the hidden layer to be 1. Shape
            (num_samples, num_hidden).
        """
        prob_hidden = self._sigmoid(visible @ self.w + self.hidden_bias)
        hidden = (prob_hidden > self.rnd.random(prob_hidden.shape)).astype(np.int0)
        return hidden, prob_hidden
    
    def generate_visible_layer(self, hidden: npt.NDArray):
        """
        Generate the visible layer given the hidden layer.

        Parameters
        ----------
        hidden : numpy.ndarray
            The hidden layer. Shape (num_samples, num_hidden).

        Returns
        -------
        visible : numpy.ndarray
            Binary visible layer values sampled from their probability
            distribution. Shape (num_samples, num_visible).
        prob_visible : numpy.ndarray
            The probability for each unit of the visible layer to be 1. Shape
            (num_samples, num_visible).
        """
        prob_visible = self._sigmoid(hidden @ self.w.T + self.visible_bias)
        visible = (prob_visible > self.rnd.random(prob_visible.shape)).astype(np.int0)
        return visible, prob_visible
    
    def predict(self, features, num_particles = 100, num_gibbs_updates = 10):
        """
        Predict the labels for the given feature values using CD.

        Parameters
        ----------
        features : numpy.ndarray[np.int0]
            The feature values to predict the labels for. Shape (num_samples,
            num_features).
        num_particles : int
            Number of particles to use for the sampling, i.e. how may times to
            run the label sampling process for each sample.
        num_gibbs_updates : int, optional
            Number of Gibbs updates to perform for each particle. If not
            provided, self.num_gibbs_updates will be used.

        Returns
        -------
        labels : numpy.ndarray[np.float]
            The predicted labels. Shape (num_samples, num_label_classes).
        features: numpy.ndarray[np.float]
            The reconstructed features. Shape (num_samples, num_features).
        """
        num_samples = features.shape[0];
        num_features = features.shape[1];
        num_labels = self.num_labels;
        label_predictions = np.zeros((num_samples, num_labels))
        features_predictions = np.zeros((num_samples, num_features))
        
        for _ in range(num_particles):
            output = np.zeros(label_predictions.shape)
            for _ in range(num_gibbs_updates):
                visible = np.hstack((features, output))
                hidden, _ = self.infer(visible)
                visible, visible_prob = self.generate_visible_layer(hidden)
                output_prob = visible_prob[:,-num_labels:]
                output = visible[:,-num_labels:]
            if num_gibbs_updates > 0:
                label_predictions += output_prob / num_particles
                features_predictions += visible_prob[:,:num_features] / num_particles
        return label_predictions, features_predictions
    
    
    def generate(self, v):
        """Generates data and labels based on features ONLY 
        (no labels to be submitted, these are generated)
        This function invokes two sampling processes on the
        Dynex platform
        """
        
        # add v values for labels, these are not chosen in layer:
        v = np.hstack((v,np.zeros(self.num_features)));
        
        sample_h = self.sample_opposite_layer_pyqubo(v,
                                                     self.visible_bias,
                                                     self.w,
                                                     self.hidden_bias,
                                                     )
        sample_output = self.sample_opposite_layer_pyqubo(sample_h,
                                                          self.hidden_bias,
                                                          self.w.T,
                                                          self.visible_bias,
                                                         )
        gen_data = sample_output[:self.num_features];
        gen_labels = np.array([]);
        if self.num_labels > 0:
            gen_labels = sample_output[self.num_features:]
        
        return gen_data, gen_labels
            
    def get_weights(self):
        return self.w, \
               self.visible_bias, \
               self.hidden_bias

    def set_weights(self, w, visible_bias, hidden_bias):
        self.w = w
        self.visible_bias = visible_bias
        self.hidden_bias = hidden_bias

    def train(self, training_data, training_labels = None, epochs=50, 
              lr=0.1, lr_decay=0.1, epoch_drop = None, momentum = 0, 
              batch_size = None, autosave = False, stop_at_mse = 0.0,
              plot_reconstructed_images = False, mse_every_epochs = 1,
              early_stop_after = 1e99
             ): 
        """
            maximize the product of probabilities assigned to some training set V
            optimize the weight vector

            single-step contrastive divergence (CD-1):
            1. Take a training sample v,
                compute the probabilities of the hidden units and
                sample a hidden activation vector h from this probability distribution.
            2. Compute the outer product of v and h and call this the positive gradient.
            3. From h, sample a reconstruction v' of the visible units,
                then resample the hidden activations h' from this. (Gibbs sampling step)
            4. Compute the outer product of v' and h' and call this the negative gradient.
            5. Let the update to the weight matrix W be the positive gradient minus the negative gradient,
                times some learning rate
            6. Update the biases a and b analogously: a=epsilon (v-v'), b=epsilon (h-h')

            https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine
        """
        
        # labels provided?
        self.num_features = training_data.shape[1];
        if type(training_labels) != type(None):
            self.num_labels = training_labels.shape[1];
        
        if epoch_drop == None:
            epoch_drop = epochs / 5

        # initial momentum velocity value
        momentum_w = np.zeros((len(self.visible_bias), len(self.hidden_bias)))
        momentum_v = np.zeros(len(self.visible_bias))
        momentum_h = np.zeros(len(self.hidden_bias))

        best_mse = 1e10;
        mse = None;
        last_mse = None;
        early_stop_cnt = 0;
        
        pbar = tqdm(range(epochs));
        for epoch in pbar:
            
            # single step
            # 1
            # 1.1 Take a training sample v (if labels provided, add them to v)
            random_selected_training_data_idx = 0, #epoch % len(training_data); # MULTI PROCESSING TAKES ALWAYS THE SAME idx FOR COMPARISON!
            if self.num_labels == 0:
                v = training_data[random_selected_training_data_idx];
            else:
                v = np.hstack((training_data[random_selected_training_data_idx], training_labels[random_selected_training_data_idx]))
            
            old_v = v;

            """
            if batch_size is not None:
                if epoch % batch_size != 0:
                    old_v = v_prim;
                    print('old_v set to v_prim')
            """
            
            # Sample every item in batch:
            for i in range(0, len(training_data)):
                #print('DEBUG:',i,'/',len(training_data));
                if self.num_labels == 0:
                    v = training_data[i];
                else:
                    v = np.hstack((training_data[i], training_labels[i]))
                # sample:
                pbar.set_postfix({'MSE': mse, 'MSE(best)':best_mse,'INFO':"["+str(i+1)+"/"+str(len(training_data))+"][1/3] sampling probabilities of hidden units"});
                h = self.sample_opposite_layer_pyqubo(v, self.visible_bias, self.w, self.hidden_bias);
                pbar.set_postfix({'MSE': mse, 'MSE(best)':best_mse,'INFO':"["+str(i+1)+"/"+str(len(training_data))+"][2/3] sampling reconstruction v' from visible units"});
                v_prim = self.sample_opposite_layer_pyqubo(h, self.hidden_bias, self.w.T, self.visible_bias);
                pbar.set_postfix({'MSE': mse, 'MSE(best)':best_mse,'INFO':"["+str(i+1)+"/"+str(len(training_data))+"][3/3] resampling hidden activations h' from v'"});
                h_prim = self.sample_opposite_layer_pyqubo(v_prim, self.visible_bias, self.w, self.hidden_bias)
                # update weights:
                momentum_w = np.zeros((len(self.visible_bias), len(self.hidden_bias)))
                momentum_v = np.zeros(len(self.visible_bias))
                momentum_h = np.zeros(len(self.hidden_bias))
                pos_grad = np.outer(v, h);
                neg_grad = np.outer(v_prim, h_prim);
                momentum_w += momentum * momentum_w + lr * (pos_grad - neg_grad)
                momentum_v += momentum * momentum_v + lr * (np.array(v) - np.array(v_prim))
                momentum_h += momentum * momentum_h + lr * (np.array(h) - np.array(h_prim))
                self.w += momentum_w; 
                self.visible_bias += momentum_v; 
                self.hidden_bias += momentum_h; 

            """
            # Parallel sampling of one entire batch:
            v_tmp = [];
            for batch in range(0,batch_size):
                v_tmp.append(np.hstack((training_data[batch], training_labels[batch])))
            pbar.set_postfix({'MSE': mse, 'MSE(best)':best_mse,'INFO':"[1/3] sampling probabilities of hidden units"});
            h_tmp = self.sample_opposite_layer_pyqubo_batch(v_tmp, self.visible_bias, self.w, self.hidden_bias)
            pbar.set_postfix({'MSE': mse, 'MSE(best)':best_mse,'INFO':"[2/3] sampling reconstruction v' from visible units"});
            v_prim_tmp = self.sample_opposite_layer_pyqubo_batch(h_tmp, self.hidden_bias, self.w.T, self.visible_bias)
            pbar.set_postfix({'MSE': mse, 'MSE(best)':best_mse,'INFO':"[3/3] resampling hidden activations h' from v'"});
            h_prim_tmp = self.sample_opposite_layer_pyqubo_batch(v_prim_tmp, self.visible_bias, self.w, self.hidden_bias)
            """
            
            """
            # testing parallel processing, but sampling step by step:
            v_tmp = [];
            h_tmp = [];
            v_prim_tmp = [];
            h_prim_tmp = [];
            for batch in tqdm(range(batch_size), leave=False):
                v_ = np.hstack((training_data[batch], training_labels[batch]));
                h_ = self.sample_opposite_layer_pyqubo(v_, self.visible_bias, self.w, self.hidden_bias)
                v_prim_ = self.sample_opposite_layer_pyqubo(h_, self.hidden_bias, self.w.T, self.visible_bias)
                h_prim_ = self.sample_opposite_layer_pyqubo(v_prim_, self.visible_bias, self.w, self.hidden_bias)
                v_tmp.append(v_);
                h_tmp.append(h_);
                v_prim_tmp.append(v_prim_);
                h_prim_tmp.append(h_prim_);
            """

            """
            # weights and biases are updated with the average of the entire batch in one step:
            pbar.set_postfix({'MSE': mse, 'MSE(best)':best_mse,'INFO':"updating weights and biases"});
            momentum_w = np.zeros((len(self.visible_bias), len(self.hidden_bias)))
            momentum_v = np.zeros(len(self.visible_bias))
            momentum_h = np.zeros(len(self.hidden_bias))
            for i in range(0,batch_size):
                pos_grad = np.outer(v_tmp[i], h_tmp[i]);
                neg_grad = np.outer(v_prim_tmp[i], h_prim_tmp[i]);
                momentum_w += momentum * momentum_w + lr * (pos_grad - neg_grad)
                momentum_v += momentum * momentum_v + lr * (np.array(v_tmp[i]) - np.array(v_prim_tmp[i]))
                momentum_h += momentum * momentum_h + lr * (np.array(h_tmp[i]) - np.array(h_prim_tmp[i]))
                
            self.w += momentum_w / batch_size; #average of batch
            self.visible_bias += momentum_v / batch_size; #average of batch
            self.hidden_bias += momentum_h / batch_size; #average of batch
            """
                
            """
            # # 1.2 compute the probabilities of the hidden units
            # persistent CD takes v from previous iterations
            infomsg = 'sampling probabilities of hidden units';
            msg = {'MSE': mse, 'MSE(best)':best_mse,'INFO':infomsg};
            pbar.set_postfix(msg);
            h = self.sample_opposite_layer_pyqubo(old_v, self.visible_bias,
                                                  self.w, self.hidden_bias,
                                                  )
            
            # 2 Compute the outer product of v and h and call this the positive gradient.
            pos_grad = np.outer(v, h)
            
            # 3
            # 3.1 From h, sample a reconstruction v' of the visible units,
            infomsg = "sampling reconstruction v' from visible units";
            msg = {'MSE': mse, 'MSE(best)':best_mse,'INFO':infomsg};
            pbar.set_postfix(msg);
            v_prim = self.sample_opposite_layer_pyqubo(h, self.hidden_bias,
                                                       self.w.T,
                                                       self.visible_bias,
                                                       )

            # 3.2 then resample the hidden activations h' from this. (Gibbs sampling step)
            infomsg = "resampling hidden activations h' from v'";
            msg = {'MSE': mse, 'MSE(best)':best_mse,'INFO':infomsg};
            pbar.set_postfix(msg);
            h_prim = self.sample_opposite_layer_pyqubo(v_prim,
                                                       self.visible_bias,
                                                       self.w, self.hidden_bias,
                                                       )

            # 4 Compute the outer product of v' and h' and call this the negative gradient.
            neg_grad = np.outer(v_prim, h_prim)
            
            # 5 Let the update to the weight matrix W be the positive gradient minus the negative gradient,
            #        times some learning rate
            #this is for momentum (default value 0 doesn't change anything)
            momentum_w = momentum * momentum_w + lr * (pos_grad - neg_grad)
            self.w += momentum_w
            
            # 6 Update the biases a and b analogously: a=epsilon (v-v'), b=epsilon (h-h')
            # momentum here
            momentum_v = momentum * momentum_v + lr * (np.array(v) - np.array(v_prim))
            momentum_h = momentum * momentum_h + lr * (np.array(h) - np.array(h_prim))
            self.visible_bias += momentum_v
            self.hidden_bias += momentum_h
            """
            
            if epoch % epoch_drop == (epoch_drop-1):
                #learning_rate_decay
                lr *= (1 - lr_decay)
                infomsg = "learning rate set to "+str(lr);
                msg = {'MSE': mse, 'MSE(best)':best_mse,'INFO':infomsg};
                pbar.set_postfix(msg);
            
            if epoch % mse_every_epochs == 0:
                # Reconstruct data and calculate MSE:
                infomsg = "reconstructing data to calculate MSE";
                msg = {'MSE': mse, 'MSE(best)':best_mse,'INFO':infomsg};
                pbar.set_postfix(msg);
                gen_data, gen_labels = self.generate(v[:self.num_features]);
                if plot_reconstructed_images:
                    plt.figure()
                    plt.axis('off')
                    plt.title("Image reconstructed after training", y=1.03)
                    plt.imshow(np.array(gen_data).reshape(28, -1))
                # MSE (based on data):
                mse = np.sum((np.array(v[:self.num_features]) - np.array(gen_data))**2 ) / (self.num_features); # normalized
                self.mse.append(mse);
                # better MSE?
                if mse < best_mse:
                    best_mse = mse;
                    if autosave:
                        self.save('model_epoch_'+str(epoch)+'_mse_'+str(mse)+'.model');
                # Check for early stop:
                if mse == last_mse:
                    early_stop_cnt *= 1;
                else:
                    early_stop_cnt = 0;
                last_mse = mse;
                if early_stop_cnt > early_stop_after:
                    infomsg = "TRAINING STOPPED WITH EARLY STOP EPOCHS "+str(early_stop_cnt);
                    msg = {'MSE': mse, 'MSE(best)':best_mse,'INFO':infomsg};
                    pbar.set_postfix(msg);
                    break;
                # update progress bar:
                infomsg="";
                msg = {'MSE': mse, 'MSE(best)':best_mse,'INFO':infomsg};
                pbar.set_postfix(msg);
                # MSE stop?
                if mse <= stop_at_mse:
                    infomsg = "TRAINING STOPPED WITH MSE "+str(stop_at_mse);
                    msg = {'MSE': mse, 'MSE(best)':best_mse,'INFO':infomsg};
                    pbar.set_postfix(msg);
                    break;

        return

    def save(self, filename):
        """Save the model"""
        with np.printoptions(threshold=sys.maxsize):
            parameters = [str(self.n_hidden),
                          str(self.n_visible),
                          np.array_repr(self.visible_bias),
                          np.array_repr(self.hidden_bias),
                          np.array_repr(self.w)]
            with open(filename, 'w') as file:
                file.write('#'.join(parameters))

    def load(self, filename):
        """Load the model"""
        with open(filename) as file:
            res = file.read()
            parameters = res.split('#')
            self.n_hidden = eval(parameters[0])
            self.n_visible = eval(parameters[1])
            self.visible_bias = eval('np.'+parameters[2])
            self.hidden_bias = eval('np.'+parameters[3])
            self.w = eval('np.'+parameters[4])
