import copy
import operator
import random
import sys
import matplotlib.pyplot as plt
import numpy as np
import sampler as samp

def sigmoid(x):
    result = 1 / (1 + np.exp(-x))
    return result

class MSQRBM:
    def __init__(self,
                 n_visible,
                 n_hidden,
                 err_function='mse',
                 qpu=False,
                 chain_strength=5,
                 use_tqdm=True,
                 tqdm=None,
                 result_picture_tab = None):

        if err_function not in {'mse', 'cosine'}:
            raise ValueError('err_function should be either \'mse\' or \'cosine\'')

        self._use_tqdm = use_tqdm
        self._tqdm = None

        if use_tqdm or tqdm is not None:
            from tqdm import tqdm
            self._tqdm = tqdm

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.qpu = qpu
        self.cs = chain_strength

        self.w = (np.random.rand(self.n_visible, self.n_hidden) * 2 - 1) * 1
        self.visible_bias = (np.random.rand(self.n_visible) * 2 - 1) * 1
        self.hidden_bias = (np.random.rand(self.n_hidden) * 2 - 1) * 1
        self.n_epoch = 0
        self.result_picture_tab = result_picture_tab

    def get_weights(self):
        return self.w, \
               self.visible_bias, \
               self.hidden_bias

    def set_weights(self, w, visible_bias, hidden_bias):
        self.w = w
        self.visible_bias = visible_bias
        self.hidden_bias = hidden_bias


    def train(self, training_data, len_x=1, len_y=1, epochs=50, lr=0.1, lr_decay=0.1, epoch_drop = None, momentum = 0, batch_size = None):
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
        learning_curve_plot = []

        if epoch_drop == None:
            epoch_drop = epochs / 5

        # initial momentum velocity value
        momentum_w = np.zeros((len(self.visible_bias), len(self.hidden_bias)))
        momentum_v = np.zeros(len(self.visible_bias))
        momentum_h = np.zeros(len(self.hidden_bias))

        best_mse = 1e10;

        for epoch in self.tqdm(range(epochs)):
            # single step
            # print("Training data len", len(training_data))
            # 1
            # 1.1 Take a training sample v
            random_selected_training_data_idx = epoch % len(training_data)
            # print("selected_training_data_idx: ", random_selected_training_data_idx)

            v = training_data[random_selected_training_data_idx]
            old_v = v

            if batch_size is not None:
                if epoch % batch_size != 0:
                    old_v = v_prim

            # # 1.2 compute the probabilities of the hidden units
            # prob_h = sigmoid(self.hidden_bias + np.dot(v, self.w))
            # print(self.hidden_bias)
            # print(v)
            # print(self.w)

            # print("self.hidden_bias: ", self.hidden_bias)
            # print("np.dot(v, self.w): ", np.dot(v, self.w))
            # print("self.hidden_bias + np.sum(np.dot(v, self.w)): ", self.hidden_bias + np.dot(v, self.w))
            # print("prob_h: ", prob_h)
            # h = (np.random.rand(len(self.hidden_bias)) < prob_h).astype(int)

            # persisntent CD takes v from previous iterations
            h = samp.sample_opposite_layer_pyqubo(old_v, self.visible_bias,
                                                  self.w, self.hidden_bias,
                                                  qpu=self.qpu,
                                                  chain_strength=self.cs)
            # h = samp.sample_opposite_layer_pyqubo(v, self.visible_bias, self.w, self.hidden_bias)

            # print("h: ", h)

            # 2 Compute the outer product of v and h and call this the positive gradient.
            pos_grad = np.outer(v, h)
            # print("pos_grad:", pos_grad)

            # 3
            # 3.1 From h, sample a reconstruction v' of the visible units,
            # prob_v_prim = sigmoid(self.visible_bias + np.dot(h, self.w.T))
            # v_prim = (np.random.rand(len(self.visible_bias)) < prob_v_prim).astype(int)

            v_prim = samp.sample_opposite_layer_pyqubo(h, self.hidden_bias,
                                                       self.w.T,
                                                       self.visible_bias,
                                                       qpu=self.qpu,
                                                       chain_strength=self.cs)

            # print("v_prim: ", v_prim)

            # 3.2 then resample the hidden activations h' from this. (Gibbs sampling step)
            # prob_h_prim = sigmoid(self.hidden_bias + np.dot(v_prim, self.w))
            # h_prim = (np.random.rand(len(self.hidden_bias)) < prob_h_prim).astype(int)

            h_prim = samp.sample_opposite_layer_pyqubo(v_prim,
                                                       self.visible_bias,
                                                       self.w, self.hidden_bias,
                                                       qpu=self.qpu,
                                                       chain_strength=self.cs)
            # print("h_prim: ", h_prim)

            # 4 Compute the outer product of v' and h' and call this the negative gradient.
            neg_grad = np.outer(v_prim, h_prim)
            # print("neg_grad:", neg_grad)

            # 5 Let the update to the weight matrix W be the positive gradient minus the negative gradient,
            #        times some learning rate
            #this is for momentum (default value 0 doesn't change anything)

            momentum_w = momentum * momentum_w + lr * (pos_grad - neg_grad)

            self.w += momentum_w
            # print("w: ", self.w)

            # 6 Update the biases a and b analogously: a=epsilon (v-v'), b=epsilon (h-h')
            #momentum here

            momentum_v = momentum * momentum_v + lr * (np.array(v) - np.array(v_prim))
            momentum_h = momentum * momentum_h + lr * (np.array(h) - np.array(h_prim))

            self.visible_bias += momentum_v
            self.hidden_bias += momentum_h
            # print("visible_bias: ", self.visible_bias)
            # print("hidden_bias: ", self.hidden_bias)

            if epoch % epoch_drop == (epoch_drop-1):
                #learning_rate_decay
                lr *= (1 - lr_decay)
                print("lr = ", lr)

            sample_v = v
            sample_h = samp.sample_opposite_layer_pyqubo(sample_v,
                                                         self.visible_bias,
                                                         self.w,
                                                         self.hidden_bias,
                                                         qpu=self.qpu,
                                                         chain_strength=self.cs)
            sample_output = samp.sample_opposite_layer_pyqubo(sample_h,
                                                              self.hidden_bias,
                                                              self.w.T,
                                                              self.visible_bias,
                                                              qpu=self.qpu,
                                                              chain_strength=self.cs)

            # better MSE?
            mse = np.sum((np.array(v) - np.array(sample_output))**2)/784; # normalized by dividing image size
            learning_curve_plot.append(mse);
            if mse < best_mse:
                best_mse = mse;
                print('Better model found at epoch',epoch,'mse=',mse);
                self.save('model_epoch_'+str(epoch)+'_mse_'+str(mse)+'.model');


        #plot
        x_norm = learning_curve_plot; #(learning_curve_plot-np.min(learning_curve_plot))/(np.max(learning_curve_plot)-np.min(learning_curve_plot))
        plt.figure()
        plt.plot(x_norm)
        plt.xlabel('epoch')
        plt.ylabel('normalised MSE')
        plt.show()
        return

    
    def generate(self, test_img = None):
        sample_v = []
        if test_img == None:
            sample_v = samp.sample_v(self.visible_bias, qpu=self.qpu,
                                     chain_strength=self.cs)
        else:
            sample_v = test_img
        sample_h = samp.sample_opposite_layer_pyqubo(sample_v,
                                                     self.visible_bias, self.w,
                                                     self.hidden_bias,
                                                     qpu=self.qpu,
                                                     chain_strength=self.cs)
        sample_output = samp.sample_opposite_layer_pyqubo(sample_h,
                                                          self.hidden_bias,
                                                          self.w.T,
                                                          self.visible_bias,
                                                          qpu=self.qpu,
                                                          chain_strength=self.cs)
        return sample_output


    def evaluate(self, result, test_img = None):
        # sample_output = self.generate(test_img = test_img)
        min_sum = 1000000
        for pic in self.result_picture_tab:
            new_sum = np.sum((np.array(result) - np.array(pic)) ** 2)
            if new_sum < min_sum:
                min_sum = new_sum

        return min_sum

    def save(self, filename):
        with np.printoptions(threshold=sys.maxsize):
            parameters = [str(self.n_hidden),
                          str(self.n_visible),
                          np.array_repr(self.visible_bias),
                          np.array_repr(self.hidden_bias),
                          np.array_repr(self.w)]
            with open(filename, 'w') as file:
                file.write('#'.join(parameters))


    def load(self, filename):
        with open(filename) as file:
            res = file.read()
            parameters = res.split('#')
            self.n_hidden = eval(parameters[0])
            self.n_visible = eval(parameters[1])
            self.visible_bias = eval('np.'+parameters[2])
            self.hidden_bias = eval('np.'+parameters[3])
            self.w = eval('np.'+parameters[4])
