import copy
import operator
import random
import matplotlib.pyplot as plt
import numpy as np
import QRBM.sampler as samp

def sigmoid(x):
    result = 1 / (1 + np.exp(-x))
    return result

class classicalRBM:
    def __init__(self,
                 n_visible,
                 n_hidden,
                 err_function='mse',
                 use_tqdm=True,
                 tqdm=None):

        if err_function not in {'mse', 'cosine'}:
            raise ValueError('err_function should be either \'mse\' or \'cosine\'')

        self._use_tqdm = use_tqdm
        self._tqdm = None
        
        if use_tqdm or tqdm is not None:
            from tqdm import tqdm
            self._tqdm = tqdm
            

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        self.w = (np.random.rand(self.n_visible, self.n_hidden) * 2 - 1) * 1
        self.visible_bias = (np.random.rand(self.n_visible) * 2 - 1) * 1
        self.hidden_bias = (np.random.rand(self.n_hidden) * 2 - 1) * 1

        self.n_epoch = 0

    def get_weights(self):
        return self.w, \
               self.visible_bias, \
               self.hidden_bias

    def set_weights(self, w, visible_bias, hidden_bias):
        self.w = w
        self.visible_bias = visible_bias
        self.hidden_bias = hidden_bias

    def get_Z(self):
        Z = np.sum(np.exp(-1 * self.energies))
        self.Z = Z
        return Z

    def train(self, training_data, len_x=1, len_y=1, epochs=1, lr=0.11, lr_decay=0.1, epoch_drop = None):
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

        best_mse = 1e10;

        for epoch in self.tqdm(range(epochs)):
            # single step
            # print("Training data len", len(training_data))

            # 1
            # 1.1 Take a training sample v
            random_selected_training_data_idx = epoch % len(training_data)
            # print("selected_training_data_idx: ", random_selected_training_data_idx)

            v = training_data[random_selected_training_data_idx]
            # print("v: ", v)

            # # 1.2 compute the probabilities of the hidden units
            prob_h = sigmoid(self.hidden_bias + np.dot(v, self.w))

            # print("self.hidden_bias: ", self.hidden_bias)
            # print("np.dot(v, self.w): ", np.dot(v, self.w))
            # print("self.hidden_bias + np.sum(np.dot(v, self.w)): ", self.hidden_bias + np.dot(v, self.w))
            # print("prob_h: ", prob_h)
            h = (np.random.rand(len(self.hidden_bias)) < prob_h).astype(int)
            # print("h: ", h)

            # 2 Compute the outer product of v and h and call this the positive gradient.
            pos_grad = np.outer(v, h)
            # print("pos_grad:", pos_grad)

            # 3
            # 3.1 From h, sample a reconstruction v' of the visible units,
            prob_v_prim = sigmoid(self.visible_bias + np.dot(h, self.w.T))
            v_prim = (np.random.rand(len(self.visible_bias)) < prob_v_prim).astype(int)
            # print("v_prim: ", v_prim)

            # 3.2 then resample the hidden activations h' from this. (Gibbs sampling step)
            prob_h_prim = sigmoid(self.hidden_bias + np.dot(v_prim, self.w))
            h_prim = (np.random.rand(len(self.hidden_bias)) < prob_h_prim).astype(int)
            # print("h_prim: ", h_prim)

            # 4 Compute the outer product of v' and h' and call this the negative gradient.
            neg_grad = np.outer(v_prim, h_prim)
            # print("neg_grad:", neg_grad)

            # 5 Let the update to the weight matrix W be the positive gradient minus the negative gradient,
            #        times some learning rate
            self.w += lr * (pos_grad - neg_grad)
            # print("w: ", self.w)

            # 6 Update the biases a and b analogously: a=epsilon (v-v'), b=epsilon (h-h')
            self.visible_bias += lr * (np.array(v) - np.array(v_prim))
            self.hidden_bias += lr * (np.array(h) - np.array(h_prim))
            # print("visible_bias: ", self.visible_bias)
            # print("hidden_bias: ", self.hidden_bias)
            
            if epoch % epoch_drop == (epoch_drop - 1):
                #learning_rate_decay
                lr *= (1 - lr_decay)
                print('lr=',lr)

            sample_v = v
            prob_sample_h = sigmoid(self.hidden_bias + np.dot(v, self.w))
            sample_h = (np.random.rand(len(self.hidden_bias)) < prob_sample_h).astype(int)
            prob_sample_v_out = sigmoid(self.visible_bias + np.dot(sample_h, self.w.T))
            sample_output = (np.random.rand(len(self.visible_bias)) < prob_sample_v_out).astype(int)
            
            # better MSE?
            mse = np.sum((np.array(v) - np.array(sample_output))**2)/784; # normalized by dividing image size
            learning_curve_plot.append(mse);
            if mse < best_mse:
                best_mse = mse;
                print('Better model found at epoch',epoch,'mse=',mse);
            
        # plot
        x_norm = learning_curve_plot;# (learning_curve_plot-np.min(learning_curve_plot))/(np.max(learning_curve_plot)-np.min(learning_curve_plot))
        plt.figure()
        plt.plot(x_norm)
        plt.xlabel('epoch')
        plt.ylabel('normalised MSE')
        plt.show()
        return

    def generate(self, test_img=None):
        sample_v = []
        if test_img == None:
            sample_v = (np.random.rand(len(self.visible_bias)) < self.visible_bias).astype(int)
            # print("sample_v: ", sample_v)
            # print("visible_bias: ", self.visible_bias)
        else:
            sample_v = test_img
        prob_h = sigmoid(self.hidden_bias + np.dot(sample_v, self.w))
        sample_h = (np.random.rand(len(self.hidden_bias)) < prob_h).astype(int)

        prob_v_out = sigmoid(self.visible_bias + np.dot(sample_h, self.w.T))
        v_out = (np.random.rand(len(self.visible_bias)) < prob_v_out).astype(int)

        return v_out

    def evaluate(self, result, test_img=None):
        # sample_output = self.generate(test_img=test_img)
        min_sum = 1000000
        for pic in self.result_picture_tab:
            new_sum = np.sum((np.array(result) - np.array(pic)) ** 2)
            if new_sum < min_sum:
                min_sum = new_sum

        return min_sum


