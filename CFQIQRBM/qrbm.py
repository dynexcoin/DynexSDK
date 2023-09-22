import numpy as np
import copy
import operator
import matplotlib.pyplot as plt

from CFQIRBM.sampler import Sampler


class QRBM:
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

        self.w = (np.random.rand(self.n_visible, self.n_hidden) * 2 - 1) * 0.01
        self.visible_bias = (np.random.rand(self.n_visible) * 2 - 1) * 0.01
        self.hidden_bias = (np.random.rand(self.n_hidden) * 2 - 1) * 0.01

        self.sampler = Sampler()

        self.n_epoch = 0

    def get_weights(self):
        return self.w, \
               self.visible_bias, \
               self.hidden_bias

    def set_weights(self, w, visible_bias, hidden_bias):
        self.w = w
        self.visible_bias = visible_bias
        self.hidden_bias = hidden_bias

    def set_qubo(self):

        visible_bias = self.visible_bias
        hidden_bias = self.hidden_bias
        w = self.w

        Q = {}
        for i in range(self.n_visible):
            Q[(i, i)] = -1 * visible_bias[i]
        for i in range(self.n_hidden):
            Q[(i + self.n_visible, i + self.n_visible)] = -1 * hidden_bias[i]
        for i in range(self.n_visible):
            for j in range(self.n_hidden):
                Q[(i, self.n_visible + j)] = -1 * w[i][j]
        self.Q = Q

    def sample_clamped_qubo(self, position_, value_, num_samps=100):
        """
            Force samples to have specific values in positions
        """
        assert(len(position_) == len(value_))

        if not hasattr(self, 'Q'):
            self.set_qubo()


        n_to_clamp = len(position_)
        position = copy.deepcopy(position_)
        value = copy.deepcopy(value_)
        Q = copy.deepcopy(self.Q)

        clamp_strength = 50

        for to_clamp in range(n_to_clamp):
            this_idx = position[to_clamp]
            this_value = value[to_clamp]
            if this_value == 0:  # x=0
                # Update bias to force sample[this_idx] to be 0
                Q[(this_idx, this_idx)] = clamp_strength
                # Update w given sample[this_idx] = 0
                for (x, y) in Q:
                    if (x == this_idx or y == this_idx) and x != y:
                        Q[(x, y)] *= 0
            else:  # x == 1:
                # Update bias to force sample[this_idx] to be 1
                Q[(this_idx, this_idx)] = -1 * clamp_strength
                # Update w given sample[this_idx] = 1
                for (x, y) in Q:
                    if (x == this_idx or y == this_idx) and x != y:
                        pass  # no need to change

        self.samples, self.energies, self.num_occurrences = self.sampler.sample_qubo(Q, num_samps = num_samps)
        self.energies /= np.max(np.abs(self.energies))

        return self.samples

    def get_Z(self):
        Z = np.sum(np.exp(-1 * self.energies))
        self.Z = Z
        return Z

    def sample_qubo(self, num_samps=100):
        if not hasattr(self, 'Q'):
            self.set_qubo()
        self.samples, self.energies, self.num_occurrences = self.sampler.sample_qubo(self.Q, num_samps=num_samps)
        self.energies /= np.max(np.abs(self.energies))
        self.get_Z()
        return self.samples

    def prediction_sample_to_probability_dict(self, ranges_to_predict, sample_idx_to_clamp, sample_value_to_clamp):
        self.get_Z()
        predictions_dicts = []
        #print('DEBUG: self.samples',self.samples);
        for range_to_predict_start, range_to_predict_end in ranges_to_predict:
            if len(self.samples) == 0:
                predictions_dicts.append(None)
                continue

            predictions_dict = {}

            sample_idx = 0
            for sample, energy in zip(self.samples, self.energies):

                #print('sample_idx_to_clamp:',sample_idx_to_clamp);
                #print('sample:',sample);
                values_in_sample_which_should_be_clamped = sample[sample_idx_to_clamp]

                if type(values_in_sample_which_should_be_clamped) != list:
                    values_in_sample_which_should_be_clamped = values_in_sample_which_should_be_clamped.tolist()
                if type(sample_value_to_clamp) != list:
                    sample_value_to_clamp = sample_value_to_clamp.tolist()
                if values_in_sample_which_should_be_clamped != sample_value_to_clamp:
                    continue

                y = sample[range_to_predict_start:range_to_predict_end]
                y_str = ','.join(str(y))
                if y_str in predictions_dict:
                    predictions_dict[y_str] += np.exp(-1.0 * energy) / self.Z
                else:
                    predictions_dict[y_str] = np.exp(-1.0 * energy) / self.Z
                sample_idx += 1
            predictions_dicts.append(predictions_dict)

        return predictions_dicts

    def predictions_dicts_to_max_values(self, predictions_dicts, digit_num=1):
        if digit_num != 1:
            raise ValueError('digit_num != 1 not supported yet')
        predictions = []
        for predictions_dict in predictions_dicts:
            if predictions_dict == None or len(predictions_dict) == 0:
                prediction = np.random.randint(2)
            else:
                #print('DEBUG: predictions_dict:',predictions_dict);
                prediction_with_max_probability_tuple = max(predictions_dict.items(), key=operator.itemgetter(1))
                # print "max_y_probability_tuple: ",max_y_probability_tuple
                prediction = prediction_with_max_probability_tuple[0].split(',')[1:-1]  # [1:-1] cuts square brackets
                prediction = [int(y) for y in prediction][0]  # only 1 digit
            predictions.append(prediction)
        return predictions

    def predict_from_qubo(self, test_data, num_samps=100):
        predictions = []

        size_x = len(test_data[0])
        #size_x = len(test_data)
        # size_y = self.n_visible - size_x
        for x in test_data:
            clamped_idx = range(size_x)
            clamped_values = x

            # samples = self.sample_qubo(num_samps=num_samps)

            ranges_to_predict = [(size_x, self.n_visible)]
            predictions_dicts = self.prediction_sample_to_probability_dict(ranges_to_predict, clamped_idx,
                                                                           clamped_values)
            predictions_dict = predictions_dicts[0]
            if predictions_dict == None or len(predictions_dict) == 0:
                predictions.append(None)
                continue

            max_y_probability_tuple = max(predictions_dict.items(), key=operator.itemgetter(1))
            max_y = max_y_probability_tuple[0].split(',')[1:-1]  # [1:-1] cuts square brackets
            max_y = [int(y) for y in max_y]
            max_y_probability = max_y_probability_tuple[1]
            predictions.append((max_y, max_y_probability))
        return predictions

    def train(self, training_data, len_x=1, len_y=1, epochs=1, lr=1, decay=0.01, num_samps=100, momentum=0.8,
              epochs_to_test=1, num_sams_for_test=100, print_training_data=False):
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
        if len_y != 0:
            random_idx_to_predict = np.random.randint(len(training_data))
            print("Initial state, predicting data ", [
                training_data[random_idx_to_predict][:len_x]], "result to be:", self.predict_from_qubo(
                [training_data[random_idx_to_predict][:len_x]], num_samps=num_sams_for_test))
            print("--------------------------")

        for epoch in self.tqdm(range(epochs)):
            # single step

            # 1
            # 1.1 Take a training sample v
            random_selected_training_data_idx = epoch % len(training_data)

            v = training_data[random_selected_training_data_idx]
            if print_training_data:
                print("epoch #", self.n_epoch, "   training data:", len(v))
            # # 1.2 compute the probabilities of the hidden units
            clamped_idx = range(self.n_visible)
            clamped_values = v
            ranges_to_predict = []
            for i in range(self.n_hidden):
                range_to_predict_start = self.n_visible + i
                range_to_predict_end = self.n_visible + i + 1
                ranges_to_predict.append((range_to_predict_start, range_to_predict_end))

            # _ = self.sample_clamped_qubo(clamped_idx, clamped_values, num_samps=1)
            _ = self.sample_qubo(num_samps=num_samps)
            predictions_dicts = self.prediction_sample_to_probability_dict(ranges_to_predict, clamped_idx,
                                                                           clamped_values)

            h = self.predictions_dicts_to_max_values(predictions_dicts)

            # 2 Compute the outer product of v and h and call this the positive gradient.
            pos_grad = np.outer(v, h)
            pos_w_grad = np.zeros(shape=[self.n_visible + self.n_hidden, self.n_visible + self.n_hidden])
            # print "self.n_visible:",self.n_visible
            # print "self.n_hidden:",self.n_hidden
            # # print "left:",pos_J_grad[0:self.n_visible,self.n_visible:self.n_visible+self.n_hidden]
            # print "pos_grad:",pos_grad
            pos_w_grad[0:self.n_visible, self.n_visible:self.n_visible + self.n_hidden] = pos_grad

            # 3 
            # 3.1 From h, sample a reconstruction v' of the visible units, 
            clamped_values = h
            clamped_idx = range(self.n_visible, self.n_visible + self.n_hidden)

            ranges_to_predict = []
            for i in range(self.n_visible):
                ranges_to_predict.append((range_to_predict_start, range_to_predict_end))

            # _ = self.sample_clamped_qubo(clamped_idx, clamped_values, num_samps=1)
            _ = self.sample_qubo(num_samps=num_samps)
            predictions_dicts = self.prediction_sample_to_probability_dict(ranges_to_predict, clamped_idx,
                                                                           clamped_values)
            v_dash = self.predictions_dicts_to_max_values(predictions_dicts)
            # print "v: ",v,"    v_dash: ",v_dash

            # 3.2 then resample the hidden activations h' from this. (Gibbs sampling step)
            clamped_values = v_dash
            clamped_idx = range(self.n_visible)

            ranges_to_predict = []
            for i in range(self.n_hidden):
                ranges_to_predict.append((range_to_predict_start, range_to_predict_end))

            # _ = self.sample_clamped_qubo(clamped_idx, clamped_values, num_samps=1)
            _ = self.sample_qubo(num_samps=num_samps)
            predictions_dicts = self.prediction_sample_to_probability_dict(ranges_to_predict, clamped_idx,
                                                                           clamped_values)

            h_dash = self.predictions_dicts_to_max_values(predictions_dicts)

            # 4 Compute the outer product of v' and h' and call this the negative gradient.
            neg_grad = np.outer(v_dash, h_dash)
            neg_w_grad = np.zeros(shape=[self.n_visible + self.n_hidden, self.n_visible + self.n_hidden])
            neg_w_grad[0:self.n_visible, self.n_visible:self.n_visible + self.n_hidden] = neg_grad

            # 5 Let the update to the weight matrix W be the positive gradient minus the negative gradient, 
            #        times some learning rate                

            def update_with_momentum(to_update, old_delta, new_delta, momentum):
                to_update += old_delta * momentum + (1 - momentum) * new_delta

            delta_w = lr * (pos_w_grad - neg_w_grad)

            if not hasattr(self, "delta_w"):
                self.delta_w = delta_w

            for i in range(self.n_visible):
                for j in range(self.n_hidden):
                    # update_with_momentum(self.w[i,j], self.delta_w[(i,j)], delta_w[(i,j)], momentum)
                    self.w[i, j] += delta_w[(i, j)]
            # 6 Update the biases a and b analogously: a=epsilon (v-v'), b=epsilon (h-h')
            delta_visible_bias = lr * (np.array(v) - np.array(v_dash))
            # if not hasattr(self, "delta_visible_bias"):
            #     self.delta_visible_bias = delta_visible_bias
            # update_with_momentum(self.visible_bias, self.delta_visible_bias, delta_visible_bias, momentum)
            self.visible_bias += delta_visible_bias
            delta_hidden_bias = lr * (np.array(h) - np.array(h_dash))
            # if not hasattr(self, "delta_hidden_bias"):
            #     self.delta_hidden_bias = delta_hidden_bias
            # update_with_momentum(self.hidden_bias, self.delta_hidden_bias, delta_hidden_bias, momentum)
            self.hidden_bias += delta_hidden_bias
            # save hyperparameters
            lr *= (1 - decay)
            self.lr = lr

            self.delta_w = delta_w
            self.delta_visible_bias = delta_visible_bias
            self.delta_hidden_bias = delta_hidden_bias

            self.set_qubo();

            if epoch == 0 and len_y == 0 and self.image_height != None:
                plt.figure()
                plt.axis('off')
                plt.title("Image reconstructed before training", y=1.03)
                plt.imshow(np.array(v_dash).reshape(self.image_height, -1))

            # show training progress
            if epoch % epochs_to_test == 0 and len_y != 0:
                random_idx_to_predict = np.random.randint(len(training_data))
                print("predicting data ", [
                    training_data[random_idx_to_predict][:len_x]], "result to be:", self.predict_from_qubo(
                    [training_data[random_idx_to_predict][:len_x]], num_samps=num_sams_for_test))
                print("--------------------------")
            if epochs_to_test != -1 and epoch % epochs_to_test == 0 and len_y == 0:
                self.sample_qubo();
                sample = self.samples[0];
                mse = np.sum((np.array(v) - np.array(sample[:self.n_visible]))**2)/len_x;
                print('Epoch:',epoch+1,'MSE:',mse,'lr:',lr,'Result:',len(sample[:self.n_visible]));
                if self.image_height != None:
                    plt.figure()
                    plt.axis('off')
                    plt.title("Image reconstructed after training " + str(epoch + 1) + " epochs", y=1.03)

                    plt.imshow(sample[:len_x].reshape(self.image_height, -1))
                else:
                    print("sampling data to be", sample[:self.n_visible])

            original = training_data[0]
            samples = self.samples[0][:self.n_visible]
            if type(original) != list:
                original = original.tolist()
            if type(samples) != list:
                samples = samples.tolist()

            if original == samples:
                print("Stopped training early because the model can reconstruct the inputs")
                break

            self.n_epoch += 1
        return
