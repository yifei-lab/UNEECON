from __future__ import print_function, division
import numpy as np
import scipy as sp
import tensorflow as tf
import pandas as pd
import random
import argparse
import time
import sys
import os
from sklearn.model_selection import train_test_split
from collections import OrderedDict


class NeuralNetworkConstraintModel(object):
    def __init__(self):
        pass

    def build(self, n_inputs, hidden_shape=None,
              dropout_rate=0.5, learning_rate=0.001, seed=42, l2_penalty=0.,
              n_gauss_bins=20, float_type=tf.float32, int_type=tf.int32):
        self.n_inputs = n_inputs
        self.hidden_shape = hidden_shape
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.l2_penalty = l2_penalty
        self.n_gauss_bins = n_gauss_bins
        self.float_type = float_type
        self.int_type = int_type

        # reset graph and seed
        tf.reset_default_graph()
        tf.set_random_seed(seed)

        # build model from scratch
        self.__input_output_tensor()

        self.__variant_effect_layer()

        self.__selection_layer()

        self.__loss_function()

        self.__training_op()

    def fit(self, model_dir, data_train, gene_train, data_val, gene_val,
            data_test, gene_test, n_epochs=100, max_checks_without_progress=5):
        assert self.n_inputs is not None, "Please build model before fitting"
        saver = tf.train.Saver()
        best_loss = np.infty

        # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(n_epochs):
                # train model using sgd
                self.__train_model(sess, data_train, gene_train)

                # validate model. loss is defined as average
                # negative log likelihood per gene
                loss_val = self.__validate_model(sess, data_val, gene_val)
                print("Epoch = {}; Validation loss = {}".format(epoch,
                                                                loss_val))

                # early stopping
                if loss_val < best_loss:
                    # save parameter files
                    saver.save(sess, model_dir)
                    best_loss = loss_val
                    checks_without_progress = 0
                else:
                    checks_without_progress += 1
                    if checks_without_progress > max_checks_without_progress:
                        print("Early stopping!")
                        break
            print("best validation loss = {}".format(best_loss))

            # evaluate model in independent test data
            loss_test = self.__validate_model(sess, data_test, gene_test)
            print("test data loss = {}".format(loss_test))

    def predict(self, model_dir, all_data, all_gene,
                prediction_features, prediction_genes):
        tf.reset_default_graph()
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(model_dir + ".meta")
            saver.restore(sess, model_dir)

            self.__calculate_gene_random_effect(sess, all_data,
                                                all_gene)

            res, random_effect = self.__calculate_variant_score(sess,
                                                                prediction_features,
                                                                prediction_genes)

        return(res, random_effect)

    def save_tensor_board(self, logdir):
        writer = tf.summary.FileWriter(logdir=logdir,
                                       graph=tf.get_default_graph())
        writer.flush()

    def __calculate_gene_random_effect(self, sess, all_data, all_gene):
        gene_random_effect = dict()
        for i, gene in enumerate(all_genes):
            local_data = gene_data[i]
            feature = local_data.iloc[:, 4:].values
            rate = local_data[2].values
            obs = local_data[1].values
            prob_value = sess.run('loss/partial_loss:0',
                                  feed_dict={'input/X:0': feature,
                                             'output/Y:0': obs,
                                             'output/R:0': rate}
                                  )
            gene_random_effect[gene] = prob_value

        self.gene_random_effect = gene_random_effect

    def __calculate_variant_score(self, sess, prediction_features,
                                  prediction_genes):
        all_id = list()
        all_symbol = list()
        all_score = list()
        all_gene_mean_random_effect = OrderedDict([('gene', list()),
                                                   ('constraint', list()),
                                                   ('random_effect', list())
                                                  ])

        gauss_value = sess.run('selection/gauss_point:0')
        # print(gauss_value)
        for i, gene in enumerate(prediction_genes):
            if gene not in self.gene_random_effect:
                random_effect = np.log(sess.run('selection/gauss_weight:0'))
                # print("random effect = {}".format(random_effect))
            else:
                random_effect = self.gene_random_effect[gene]

            local_data = prediction_features[i]
            feature = local_data.iloc[:, 2:].values
            sel_value, gauss_p = sess.run(['selection/selection:0',
                                           'selection/gauss_point:0'],
                                           feed_dict={'input/X:0': feature})
            # print(random_effect)
            # print("shape = {}".format(random_effect.shape))
            # print("shape2 = {}".format(sel_value.shape))
            weight = random_effect - sp.misc.logsumexp(random_effect)
            # weight = np.expand_dims(weight, 0)
            exp_weight = np.exp(weight)
            exp_weight = exp_weight / np.sum(exp_weight)
            exp_weight = np.expand_dims(exp_weight, 0)
            # print("shape3 = {}".format(weight.shape))
            # score = np.sum(np.exp(weight) * sel_value, axis=1)
            score = np.sum(exp_weight * sel_value, axis=1)
            # print(gene)
            # print(np.min(score))
            # print(local_data.iloc[:, 0].values)
            all_id += local_data.iloc[:, 0].values.tolist()
            all_score += score.tolist()
            all_symbol += local_data.iloc[:, 1].values.tolist()

            mean_random_effect = np.sum(exp_weight * gauss_value)
            all_gene_mean_random_effect['gene'].append(gene)
            all_gene_mean_random_effect['random_effect'].append(mean_random_effect)
            all_gene_mean_random_effect['constraint'].append(1. - np.mean(score))
            # if i % 1000 == 0:
            #     print (i)

        df = pd.DataFrame(data=OrderedDict([("id", all_id),
                                            ("gene", all_symbol),
                                            ("score", 1. - np.array(all_score))]))

        all_gene_mean_random_effect = pd.DataFrame(data=all_gene_mean_random_effect)

        return(df, all_gene_mean_random_effect)

    def __train_model(self, sess, data_train, gene_train):
        for i, gene in enumerate(gene_train):
            local_data = data_train[i]
            feature = local_data.iloc[:, 4:].values

            rate = local_data[2].values
            obs = local_data[1].values
            sess.run([self.training_op],
                     feed_dict={self.X: feature,
                                self.Y: obs,
                                self.R: rate,
                                self.training: True}
                     )

    def __validate_model(self, sess, data_val, gene_val):
        all_loss_val = list()
        for i, gene in enumerate(gene_val):
            local_data = data_val[i]
            feature = local_data.iloc[:, 4:].values
            rate = local_data[2].values
            obs = local_data[1].values
            loss_val = sess.run(self.loss,
                                feed_dict={self.X: feature,
                                           self.Y: obs,
                                           self.R: rate,
                                           self.training: False}
                                )

            all_loss_val.append(loss_val)
        loss_val = np.mean(all_loss_val)
        return(loss_val)

    def __input_output_tensor(self):
        with tf.name_scope("input"):
            self.X = tf.placeholder(dtype=self.float_type,
                                    shape=[None, self.n_inputs],
                                    name="X")

        with tf.name_scope("training_flag"):
            self.training = tf.placeholder_with_default(False,
                                                        shape=[],
                                                        name="training")

        with tf.name_scope("output"):
            # variant presence/absence label
            self.Y = tf.placeholder(dtype=self.int_type, shape=[None],
                                    name="Y")
            # neutral mutation rate
            self.R = tf.placeholder(dtype=self.float_type, shape=[None],
                                    name="R")
            self.Y_reshaped = tf.tile(tf.expand_dims(self.Y, axis=1),
                                      [1, self.n_gauss_bins],
                                      name="Y_reshaped")

            self.R_reshaped = tf.tile(tf.expand_dims(self.R, axis=1),
                                      [1, self.n_gauss_bins],
                                      name="R_reshaped")

    def __variant_effect_layer(self):
        with tf.name_scope("variant_effect"):
            X_feature = self.X
            # initialize hidden shape
            if self.hidden_shape is None:
                hidden_shape_list = []
            else:
                hidden_shape_list = self.hidden_shape

            # add hidden layers
            for i, n_hiddens in enumerate(hidden_shape_list):
                X_feature = tf.keras.layers.Dense(n_hiddens, tf.keras.activations.relu,
                                                  kernel_regularizer=tf.keras.regularizers.l2(self.l2_penalty)
                                                 )(X_feature)

                X_feature = tf.keras.layers.Dropout(self.dropout_rate)(X_feature, training=self.training)

            # linear model for variant effect
            self.variant_effect = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(self.l2_penalty)
                                                       )(X_feature)

    def __selection_layer(self):
        with tf.name_scope("selection"):
            # Gauss-Hermite quadrature
            gauss_point, gauss_weight = np.polynomial.hermite.hermgauss(
                self.n_gauss_bins)

            # convert for standard Gaussian distribution
            gauss_point *= np.sqrt(2)
            gauss_weight /= np.sqrt(np.pi)

            # plot gauss points and weights
            # plt.plot(gauss_point, gauss_weight)
            # plt.xlim(-2, 2)
            # plt.show()
            # print(gauss_weight)
            # print(gauss_point)
            # print(sum(gauss_weight))

            # reshape gauss point and weights
            gauss_point = np.reshape(gauss_point, [1, -1])

            gauss_point = tf.Variable(gauss_point, dtype=self.float_type,
                                      trainable=False, name="gauss_point")

            gauss_point = tf.tile(gauss_point,
                                  tf.stack([tf.shape(self.X)[0], 1]))

            # reshape gauss weight
            # gauss_weight = np.reshape(gauss_weight, [1, -1])

            gauss_weight = tf.Variable(gauss_weight, dtype=self.float_type,
                                       trainable=False, name="gauss_weight")

            # standard deviation (log_scale)
            gauss_sd = tf.Variable([np.log(1)], dtype=self.float_type,
                                   trainable=True, name="log_sd")

            gene_constraint = tf.multiply(tf.exp(gauss_sd), gauss_point,
                                          name="gene_constraint")

            # X_reshaped = tf.concat([X_reshaped, gene_constraint], axis=2)
            # print("Gauss point", gauss_point.get_shape())
            # print("Gauss weight", gauss_weight.get_shape())
            # print("gene constraint", gene_constraint.get_shape())

            # relative evolution rate
            selection = tf.nn.sigmoid(self.variant_effect + gene_constraint,
                                      name="selection")
            # print(selection.get_shape())
            self.output = tf.multiply(self.R_reshaped, selection,
                                      name="output")
            self.gauss_weight = gauss_weight

    def __loss_function(self):
        with tf.name_scope("loss"):
            # calculate logit of the output probabilities
            # used a jitter term for stability
            # original implementation of logit function
            # logit = tf.log(self.output) - tf.log(1.0 - self.output)
            # logit = tf.log(self.output + 1.0e-9) - tf.log(1.0 - self.output)
            # another implementation of logit function
            self.output = self.output + 1.0e-9
            logit = tf.negative(tf.log(1. / self.output - 1.), name="logit")

            partial_loss = -tf.losses.sigmoid_cross_entropy(self.Y_reshaped,
                                                            logit,
                                                            reduction=tf.losses.Reduction.NONE)

            partial_loss = tf.reduce_sum(partial_loss, axis=0,
                                         name="partial_loss_reduced")
            partial_loss = tf.add(partial_loss, tf.log(self.gauss_weight,
                                                       name="prior_weight"),
                                  name="partial_loss")
            # print("partial_loss", partial_loss.get_shape())

            # reduce partial loss
            self.loss = tf.negative(tf.reduce_logsumexp(partial_loss),
                                    name="final_loss")
            # print("loss", self.loss.get_shape())

    def __training_op(self):
        self.training_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", dest="train", required=True,
                        help="training data file")

    parser.add_argument("-p", dest="prediction", required=True,
                        help="feature file for prediction")

    parser.add_argument("-s", dest="seed", required=False, default=-1,
                        type=int, help="random seed for data preparation")

    parser.add_argument("-r", dest="training_seed", required=False, default=-1,
                        type=int, help="random seed for training")

    parser.add_argument("-u", dest="hidden", nargs="+", required=False, default=None,
                        type=int, help="numbers of hidden units")

    parser.add_argument("-l", dest="rate", required=False, default=1.0e-3,
                        type=float, help="learning rate")

    parser.add_argument("-d", dest="dropout", required=False, default=0.5,
                        type=float, help="dropout rate")

    parser.add_argument("-w", dest="weight_decay", required=False, default=0.,
                        help="weight decay (L2 penalty)")

    parser.add_argument("-n", dest="epoch", required=False, default=50,
                        type=int, help="number of epochs")

    parser.add_argument("-o", dest="output", required=True,
                        help="output directory")

    args = parser.parse_args()

    if args.seed == -1:
        args.seed = int(time.time())

    if args.training_seed == -1:
        args.training_seed = int(time.time())

    print("Training parameters:")
    for key, value in vars(args).items():
        print("{}={}".format(key, value))

    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)

    # read training data
    data = pd.read_csv(args.train, compression="gzip",
                       header=None, delim_whitespace=True)
    model_dir = sys.argv[3]
    n_inputs = data.shape[1] - 4

    model_dir = os.path.join(args.output, "trained_model/")

    # process data
    all_genes = list()
    gene_data = list()

    gb = data.groupby(3)
    for x in gb.groups:
        all_genes.append(x)
        gene_data.append(gb.get_group(x))
        gene_train, gene_val, data_train, data_val = train_test_split(all_genes, gene_data, test_size=0.2, random_state=42)
        gene_test, gene_val, data_test, data_val = train_test_split(gene_val, data_val, test_size=0.5, random_state=42)

    # write gene lists
    pd.DataFrame({'gene': gene_train}).to_csv(os.path.join(args.output,
                                                           "training_gene.txt"),
                                              header=False, index=False)

    pd.DataFrame({'gene': gene_val}).to_csv(os.path.join(args.output,
                                                         "validation_gene.txt"),
                                            header=False, index=False)

    pd.DataFrame({'gene': gene_test}).to_csv(os.path.join(args.output,
                                                          "test_gene.txt"),
                                             header=False, index=False)

    # read feature data for prediction
    data = pd.read_csv(args.prediction,
                       compression='gzip',
                       header=None, delim_whitespace=True)
    # print(data.head())

    prediction_genes = list()
    prediction_features = list()

    gb = data.groupby(1)
    for x in gb.groups:
        prediction_genes.append(x)
        prediction_features.append(gb.get_group(x))

    model = NeuralNetworkConstraintModel()
    model.build(n_inputs, args.hidden, args.dropout,
                args.rate, args.training_seed, args.weight_decay)

    model.save_tensor_board(os.path.join(args.output, "tensorboard/"))
    model.fit(model_dir, data_train, gene_train, data_val, gene_val, data_test, gene_test, args.epoch)
    res, random_effect = model.predict(model_dir, gene_data, all_genes,
                                       prediction_features,
                                       prediction_genes)
    # res.to_csv("predicted_variant_score.csv", sep="\t", index=False)
    res.to_csv(os.path.join(args.output, "variant_score.tsv"), sep="\t", index=False, float_format='%.6f')
    # random_effect.to_csv("predicted_random_effect.csv", sep="\t", index=False)
    random_effect.to_csv(os.path.join(args.output, "gene_random_effect.tsv"), sep="\t", index=False, float_format='%.6f')
