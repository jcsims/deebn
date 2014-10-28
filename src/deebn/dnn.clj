(ns deebn.dnn
  (:refer-clojure :exclude [+ * -])
  (:require [deebn.util :refer [sigmoid gen-softmax]]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :refer [+ * -]]
            [clojure.core.matrix.random :as rand]
            [clojure.core.matrix.select :as s]))

(m/set-current-implementation :vectorz)

(defrecord DNN [weights biases layers classes])

(defn dbn->dnn
  "Given a pretrained Deep Belief Network, use the trained weights and
  biases to build a Deep Neural Network."
  [dbn classes]
  (let [top-layer {:w (m/matrix (repeatedly
                                 (last (:layers dbn))
                                 #(rand/sample-normal classes)))
                   :bias (m/zero-vector classes)}]
    (map->DNN {:classes classes
               :weights (conj (mapv :w (:rbms dbn)) (:w top-layer))
               :biases (conj (mapv :hbias (:rbms dbn)) (:bias top-layer))
               :layers (:layers dbn)})))

(defn prop-up
  "Given an input matrix, weight matrix, and bias vector, propagate
  the signal through the layer."
  [input weights bias]
  (m/emap sigmoid (+ bias (m/mmul input weights))))

(defn feed-forward
  "Given an initial input batch and a DNN, feed the batch through the
  net, retaining the output of each layer."
  [batch dnn]
  (reductions #(prop-up %1 (get %2 0) (get %2 1))
              batch
              (map #(vector %1 %2) (:weights dnn) (:biases dnn))))

(defn net-output
  "Propagate an input matrix through the network."
  [net input]
  (reduce #(prop-up %1 (first %2) (second %2))
          input
          (mapv #(vector %1 %2) (:weights net) (:biases net))))

(defn softmax->class
  "Get the predicted class from a softmax output."
  [x]
  (let [largest (m/emax x)
        indexed (zipmap x (range (m/row-count x)))]
    (get indexed largest)))

(defn layer-error
  "Calculate the error for a particular layer in a net, given the
  weights for the next layer, the error for the next layer, and the
  output for the current layer."
  [weights next-error output]
  (let [impact (m/mmul next-error (m/transpose weights))
        sig-deriv (* output (- 1 output))]
    (* impact sig-deriv)))

(defn update-layer
  "Update the weights and biases of a layer, given the previous
  weights and biases, input coming into the weights, the error for the
  layer, the learning rate, and the batch size."
  [weights biases input error learning-rate batch-size]
  (let [weights (- weights
                   (* (/ learning-rate batch-size)
                      (reduce + (mapv m/outer-product
                                      (m/rows input)
                                      (m/rows error)))))
        biases (- biases (* (/ learning-rate batch-size)
                            (reduce + (m/rows error))))]
    [weights biases]))

(defn train-batch
  "Given a batch of training data and a DNN, update the weights and
  biases accordingly."
  [batch dnn learning-rate min-gradient]
  (let [data (m/matrix (s/sel batch (s/irange) (range 0 (m/column-count batch))))
        targets (mapv #(gen-softmax %1 (:classes dnn))
                      (s/sel batch (s/irange) s/end))
        data (feed-forward data dnn)
        errors (m/emap #(* %1 (- 1 %1) (- %2 %1))
                       (last data) targets)
        errors (reverse (reductions #(layer-error (get %2 0) %1 (get %2 1))
                                    errors
                                    (map #(vector %1 %2)
                                         (reverse (rest (:weights dnn)))
                                         (reverse (butlast (rest data))))))
        [weights biases] (mapv #(update-layer %1 %2 %3 %4
                                              learning-rate (m/row-count batch))
                               (:weights dnn)
                               (:biases dnn)
                               (butlast data)
                               errors)]
    (assoc dnn :weights weights :biases biases)))

(defn train-epoch
  "Given a training dataset and a net, train it for one epoch (one
  pass over the dataset)."
  [net dataset learning-rate min-gradient batch-size]
  (loop [net net
         batch (s/sel dataset (range 0 batch-size) (s/irange))
         batch-num 0]
    (let [start (* batch-num batch-size)
          end (min (* (inc batch-num) batch-size) (m/row-count dataset))]
      (if (>= start (m/row-count dataset))
        net
        (do
          (print ".")
          (flush)
          (recur (train-batch batch net learning-rate min-gradient)
                 (m/matrix (s/sel dataset (range start end) (s/irange)))
                 (inc batch-num)))))))

(defn train-dnn
  "Given a labeled dataset, train a DNN.

  The dataset should have the label as the last element of each input
  vector.

  params is a map that may have the following keys:
  batch-size: default 100
  epochs: default 100
  learning-rate: default 0.01
  min-gradient: default 0.001"
  [dnn dataset params]
  (let [{:keys [batch-size epochs learning-rate min-gradient]
         :or {batch-size 100
              epochs 100
              learning-rate 0.01
              min-gradient 0.001}} params]
    (println "Training epoch 1")
    (loop [epoch 2
           net (train-epoch dnn dataset learning-rate min-gradient batch-size)]
      (if (> epoch epochs)
        net
        (do
          (println "Training epoch " epoch)
          (recur (inc epoch)
                 (train-epoch net dataset learning-rate
                              min-gradient batch-size)))))))
