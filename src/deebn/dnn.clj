(ns deebn.dnn
  (:refer-clojure :exclude [+ * -])
  (:require [deebn.protocols :as p]
            [deebn.util :refer [sigmoid gen-softmax]]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :refer [+ * -]]
            [clojure.core.matrix.random :as rand]
            [clojure.core.matrix.select :as s]
            [clojure.tools.reader.edn :as edn]
            [taoensso.timbre :refer [spy debug]])
  (:import java.io.Writer))

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
  (reductions #(prop-up %1 (first %2) (second %2))
              batch
              (map #(vector %1 %2) (:weights dnn) (:biases dnn))))

(defn net-output
  "Propagate an input matrix through the network."
  [net input]
  (m/matrix (reduce #(prop-up %1 (first %2) (second %2))
                    input
                    (mapv #(vector %1 %2) (:weights net) (:biases net)))))

(defn layer-error
  "Calculate the error for a particular layer in a net, given the
  weights for the next layer, the error for the next layer, and the
  output for the current layer."
  [weights next-error output]
  (* (m/mmul next-error (m/transpose weights)) (* output (- 1 output))))

(defn update-layer
  "Update the weights and biases of a layer, given the previous
  weights and biases, input coming into the weights, the error for the
  layer, the learning rate, and the batch size."
  [weights biases input error learning-rate lambda batch-size observations]
  (let [weights (- (* weights (- 1 (/ (* learning-rate lambda) observations)))
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
  [batch dnn observations learning-rate lambda]
  (let [data (m/matrix (s/sel batch
                              (s/irange)
                              (range 0 (dec (m/column-count batch)))))
        targets (mapv #(gen-softmax %1 (:classes dnn))
                      (s/sel batch (s/irange) s/end))
        data (feed-forward data dnn)
        errors (m/emap #(- %1 %2)
                       (last data) targets)
        errors (reverse (reductions #(layer-error (first %2) %1 (second %2))
                                    errors
                                    (map #(vector %1 %2)
                                         (reverse (rest (:weights dnn)))
                                         (reverse (butlast (rest data))))))
        updated (mapv #(update-layer %1 %2 %3 %4
                                     learning-rate
                                     lambda
                                     (m/row-count batch)
                                     observations)
                      (:weights dnn)
                      (:biases dnn)
                      (butlast data)
                      errors)]
    (assoc dnn :weights (mapv first updated) :biases (mapv second updated))))

(defn train-epoch
  "Given a training dataset and a net, train it for one epoch (one
  pass over the dataset)."
  [net dataset observations learning-rate lambda batch-size]
  (loop [net net
         batch (m/matrix (s/sel dataset (range 0 batch-size) (s/irange)))
         batch-num 0]
    (let [start (* batch-num batch-size)
          end (min (* (inc batch-num) batch-size) (m/row-count dataset))]
      (if (>= start (m/row-count dataset))
        net
        (do
          (recur (train-batch batch net observations learning-rate lambda)
                 (m/matrix (s/sel dataset (range start end) (s/irange)))
                 (inc batch-num)))))))

(defn train-top-layer
  "Pre-train the top logistic regression layer before moving to fine-tuning."
  [dnn dataset observations batch-size epochs learning-rate lambda]
  (println "Propagating dataset through DNN...")
  (let [top-layer {:weights (vector (last (:weights dnn)))
                   :biases (vector (last (:biases dnn)))
                   :classes (:classes dnn)}
        output (net-output
                (assoc dnn
                  :weights (butlast (:weights dnn))
                  :biases (butlast (:biases dnn)))
                (m/matrix
                 (s/sel dataset
                        (s/irange)
                        (range 0 (dec (m/column-count dataset))))))
        targets (m/reshape (m/matrix (s/sel dataset (s/irange) s/end))
                           [(m/row-count dataset) 1])
        dataset (m/join-along 1 output targets)]
    (println "Pre-training logistic regression layer, epoch 1")
    (loop [epoch 2
           top-layer (train-epoch top-layer dataset observations
                                  learning-rate lambda batch-size)]
      (if (> epoch epochs)
        (assoc dnn
          :weights (assoc (:weights dnn) (dec (count (:weights dnn)))
                          (first (:weights top-layer)))
          :biases (assoc (:biases dnn) (dec (count (:biases dnn)))
                         (first (:biases top-layer))))
        (do
          (println "Pre-training logistic regression layer, epoch" epoch)
          (recur (inc epoch)
                 (train-epoch top-layer dataset observations
                              learning-rate lambda batch-size)))))))

(defn train-dnn
  "Given a labeled dataset, train a DNN.

  The dataset should have the label as the last element of each input
  vector.

  params is a map that may have the following keys:
  batch-size: default 100
  epochs: default 100
  learning-rate: default 0.5
  lambda: default 0.1 "
  [dnn dataset params]
  (let [{:keys [batch-size epochs learning-rate lambda train-lower]
         :or {batch-size 100
              epochs 100
              learning-rate 0.5
              lambda 0.1}} params
              observations (m/row-count dataset)
              net (train-top-layer dnn dataset observations batch-size
                                   epochs learning-rate lambda)]
    (println "Training epoch 1")
    (loop [epoch 2
           net (train-epoch net dataset observations
                            learning-rate lambda batch-size)]
      (if (> epoch epochs)
        net
        (do
          (println "\nTraining epoch" epoch)
          (recur (inc epoch)
                 (train-epoch net dataset observations learning-rate
                              lambda batch-size)))))))

(extend-protocol p/Trainable
  DNN
  (train-model [m dataset params]
    (train-dnn m dataset params)))


;;;===========================================================================
;;; Testing a DNN trained on a data set
;;;===========================================================================

(defn softmax->class
  "Get the predicted class from a softmax output."
  [x]
  (let [largest (m/emax x)
        indexed (zipmap x (range (m/row-count x)))]
    (get indexed largest)))

(defn classify-obv
  "Given a DNN and a single observation, return the model's prediction."
  [dnn obv]
  (softmax->class (net-output dnn obv)))

(extend-protocol p/Classify
  DNN
  (classify [m obv]
    (classify-obv m obv)))

(defn test-dnn
  "Test a Deep Neural Network on a dataset. Returns an error percentage.

  dataset should have the label as the last entry in each observation."
  [dnn dataset]
  (let [num-observations (m/row-count dataset)
        predictions (mapv #(softmax->class (net-output dnn %1))
                          (m/matrix (s/sel
                                     dataset
                                     (s/irange)
                                     (range 0 (dec (m/column-count dataset))))))
        errors (mapv #(if (== (last %1) %2) 0 1) dataset predictions)]
    (double (/ (m/esum errors) num-observations))))

(extend-protocol p/Testable
  DNN
  (test-model [m dataset]
    (test-dnn m dataset)))

;;;===========================================================================
;;; Utility functions for a DNN
;;;===========================================================================

(defmethod clojure.core/print-method DNN print-DNN [dnn ^Writer w]
  (.write w (str "#deebn.dnn.DNN {"
                 " :weights " (mapv m/to-nested-vectors (:weights dnn))
                 " :biases " (mapv m/to-nested-vectors (:biases dnn))
                 " :layers " (:layers dnn)
                 " :classes " (:classes dnn)
                 " }")))

(defn save-dnn
  "Save a DNN to disk."
  [dnn filepath]
  (spit filepath (pr-str dnn)))

(defn edn->DNN
  "The default map->DNN function provided by the defrecord doesn't
  provide us with the performant implementation (i.e. matrices and
  arrays from core.matrix), so this function adds a small step to
  ensure that."
  [data]
  (->DNN (mapv m/matrix (:weights data))
         (mapv m/matrix (:biases data))
         (:layers data)
         (:classes data)))

(defn load-dnn
  "Load a DNN from disk."
  [filepath]
  (edn/read-string {:readers {'deebn.dnn.DNN edn->DNN}} (slurp filepath)))
