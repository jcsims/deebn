(ns deebn.dbn
  (:require [deebn.protocols :as p]
            [deebn.rbm :refer [build-rbm build-jd-rbm edn->CRBM edn->RBM]]
            [deebn.util :refer [query-hidden]]
            [clojure.core.matrix :as m]
            [clojure.tools.reader.edn :as edn]
            [clojure.core.matrix.select :as s]))

(defrecord DBN [rbms layers])
(defrecord CDBN [rbms layers classes])

(m/set-current-implementation :vectorz)

(defn build-dbn
  "Build a Deep Belief Network composed of Restricted Boltzmann Machines.

  layers is a vector of nodes in each layer, starting with the visible
  layer.

  Ex: [784 500 500 2000] -> 784-500 RBM, a 500-500 RBM, and a 500-2000
  RBM"
  [layers]
  (let [rbms (mapv #(build-rbm %1 %2) (butlast layers) (rest layers))]
    (->DBN rbms layers)))

(defn build-classify-dbn
  "Build a Deep Belief Network using Restricted Boltzmann Machines
  designed to classify an observation.

  See `build-dbn` for layers usage. classes is the number of possible
  classes the observation could be."
  [layers classes]
  (let [base (build-dbn (butlast layers))
        associative (build-jd-rbm (last (butlast layers)) (last layers) classes)]
    (map->CDBN {:rbms (conj (:rbms base) associative)
                :layers layers
                :classes classes})))

(defn train-dbn
  "Train a generative Deep Belief Network on a dataset. This trained
  model doesn't have an inherent value, unless the trained weights are
  subsequently used to initialize another network, e.g. a simple
  feedforward neural network.

  dataset is an unlabeled dataset used for unsupervised training.

  mean-field? is a key in the params map, and is a boolean indicating
  whether to use the expected value from a hidden layer as the input
  to the next RBM in the network, or use the sampled binary
  value. Defaults to true.

  query-final? is a key in the params map, and is used to determine if
  the final RBm trained is queried for the state of its hidden
  layer. This is only used when training the generative layers of a
  classification DBN, and changes the return type (both the trained
  DBN and the final transformed dataset are returned if this is true).

  See `train-rbm` for details on hyper-parameters passed in the param map."
  [dbn dataset params]
  (let [{:keys [mean-field? query-final?]
         :or {mean-field? true query-final? false}} params
         ;; Train the first RBM
         rbms (assoc (:rbms dbn) 0
                     (p/train-model (first (:rbms dbn)) dataset params))]
    (loop [rbms rbms
           iter 1
           data (query-hidden (first rbms) dataset mean-field?)]
      (if (>= iter (count rbms))
        (if query-final?
          {:dbn (assoc dbn :rbms rbms) :data data}
          (assoc dbn :rbms rbms))
        (let [new-rbm (p/train-model (get rbms iter) data params)]
          (recur (assoc rbms iter new-rbm)
                 (inc iter)
                 ;; Shortcut to prevent a final, unnecessary calculation
                 (when (or (< (inc iter) (count rbms)) query-final?)
                   (query-hidden new-rbm data mean-field?))))))))

(defn train-classify-dbn
  "Train a Deep Belief Network designed to classify data vectors.

  dataset is a softmax-labeled dataset, in the same format as that
  produced by deebn.mnist/load-data-with-softmax (the softmax precedes
  the data vector).

  Check train-rbm and train-dbn for more information about
  parameters."
  [dbn dataset params]
  (let [{:keys [mean-field?] :or {mean-field? true}} params
        softmaxes (m/matrix (s/sel dataset (s/irange)
                                   (range 0 (:classes dbn))))
        {gen-dbn :dbn xform-data :data}
        (train-dbn
         (assoc dbn :rbms
                (vec (butlast (:rbms dbn))))
         (m/matrix (s/sel dataset (s/irange)
                          (range (:classes dbn) (m/column-count dataset))))
         (assoc params :query-final? true))]
    (assoc dbn :rbms
           (conj (:rbms gen-dbn)
                 (p/train-model (last (:rbms dbn))
                                (m/join-along 1 softmaxes xform-data)
                                params)))))

(extend-protocol p/Trainable
  DBN
  (train-model [m dataset params]
    (train-dbn m dataset params)))

(extend-protocol p/Trainable
  CDBN
  (train-model [m dataset params]
    (train-classify-dbn m dataset params)))


;;;===========================================================================
;;; Testing a DBN trained on a data set
;;;===========================================================================

(defn classify-obv
  "Given a DBN and a single observation, return the model's prediction."
  [dbn obv]
  (let [prop-data (reduce #(query-hidden %2 %1 true)
                          obv
                          (butlast (:rbms dbn)))]
    (p/classify (last (:rbms dbn)) prop-data)))

(extend-protocol p/Classify
  CDBN
  (classify [m obv]
    (classify-obv m obv)))


(defn test-dbn
  "Test a classification Deep Belief Network on a given dataset.

  The dataset should have the label as the last entry in each
  observation."
  [dbn dataset]
  (let [columns (m/column-count dataset)
        labels (m/matrix (mapv vector (s/sel dataset (s/irange) (s/end dataset 1))))
        ;; Propagate the dataset up through the lower layers of the DBN
        prop-data (reduce #(query-hidden %2 %1 true)
                          (m/matrix (s/sel dataset
                                           (s/irange)
                                           (range 0 (dec columns))))
                          (butlast (:rbms dbn)))]
    (p/test-model (last (:rbms dbn)) (m/join-along 1 prop-data labels))))

(extend-protocol p/Testable
  CDBN
  (test-model [m dataset]
    (test-dbn m dataset)))


;;;===========================================================================
;;; Utility functions for a DBN
;;;===========================================================================

(defn save-dbn
  "Save a DBN."
  [dbn filepath]
  (spit filepath (pr-str dbn)))

(defn load-dbn
  "Load a DBN from disk."
  [filepath]
  (edn/read-string {:readers {'deebn.rbm.RBM edn->RBM
                              'deebn.rbm.CRBM edn->CRBM
                              'deebn.dbn.DBN map->DBN
                              'deebn.dbn.CDBN map->CDBN}} (slurp filepath)))
