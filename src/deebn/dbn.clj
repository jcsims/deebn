(ns deebn.dbn
  (:require [deebn.protocols :as p]
            [deebn.rbm :refer [build-rbm edn->CRBM edn->RBM]]
            [deebn.util :refer [query-hidden]]
            [clojure.core.matrix :as m]
            [clojure.tools.reader.edn :as edn]))

(defrecord DBN [rbms layers])

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
        (recur (assoc rbms iter (p/train-model (get rbms iter) data params))
               (inc iter)
               ;; Shortcut to prevent a final, unnecessary calculation
               (when (or (< (inc iter) (count rbms)) query-final?)
                 (query-hidden (get rbms iter) data mean-field?)))))))

(extend-protocol p/Trainable
  DBN
  (train-model [m dataset params]
    (train-dbn m dataset params)))


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
                              'deebn.dbn.DBN map->DBN}} (slurp filepath)))
