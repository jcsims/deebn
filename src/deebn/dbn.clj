(ns deebn.dbn
  (:require [deebn.protocols :as p]
            [deebn.rbm :refer [build-rbm build-jd-rbm query-hidden sigmoid]]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as op]
            [clojure.core.matrix.select :as s]
            [taoensso.timbre.profiling :as prof]))

(defrecord DBN [rbms layers])
(defrecord CDBN [rbms layers classes])

(defn build-dbn
  "Build a Deep Belief Network composed of Restricted Boltzmann Machines.

  layers is a vector of nodes in each layer, starting with the visible layer.

  Ex: [784 500 500 2000] -> 784-500 RBM, a 500-500 RBM, and a
  top-level 500-2000 associative memory"
  [layers]
  {:pre [(> (count layers) 2)]}
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
      (println (m/shape data))
      (if (>= iter (count rbms))
        (if query-final?
          {:dbn (assoc dbn :rbms rbms) :data data}
          (assoc dbn :rbms rbms))
        (recur (assoc rbms iter (p/train-model (get rbms iter) data params))
               (inc iter)
               ;; Shortcut to prevent a final, unnecessary calculation
               (when (or (< (inc iter) (count rbms)) query-final?)
                 (query-hidden (get rbms iter) data mean-field?)))))))

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
