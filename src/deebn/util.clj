(ns deebn.util
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :refer [+ - * /]])
  (:import mikera.vectorz.Scalar))

(defn bernoulli
  "Take a single Bernoulli sample, given a probability"
  [p]
  (if (> (rand) p) 0 1))

(defn sigmoid
  "Sigmoid function, used as an activation function for nodes in a
  network."
  [^double x]
  (/ (+ 1 (Math/exp (* -1 x)))))

(defn query-hidden
  "Given an RBM and an input vector, query the RBM for the state of
  the hidden nodes."
  [rbm x mean-field?]
  (let [pre-sample (m/emap sigmoid (+ (:hbias rbm) (m/mmul x (:w rbm))))]
    (println (first pre-sample))
    (if mean-field? pre-sample
        (map bernoulli pre-sample))))

(defn gen-softmax
  "Generate a softmax output. x is the class represented by the
  output, with 0 represented by the first element in the vector."
  [x num-classes]
  (m/mset (m/array (repeat num-classes 0.1)) x 0.9))

(defn softmax-from-obv
  "Given an observation with label attached, replace the label value
  with an appropriate softmax unit. This assumes that the label is the
  last element in an observation."
  [x num-classes]
  (let [label (last x)
        obv (butlast x)
        new-label (gen-softmax label num-classes)]
    (vec (concat new-label obv))))

(defn get-min-position
  "Get the position of the minimum element of a collection."
  [x]
  (if (not (empty? x))
    (let [least (m/emin x)
          indexed (zipmap (map #(.get ^Scalar %) x) (range (count x)))]
      (get indexed least))))
