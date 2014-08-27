(ns dbn-rbm.rbm
  (:refer-clojure :exclude [+ - * / ==])
  (:import (java.io Writer))
  (:use [clojure.core.matrix]
        [clojure.core.matrix.operators]
        [dbn-rbm.util]))

(declare sigmoid)

(defrecord RBM [w hbias vbias visible hidden])

(defn rand-vec
  "Create a n-length vector of random real numbers in range [-range/2 range/2]"
  [n range]
  (take n (repeatedly #(- (rand range) (/ range 2)))))

(defn build-rbm
  "Factory function to produce an RBM record."
  [visible hidden]
  (set-current-implementation :vectorz)
  (let [w (matrix (take visible
                        (repeatedly #(rand-vec hidden 0.01))))
        hbias (zero-vector hidden)
        ;; TODO: The visual biases should really be set to
        ;; log(p_i/ (1  - p_i)), where p_i is the proportion of
        ;; training vectors in which unit i is turned on.
        vbias (zero-vector visible)]
    (->RBM w hbias vbias visible hidden)))

(defn update-weights
  "Determine the weight gradient from this batch"
  [ph ph2 batch v]
  (let [updated (pmap #(- (outer-product % %2) (outer-product %3 %4))
                     (rows batch) (rows ph)
                     (rows v)     (rows ph2))]
    (reduce + updated)))

(defn update-rbm
  "Single batch step update of RBM parameters"
  [batch rbm learning-rate]
  (let [ph (emap sigmoid (+ (:hbias rbm) (mmul batch (:w rbm))))
        h (emap bernoulli ph)
        pv (emap sigmoid (+ (:vbias rbm)
                            (mmul h (transpose (:w rbm)))))
        v (emap bernoulli pv)
        ph2 (emap sigmoid (+ (:hbias rbm) (mmul v (:w rbm))))
        delta-w (update-weights ph ph2 batch v)
        delta-vbias (reduce + (map #(- % %2)
                                   (rows batch)
                                   (rows v)))
        delta-hbias (reduce + (map #(- % %2)
                                   (rows h)
                                   (rows ph2)))]
    (assoc rbm
      :w (+ (:w rbm) (mmul learning-rate delta-w))
      :vbias (+ (:vbias rbm) delta-vbias)
      :hbias (+ (:hbias rbm) delta-hbias))))

(defn train-rbm
  "Given a training set, train an RBM"
  [rbm train-set learning-rate batch-size epochs]
  (let [columns (column-count train-set)]
    (loop [rbm rbm
           batch (array (submatrix train-set 0 batch-size 0 columns))
           batch-num 1]
      (let [start (* (dec batch-num) batch-size)
            end (min (* batch-num batch-size) (row-count train-set))]
        (if (== end (row-count train-set))
          rbm
          (recur (update-rbm batch rbm learning-rate)
                 (array (submatrix train-set start (- end start) 0 columns))
                 (inc batch-num)))))))

(defmethod clojure.core/print-method RBM [rbm ^Writer w]
  (.write w (str "RBM with " (:visible rbm) " visible units "))
  (.write w (str "and " (:hidden rbm) " hidden units.\n")))

(defn sigmoid
  "Sigmoid function, used as an activation function for nodes in a
  network."
  [^double x]
  (/ (+ 1 (Math/exp (* -1 x)))))
