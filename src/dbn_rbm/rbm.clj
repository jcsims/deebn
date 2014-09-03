(ns dbn-rbm.rbm
  (:refer-clojure :exclude [+ - * / ==])
  (:import (java.io Writer))
  (:require [clojure.tools.reader.edn :as edn])
  (:use [clojure.core.matrix]
        [clojure.core.matrix.operators]
        [dbn-rbm.util]))

(declare sigmoid)

(defrecord RBM [w vbias hbias visible hidden])

;; FIXME: This should really be Gaussian instead of uniform.
(defn rand-vec
  "Create a n-length vector of random real numbers in range [-range/2 range/2]"
  [n range]
  (take n (repeatedly #(- (rand range) (/ range 2)))))

(defn build-rbm
  "Factory function to produce an RBM record."
  [visible hidden]
  (let [w (matrix (take visible
                        (repeatedly #(rand-vec hidden 0.01))))
        hbias (zero-vector hidden)
        ;; TODO: The visual biases should really be set to
        ;; log(p_i/ (1  - p_i)), where p_i is the proportion of
        ;; training vectors in which unit i is turned on.
        vbias (zero-vector visible)]
    (->RBM w vbias hbias visible hidden)))

(defn build-jd-rbm
  "Build a joint density RBM for testing purposes.

  This RBM has two sets of visible units - the typical set
  representing each observation in the data set, and a softmax unit
  representing the label for each observation."
  [visible hidden classes]
  (build-rbm (+ visible classes) hidden))

(defn update-weights
  "Determine the weight gradient from this batch"
  [ph ph2 batch pv]
  (let [updated (pmap #(- (outer-product % %2) (outer-product %3 %4))
                      (rows batch) (rows ph)
                      (rows pv)    (rows ph2))]
    (reduce + updated)))

(defn update-rbm
  "Single batch step update of RBM parameters"
  [batch rbm learning-rate]
  (let [batch-size (row-count batch)
        ph (emap sigmoid (+ (:hbias rbm) (mmul batch (:w rbm))))
        h (emap bernoulli ph)
        pv (emap sigmoid (+ (:vbias rbm)
                            (mmul h (transpose (:w rbm)))))
        v (emap bernoulli pv)
        ph2 (emap sigmoid (+ (:hbias rbm) (mmul v (:w rbm))))
        delta-w (/ (update-weights ph ph2 batch pv) batch-size)
        delta-vbias (/ (reduce + (map #(- % %2)
                                      (rows batch)
                                      (rows pv)))
                       batch-size)
        delta-hbias (/ (reduce + (map #(- % %2)
                                      (rows h)
                                      (rows ph2)))
                       batch-size)]
    (assoc rbm
      :w (+ (:w rbm) (mmul learning-rate delta-w))
      :vbias (+ (:vbias rbm) (* delta-vbias learning-rate))
      :hbias (+ (:hbias rbm) (* delta-hbias learning-rate)))))

(defn train-epoch
  "Train a single epoch"
  [rbm train-set learning-rate batch-size]
  (let [columns (column-count train-set)]
    (loop [rbm rbm
           batch (array (submatrix train-set 0 batch-size 0 columns))
           batch-num 1]
      (let [start (* (dec batch-num) batch-size)
            end (min (* batch-num batch-size) (row-count train-set))]
        (if (>= start (row-count train-set))
          rbm
          (do
            (println "Batch:" batch-num)
            (recur (update-rbm batch rbm learning-rate)
                   (array (submatrix train-set start (- end start) 0 columns))
                   (inc batch-num))))))))

(defn train-rbm
  "Given a training set, train an RBM"
  [rbm train-set learning-rate batch-size epochs]
  (loop [rbm rbm
         epoch 1]
    (if (> epoch epochs)
      rbm
      (do
        (println "Training epoch" epoch)
        (recur (train-epoch rbm train-set learning-rate batch-size)
               (inc epoch))))))

(defn sigmoid
  "Sigmoid function, used as an activation function for nodes in a
  network."
  [^double x]
  (/ (+ 1 (Math/exp (* -1 x)))))

(defn free-energy
  "Compute the free energy of a given visible vector and RBM. Lower is
  better."
  [x rbm]
  (let [hidden-input (+ (:hbias rbm) (mmul x (:w rbm)))]
    (- (- (mmul x (:vbias rbm)))
       (reduce + (mapv #(Math/log (+ 1 (Math/exp %))) hidden-input)))))


;;==============================================================================
;; Testing an RBM trained on a data set
;;==============================================================================


(defn gen-softmax
  "Generate a softmax output. x is the class represented by the
  output, with 0 represented by the first element in the vector."
  [x num-classes]
  (mset (zero-vector num-classes) x 1))

(defn softmax-from-obv
  "Given an observation with label attached, replace the label value
  with an appropriate softmax unit. This assumes that the label is the
  last element in an observation."
  [x num-classes]
  (let [label (peek x)
        obv (pop x)
        new-label (gen-softmax label num-classes)]
    (vec (concat new-label obv))))

(defn get-min-position
  "Get the position of the minimum element of a collection.

  TODO: This may not be the best approach."
  [x]
  (if (not (empty? x))
    (let [least (reduce min x)
          indexed (zipmap x (range (count x)))]
      (get indexed least))))

(defn get-prediction
  "For a given observation and RBM, return the predicted class."
  [x rbm num-classes]
  (let [softmax-cases (mapv #(gen-softmax % num-classes) (range num-classes))
        trials (mapv #(vec (concat % %2)) softmax-cases (repeat (pop x)))
        results (mapv #(free-energy % rbm) trials)]
    (get-min-position results)))

(defn test-rbm
  "Test a joint density RBM trained on a data set. Returns an error
  percentage."
  [rbm test-set num-classes]
  (let [num-observations (row-count test-set)
        predictions (doall (pmap #(get-prediction % rbm num-classes) test-set))
        errors (doall (map #(if (== (peek %) %2) 0 1) test-set predictions))
        total (reduce + errors)]
    (double (/ total num-observations))))


;;==============================================================================
;; Utility functions for an RBM
;;==============================================================================

;; This is designed for EDN printing, not actually visualizing the RBM
;; at the REPL
(defmethod clojure.core/print-method RBM [rbm ^Writer w]
  (.write w (str "#dbn-rbm.rbm/RBM {"
                 " :w " (:w rbm)
                 " :vbias " (:vbias rbm)
                 " :hbias " (:hbias rbm)
                 " :visible " (:visible rbm)
                 " :hidden " (:hidden rbm)
                 " }")))

(defn save-rbm
  "Save a RBM to disk."
  [rbm filepath]
  (spit filepath (pr-str rbm)))

(defn edn->RBM
  "The default map->RBM function provided by the defrecord doesn't
  provide us with the performant implementation (i.e. matrices and
  arrays from core.matrix), so this function adds a small step to
  ensure that."
  [data]
  (->RBM (matrix (:w data))
         (array (:vbias data))
         (array (:hbias data))
         (:visible data)
         (:hidden data)))

(defn load-rbm
  "Load a RBM from disk."
  [filepath]
  (edn/read-string {:readers {'dbn-rbm.rbm/RBM edn->RBM}} (slurp filepath)))
