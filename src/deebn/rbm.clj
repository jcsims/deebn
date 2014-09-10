(ns deebn.rbm
  (:refer-clojure :exclude [+ - * / ==])
  (:import (java.io Writer))
  (:require [clojure.core.matrix.operators :refer [+ - * / ==]]
            [clojure.set :refer [difference]]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.select :as s]
            [clojure.tools.reader.edn :as edn]
            [taoensso.timbre.profiling :as prof])
  (:use [deebn.util]))

;;;===========================================================================
;;; Generate Restricted Boltzmann Machines
;;; ==========================================================================
(defrecord RBM [w vbias hbias w-vel vbias-vel hbias-vel visible hidden])

;; FIXME: This should really be Gaussian instead of uniform.
(defn rand-vec
  "Create a n-length vector of random real numbers in range [-range/2 range/2]"
  [n range]
  (take n (repeatedly #(- (rand range) (/ range 2)))))

(defn build-rbm
  "Factory function to produce an RBM record."
  [visible hidden]
  (let [w (m/matrix (take visible
                          (repeatedly #(rand-vec hidden 0.01))))
        w-vel (m/zero-matrix visible hidden)
        ;; TODO: The visual biases should really be set to
        ;; log(p_i/ (1  - p_i)), where p_i is the proportion of
        ;; training vectors in which unit i is turned on.
        vbias (m/zero-vector visible)
        hbias (m/zero-vector hidden)
        vbias-vel (m/zero-vector visible)
        hbias-vel (m/zero-vector hidden)]
    (->RBM w vbias hbias w-vel vbias-vel hbias-vel visible hidden)))

(defn build-jd-rbm
  "Build a joint density RBM for testing purposes.

  This RBM has two sets of visible units - the typical set
  representing each observation in the data set, and a softmax unit
  representing the label for each observation."
  [visible hidden classes]
  (build-rbm (+ visible classes) hidden))


;;;===========================================================================
;;; Train an RBM
;;; ==========================================================================
(defn sigmoid
  "Sigmoid function, used as an activation function for nodes in a
  network."
  [^double x]
  (/ (+ 1 (Math/exp (* -1 x)))))

(defn update-weights
  "Determine the weight gradient from this batch"
  [ph ph2 batch pv]
  (reduce + (map #(- (m/outer-product %1 %2) (m/outer-product %3 %4))
                 (m/rows batch) (m/rows ph)
                 (m/rows pv)    (m/rows ph2))))

(defn update-rbm
  "Single batch step update of RBM parameters"
  [batch rbm learning-rate momentum]
  (let [batch-size (m/row-count batch)
        ph (m/emap sigmoid (+ (:hbias rbm) (m/mmul batch (:w rbm))))
        h (m/emap bernoulli ph)
        pv (m/emap sigmoid (+ (:vbias rbm)
                              (m/mmul h (m/transpose (:w rbm)))))
        v (m/emap bernoulli pv)
        ph2 (m/emap sigmoid (+ (:hbias rbm) (m/mmul v (:w rbm))))
        delta-w (/ (update-weights ph ph2 batch pv) batch-size)
        delta-vbias (/ (reduce + (map #(- % %2)
                                      (m/rows batch)
                                      (m/rows pv)))
                       batch-size)
        delta-hbias (/ (reduce + (map #(- % %2)
                                      (m/rows h)
                                      (m/rows ph2)))
                       batch-size)
        squared-error (m/ereduce + (m/emap #(* % %) (- batch v)))
        w-vel (+ (* momentum (:w-vel rbm)) (* learning-rate delta-w))
        vbias-vel (+ (* momentum (:vbias-vel rbm)) (* learning-rate delta-vbias))
        hbias-vel (+ (* momentum (:hbias-vel rbm)) (* learning-rate delta-hbias))]
    (println " reconstruction error:" (/ squared-error batch-size))
    (assoc rbm
      :w (+ (:w rbm) w-vel)
      :vbias (+ (:vbias rbm) vbias-vel)
      :hbias (+ (:hbias rbm) hbias-vel)
      :w-vel w-vel :vbias-vel vbias-vel :hbias-vel hbias-vel)))

(defn train-epoch
  "Train a single epoch"
  [rbm dataset learning-rate momentum batch-size]
  (loop [rbm rbm
         batch (s/sel dataset (range 0 batch-size) (s/irange))
         batch-num 1]
    (let [start (* (dec batch-num) batch-size)
          end (min (* batch-num batch-size) (m/row-count dataset))]
      (if (>= start (m/row-count dataset))
        rbm
        (do
          (print "Batch:" batch-num)
          (recur (update-rbm batch rbm learning-rate momentum)
                 (s/sel dataset (range start end) (s/irange))
                 (inc batch-num)))))))

(defn select-overfitting-sets
  "Given a dataset, attempt to choose reasonable validation and test
  sets to monitor overfitting."
  [dataset]
  (let [obvs (m/row-count dataset)
        validation-indices (set (repeatedly (/ obvs 100) #(rand-int obvs)))
        validations (s/sel dataset (vec validation-indices) (s/irange))
        train-indices (difference (set (repeatedly (/ obvs 100) #(rand-int obvs)))
                                  validation-indices)
        train-sample (s/sel dataset (vec train-indices) (s/irange))]
    {:validations validations
     :train-sample train-sample}))

(defn free-energy
  "Compute the free energy of a given visible vector and RBM. Lower is
  better."
  [x rbm]
  (let [hidden-input (+ (:hbias rbm) (m/mmul x (:w rbm)))]
    (- (- (m/mmul x (:vbias rbm)))
       (reduce + (mapv #(Math/log (+ 1 (Math/exp %))) hidden-input)))))

(defn check-overfitting
  "Given an rbm, a sample from the training set, and a validation set,
  determine if the model is starting to overfit the data. This is
  measured by a difference in the average free energy over the
  training set sample and the validation set."
  [rbm train-sample validations]
  (let [avg-train-energy (mean (pmap #(free-energy %1 rbm)
                                     (m/rows train-sample)))
        avg-validation-energy (mean (pmap #(free-energy %1 rbm)
                                          (m/rows validations)))]
    (println "Avg training free energy:" avg-train-energy
             "Avg validation free energy:" avg-validation-energy
             "Gap:" (Math/abs (- avg-train-energy avg-validation-energy)))))

(defn train-rbm
  "Given a training set, train an RBM

  overfitting-sets is a map with the following two entries:
  validations is a vector of observations held out from training, to
  be used to monitor overfitting.

  train-sample is a vector of test observations that are used to
  monitor overfitting. These observations are used during training,
  and their free energy is compared to that of the validation set."
  [rbm dataset learning-rate momentum batch-size epochs
   & {:keys [overfitting-sets]
      :or {overfitting-sets (select-overfitting-sets dataset)}}]
  (let [validations (:validations overfitting-sets)
        train-sample (:train-sample overfitting-sets)]
    (loop [rbm rbm
           epoch 1]
      (if (> epoch epochs)
        rbm
        (do
          (if (== (rem epoch 2) 0)
            (check-overfitting rbm train-sample validations))
          (println "Training epoch" epoch)
          (recur (train-epoch rbm dataset learning-rate momentum batch-size)
                 (inc epoch)))))))


;;;===========================================================================
;;; Testing an RBM trained on a data set
;;;===========================================================================

(defn gen-softmax
  "Generate a softmax output. x is the class represented by the
  output, with 0 represented by the first element in the vector."
  [x num-classes]
  (m/mset (m/zero-vector num-classes) x 1))

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
        trials (mapv #(vec (concat % %2)) softmax-cases (repeat (butlast x)))
        results (mapv #(free-energy % rbm) trials)]
    (get-min-position results)))

(defn test-rbm
  "Test a joint density RBM trained on a data set. Returns an error
  percentage.

  dataset should have the label as the last entry in each
  observation."
  [rbm dataset num-classes]
  (let [num-observations (m/row-count dataset)
        predictions (doall (pmap #(get-prediction % rbm num-classes) dataset))
        errors (doall (map #(if (== (last %) %2) 0 1) dataset predictions))
        total (reduce + errors)]
    (double (/ total num-observations))))


;;;===========================================================================
;;; Utility functions for an RBM
;;;===========================================================================

;; This is designed for EDN printing, not actually visualizing the RBM
;; at the REPL
(defmethod clojure.core/print-method RBM [rbm ^Writer w]
  (.write w (str "#deebn.rbm/RBM {"
                 " :w " (:w rbm)
                 " :vbias " (:vbias rbm)
                 " :hbias " (:hbias rbm)
                 " :w-vel " (:w-vel rbm)
                 " :vbias-vel " (:vbias-vel rbm)
                 " :hbias-vel " (:hbias-vel rbm)
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
  (->RBM (m/matrix (:w data))
         (m/matrix (:vbias data))
         (m/matrix (:hbias data))
         (m/matrix (:w-vel data))
         (m/matrix (:vbias-vel data))
         (m/matrix (:hbias-vel data))
         (:visible data)
         (:hidden data)))

(defn load-rbm
  "Load a RBM from disk."
  [filepath]
  (edn/read-string {:readers {'deebn.rbm/RBM edn->RBM}} (slurp filepath)))
