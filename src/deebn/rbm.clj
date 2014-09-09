(ns deebn.rbm
  (:refer-clojure :exclude [+ - * / ==])
  (:import (java.io Writer))
  (:require [clojure.tools.reader.edn :as edn])
  (:use [clojure.core.matrix]
        [clojure.core.matrix.operators]
        [clojure.core.matrix.dataset]
        [clojure.core.matrix.select]
        [deebn.util]))

(declare sigmoid)

(defrecord RBM [w vbias hbias w-vel vbias-vel hbias-vel visible hidden])

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
        w-vel (zero-matrix visible hidden)
        ;; TODO: The visual biases should really be set to
        ;; log(p_i/ (1  - p_i)), where p_i is the proportion of
        ;; training vectors in which unit i is turned on.
        vbias (zero-vector visible)
        hbias (zero-vector hidden)
        vbias-vel (zero-vector visible)
        hbias-vel (zero-vector hidden)]
    (->RBM w vbias hbias w-vel vbias-vel hbias-vel visible hidden)))

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
  [batch rbm learning-rate momentum]
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
                       batch-size)
        squared-error (ereduce + (emap #(* % %) (- batch v)))
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
  (let [columns (column-count dataset)]
    (loop [rbm rbm
           batch (array (submatrix dataset 0 batch-size 0 columns))
           batch-num 1]
      (let [start (* (dec batch-num) batch-size)
            end (min (* batch-num batch-size) (row-count dataset))]
        (if (>= start (row-count dataset))
          rbm
          (do
            (print "Batch:" batch-num)
            (recur (update-rbm batch rbm learning-rate momentum)
                   (array (submatrix dataset start (- end start) 0 columns))
                   (inc batch-num))))))))

(defn select-overfitting-sets
  "Given a dataset, attempt to choose reasonable validation and test
  sets to monitor overfitting."
  [dataset]
  (let [obvs (row-count dataset)
        validation-indices (set (repeatedly (/ obvs 100) #(rand-int obvs)))
        validations (select-rows dataset validation-indices)
        train-indices (clojure.set/difference
                      (set (repeatedly (/ obvs 100) #(rand-int obvs)))
                      validation-indices)
        train-sample (select-rows dataset train-indices)]
    {:validations validations
     :train-sample train-sample}))

(defn free-energy
  "Compute the free energy of a given visible vector and RBM. Lower is
  better."
  [x rbm]
  (let [hidden-input (+ (:hbias rbm) (mmul x (:w rbm)))]
    (- (- (mmul x (:vbias rbm)))
       (reduce + (mapv #(Math/log (+ 1 (Math/exp %))) hidden-input)))))

(defn check-overfitting
  "Given an rbm, a sample from the training set, and a validation set,
  determine if the model is starting to overfit the data. This is
  measured by a difference in the average free energy over the
  training set sample and the validation set."
  [rbm train-sample validations]
  (let [avg-train-energy (mean (pmap #(free-energy %1 rbm) (rows train-sample)))
        avg-validation-energy (mean (pmap #(free-energy %1 rbm)
                                          (rows validations)))]
    (println "Avg training free energy:" avg-train-energy
             "Avg validation free energy:" avg-validation-energy
             "Gap:" (abs (- avg-train-energy avg-validation-energy)))))

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

(defn sigmoid
  "Sigmoid function, used as an activation function for nodes in a
  network."
  [^double x]
  (/ (+ 1 (Math/exp (* -1 x)))))


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
  (let [num-observations (row-count dataset)
        predictions (doall (pmap #(get-prediction % rbm num-classes) dataset))
        errors (doall (map #(if (== (last %) %2) 0 1) dataset predictions))
        total (reduce + errors)]
    (double (/ total num-observations))))


;;==============================================================================
;; Utility functions for an RBM
;;==============================================================================

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
  (->RBM (matrix (:w data))
         (matrix (:vbias data))
         (matrix (:hbias data))
         (matrix (:w-vel data))
         (matrix (:vbias-vel data))
         (matrix (:hbias-vel data))
         (:visible data)
         (:hidden data)))

(defn load-rbm
  "Load a RBM from disk."
  [filepath]
  (edn/read-string {:readers {'deebn.rbm/RBM edn->RBM}} (slurp filepath)))
