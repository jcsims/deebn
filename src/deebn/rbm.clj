(ns deebn.rbm
  (:refer-clojure :exclude [+ - * / ==])
  (:require [deebn.protocols :refer [Testable Trainable Classify]]
            [deebn.util :refer [bernoulli gen-softmax
                                get-min-position sigmoid]]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :refer [+ - * / ==]]
            [clojure.core.matrix.select :as s]
            [clojure.core.matrix.random :as rand]
            [clojure.core.matrix.stats :as stats]
            [clojure.set :refer [difference]]
            [clojure.tools.reader.edn :as edn])
  (:import java.io.Writer))

(m/set-current-implementation :vectorz)

;;;===========================================================================
;;; Generate Restricted Boltzmann Machines
;;; ==========================================================================
;; We define a purely generative RBM, trained without any class
;; labels, and a classification RBM
(defrecord RBM [w vbias hbias w-vel vbias-vel hbias-vel visible hidden])
(defrecord CRBM [w vbias hbias w-vel vbias-vel hbias-vel visible hidden classes])

(defn build-rbm
  "Factory function to produce an RBM record."
  [visible hidden]
  (let [w (m/matrix (repeatedly visible #(/ (rand/sample-normal hidden) 100)))
        w-vel (m/zero-matrix visible hidden)
        ;; TODO: The visual biases should really be set to
        ;; log(p_i/ (1  - p_i)), where p_i is the proportion of
        ;; training vectors in which unit i is turned on.
        vbias (m/zero-vector visible)
        hbias (m/array (repeat hidden -4))
        vbias-vel (m/zero-vector visible)
        hbias-vel (m/zero-vector hidden)]
    (->RBM w vbias hbias w-vel vbias-vel hbias-vel visible hidden)))

(defn build-jd-rbm
  "Factory function to build a joint density RBM for testing purposes.

  This RBM has two sets of visible units - the typical set
  representing each observation in the data set, and a softmax unit
  representing the label for each observation. These are combined, and
  the label becaomes part of the input vector."
  [visible hidden classes]
  (let [rbm (build-rbm (+ visible classes) hidden)]
    (map->CRBM (assoc rbm :classes classes))))


;;;===========================================================================
;;; Train an RBM
;;; ==========================================================================


(defn update-weights
  "Determine the weight gradient from this batch"
  [ph ph2 batch pv]
  (reduce + (map #(- (m/outer-product %1 %2) (m/outer-product %3 %4))
                 (m/rows batch) (m/rows ph)
                 (m/rows pv)    (m/rows ph2))))

;; TODO: Implement CD-K - currently CD-1 is hard-coded.
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
        w-vel (+ (* momentum (:w-vel rbm)) (* learning-rate delta-w))
        vbias-vel (+ (* momentum (:vbias-vel rbm))
                     (* learning-rate delta-vbias))
        hbias-vel (+ (* momentum (:hbias-vel rbm))
                     (* learning-rate delta-hbias))]
    (assoc rbm
      :w (+ (:w rbm) w-vel)
      :vbias (+ (:vbias rbm) vbias-vel)
      :hbias (+ (:hbias rbm) hbias-vel)
      :w-vel w-vel :vbias-vel vbias-vel :hbias-vel hbias-vel)))

(defn train-epoch
  "Train a single epoch"
  [rbm dataset learning-rate momentum batch-size]
  (loop [rbm rbm
         batch (m/matrix (s/sel dataset (range 0 batch-size) (s/irange)))
         batch-num 1]
    (let [start (* (dec batch-num) batch-size)
          end (min (* batch-num batch-size) (m/row-count dataset))]
      (if (>= start (m/row-count dataset))
        rbm
        (do
          (print ".")
          (flush)
          (recur (update-rbm batch rbm learning-rate momentum)
                 (m/matrix (s/sel dataset (range start end) (s/irange)))
                 (inc batch-num)))))))

(defn select-overfitting-sets
  "Given a dataset, attempt to choose reasonable validation and test
  sets to monitor overfitting."
  [dataset]
  (let [obvs (m/row-count dataset)
        validation-indices (set (repeatedly (/ obvs 100) #(rand-int obvs)))
        validations (m/matrix (s/sel dataset
                                     (vec validation-indices) (s/irange)))
        train-indices (difference
                       (set (repeatedly (/ obvs 100)
                                        #(rand-int obvs))) validation-indices)
        train-sample (m/matrix (s/sel dataset (vec train-indices) (s/irange)))]
    {:validations validations
     :train-sample train-sample
     :dataset  (s/sel dataset (s/exclude (vec validation-indices))
                      (s/irange))}))

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
  (let [avg-train-energy (stats/mean (pmap #(free-energy %1 rbm)
                                           (m/rows train-sample)))
        avg-validation-energy (stats/mean (pmap #(free-energy %1 rbm)
                                                (m/rows validations)))]
    (Math/abs ^Double (- avg-train-energy avg-validation-energy))))

(defn train-rbm
  "Given a training set, train an RBM

  params is a map with various options:
  learning-rate: defaults to 0.1
  initial-momentum: starting momentum. Defaults to 0.5
  momentum: momentum after `momentum-delay` epochs have passed. Defaults to 0.9
  momentum-delay: epochs after which `momentum` is used instead of
  `initial-momentum`. Defaults to 3
  batch-size: size of each mini-batch. Defaults to 10
  epochs: number of times to train the model over the entire training set.
  Defaults to 100
  gap-delay: number of epochs elapsed before early stopping is considered
  gap-stop-delay: number of sequential epochs where energy gap is increasing
  before stopping"
  [rbm dataset params]
  (let [{:keys [validations train-sample dataset]}
        (select-overfitting-sets dataset)
        {:keys [learning-rate initial-momentum momentum momentum-delay
                batch-size epochs gap-delay gap-stop-delay]
         :or {learning-rate 0.1
              initial-momentum 0.5
              momentum 0.9
              momentum-delay 3
              batch-size 10
              epochs 100
              gap-delay 10
              gap-stop-delay 2}} params]
    (println "Training epoch 1")
    (loop [rbm (train-epoch rbm dataset learning-rate
                            initial-momentum batch-size)
           epoch 2
           energy-gap (check-overfitting rbm train-sample validations)
           gap-inc-count 0]
      (if (> epoch epochs)
        rbm
        (do (println "\nTraining epoch" epoch)
            (let [curr-momentum (if (> epoch momentum-delay)
                                  momentum initial-momentum)
                  rbm (train-epoch rbm dataset learning-rate
                                   curr-momentum batch-size)
                  gap-after-train (check-overfitting rbm train-sample
                                                     validations)
                  _ (println "\nGap pre-train:" energy-gap
                             "After train:" gap-after-train)]
              (if (and (>= epoch gap-delay)
                       (neg? (- energy-gap gap-after-train))
                       (>= gap-inc-count gap-stop-delay))
                rbm
                (recur rbm
                       (inc epoch)
                       gap-after-train
                       (if (neg? (- energy-gap gap-after-train))
                         (inc gap-inc-count)
                         0)))))))))

(extend-protocol Trainable
  CRBM
  (train-model [m dataset params]
    (train-rbm m dataset params))
  RBM
  (train-model [m dataset params]
    (train-rbm m dataset params)))


;;;===========================================================================
;;; Testing an RBM trained on a data set
;;;===========================================================================

(defn get-prediction
  "For a given observation and RBM, return the predicted class."
  [x rbm num-classes labeled?]
  (let [softmax-cases  (mapv #(gen-softmax % num-classes) (range num-classes))
        trials  (m/matrix (mapv #(m/join % %2) softmax-cases
                                (if labeled?
                                  (repeat (butlast x))
                                  (repeat x))))
        results (mapv #(free-energy % rbm) trials)]
    (get-min-position results)))

(extend-protocol Classify
  CRBM
  (classify [m obv]
    (get-prediction obv m (:classes m) false)))

(defn test-rbm
  "Test a joint density RBM trained on a data set. Returns an error
  percentage.

  dataset should have the label as the last entry in each
  observation."
  [rbm dataset num-classes]
  (let [num-observations (m/row-count dataset)
        predictions (pmap #(get-prediction % rbm num-classes true) dataset)
        errors (mapv #(if (== (last %) %2) 0 1) dataset predictions)
        total (m/esum errors)]
    (double (/ total num-observations))))

(extend-protocol Testable
  CRBM
  (test-model [m dataset]
    (test-rbm m dataset (:classes m))))


;;;===========================================================================
;;; Utility functions for an RBM
;;;===========================================================================

;; This is designed for EDN printing, not actually visualizing the RBM
;; at the REPL (this is only needed because similar methods are not
;; defined for clojure.core.matrix implementations)
(defmethod clojure.core/print-method RBM print-RBM [rbm ^Writer w]
  (.write w (str "#deebn.rbm.RBM {"
                 " :w " (m/to-nested-vectors (:w rbm))
                 " :vbias " (m/to-nested-vectors (:vbias rbm))
                 " :hbias " (m/to-nested-vectors (:hbias rbm))
                 " :w-vel " (m/to-nested-vectors (:w-vel rbm))
                 " :vbias-vel " (m/to-nested-vectors (:vbias-vel rbm))
                 " :hbias-vel " (m/to-nested-vectors (:hbias-vel rbm))
                 " :visible " (:visible rbm)
                 " :hidden " (:hidden rbm)
                 " }")))

(defmethod clojure.core/print-method CRBM print-CRBM [rbm ^Writer w]
  (.write w (str "#deebn.rbm.CRBM {"
                 " :w " (m/to-nested-vectors (:w rbm))
                 " :vbias " (m/to-nested-vectors (:vbias rbm))
                 " :hbias " (m/to-nested-vectors (:hbias rbm))
                 " :w-vel " (m/to-nested-vectors (:w-vel rbm))
                 " :vbias-vel " (m/to-nested-vectors (:vbias-vel rbm))
                 " :hbias-vel " (m/to-nested-vectors (:hbias-vel rbm))
                 " :visible " (:visible rbm)
                 " :hidden " (:hidden rbm)
                 " :classes " (:classes rbm)
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

(defn edn->CRBM
  "The default map->RBM function provided by the defrecord doesn't
  provide us with the performant implementation (i.e. matrices and
  arrays from core.matrix), so this function adds a small step to
  ensure that."
  [data]
  (->CRBM (m/matrix (:w data))
          (m/matrix (:vbias data))
          (m/matrix (:hbias data))
          (m/matrix (:w-vel data))
          (m/matrix (:vbias-vel data))
          (m/matrix (:hbias-vel data))
          (:visible data)
          (:hidden data)
          (:classes data)))

(defn load-rbm
  "Load a RBM from disk."
  [filepath]
  (edn/read-string {:readers {'deebn.rbm.RBM edn->RBM
                              'deebn.rbm.CRBM edn->CRBM}} (slurp filepath)))
