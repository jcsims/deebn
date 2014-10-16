(ns deebn.rbm
  (:refer-clojure :exclude [+ - * / ==])
  (:require [deebn.protocols :refer [Testable Trainable]]
            [deebn.util :refer [mean bernoulli]]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :refer [+ - * / ==]]
            [clojure.core.matrix.select :as s]
            [clojure.set :refer [difference]]
            [clojure.tools.reader.edn :as edn]
            [taoensso.timbre.profiling :as prof])
  (:import java.io.Writer
           mikera.vectorz.Scalar))

(m/set-current-implementation :vectorz)

;;;===========================================================================
;;; Generate Restricted Boltzmann Machines
;;; ==========================================================================
;; We define a purely generative RBM, trained without any class
;; labels, and a classification RBM
(defrecord RBM [w vbias hbias w-vel vbias-vel hbias-vel visible hidden])
(defrecord CRBM [w vbias hbias w-vel vbias-vel hbias-vel visible hidden classes])

;; FIXME: This should really be Gaussian instead of uniform.
(defn rand-vec
  "Create a n-length vector of random real numbers in range [-range/2 range/2]"
  [n range]
  (repeatedly n #(- (rand range) (/ range 2))))

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
        squared-error (m/ereduce + (m/emap #(* % %) (- batch v)))
        w-vel (+ (* momentum (:w-vel rbm)) (* learning-rate delta-w))
        vbias-vel (+ (* momentum (:vbias-vel rbm))
                     (* learning-rate delta-vbias))
        hbias-vel (+ (* momentum (:hbias-vel rbm))
                     (* learning-rate delta-hbias))]
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
         batch (m/matrix (s/sel dataset (range 0 batch-size) (s/irange)))
         batch-num 1]
    (let [start (* (dec batch-num) batch-size)
          end (min (* batch-num batch-size) (m/row-count dataset))]
      (if (>= start (m/row-count dataset))
        rbm
        (do
          (print "Training batch:" batch-num)
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
  (let [avg-train-energy (mean (pmap #(free-energy %1 rbm)
                                     (m/rows train-sample)))
        avg-validation-energy (mean (pmap #(free-energy %1 rbm)
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
        (do (println "Training epoch" epoch)
            (let [curr-momentum (if (> epoch momentum-delay)
                                  momentum initial-momentum)
                  rbm (train-epoch rbm dataset learning-rate
                                   curr-momentum batch-size)
                  gap-after-train (check-overfitting rbm train-sample
                                                     validations)
                  _ (println "Gap pre-train:" energy-gap
                             "After train:" gap-after-train)]
              (if (and (> epoch gap-delay)
                       (neg? (- energy-gap gap-after-train))
                       (> gap-inc-count gap-stop-delay))
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
  "Get the position of the minimum element of a collection."
  [x]
  (if (not (empty? x))
    (let [least (m/emin x)
          indexed (zipmap (map #(.get ^Scalar %) x) (range (count x)))]
      (get indexed least))))

(defn get-prediction
  "For a given observation and RBM, return the predicted class."
  [x rbm num-classes]
  (let [softmax-cases  (mapv #(gen-softmax % num-classes) (range num-classes))
        trials  (m/matrix (mapv #(m/join % %2) softmax-cases (repeat (butlast x))))
        results (mapv #(free-energy % rbm) trials)]
    (get-min-position results)))

(defn test-rbm
  "Test a joint density RBM trained on a data set. Returns an error
  percentage.

  dataset should have the label as the last entry in each
  observation."
  [rbm dataset num-classes]
  (let [num-observations (m/row-count dataset)
        predictions (pmap #(get-prediction % rbm num-classes) dataset)
        errors (mapv #(if (== (last %) %2) 0 1) dataset predictions)
        total (reduce + errors)]
    (double (/ total num-observations))))

(extend-protocol Testable
  CRBM
  (test-model [m dataset]
    (test-rbm m dataset (:classes m))))


;;;===========================================================================
;;; Utility functions for an RBM
;;;===========================================================================

(defn query-hidden
  "Given an RBM and an input vector, query the RBM for the state of
  the hidden nodes."
  [rbm x mean-field?]
  (let [pre-sample (m/emap sigmoid (+ (:hbias rbm) (m/mmul x (:w rbm))))]
    (if mean-field? pre-sample
        (map bernoulli pre-sample))))

;; This is designed for EDN printing, not actually visualizing the RBM
;; at the REPL (this is only needed because similar methods are not
;; defined for clojure.core.matrix implementations)
(defmethod clojure.core/print-method RBM print-RBM [rbm ^Writer w]
  (.write w (str "#deebn.rbm.RBM {"
                 " :w " (:w rbm)
                 " :vbias " (:vbias rbm)
                 " :hbias " (:hbias rbm)
                 " :w-vel " (:w-vel rbm)
                 " :vbias-vel " (:vbias-vel rbm)
                 " :hbias-vel " (:hbias-vel rbm)
                 " :visible " (:visible rbm)
                 " :hidden " (:hidden rbm)
                 " }")))

(defmethod clojure.core/print-method CRBM print-CRBM [rbm ^Writer w]
  (.write w (str "#deebn.rbm.CRBM {"
                 " :w " (:w rbm)
                 " :vbias " (:vbias rbm)
                 " :hbias " (:hbias rbm)
                 " :w-vel " (:w-vel rbm)
                 " :vbias-vel " (:vbias-vel rbm)
                 " :hbias-vel " (:hbias-vel rbm)
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
