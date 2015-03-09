(ns deebn.cv
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.select :as s]
            [deebn.protocols :refer [train-model test-model]]
            [deebn.dnn :refer [dbn->dnn]]))

(defn select-folds
  "Returns a vector of vectors specifying holdout observations to form
  k folds in a dataset."
  [dataset k]
  (let [rows (vec (range (m/row-count dataset)))
        part-size (int (Math/ceil (/ (count rows) k)))]
    (vec (partition part-size part-size nil (shuffle rows)))))

(defn k-fold-cross-validation
  "Perform k-fold cross-validation on a training model. A single data
  set is provided, and a vector of error percentages is
  returned. Since training data and test data are in potentially
  different formats, pass both in for use. These should be two formats
  of the same data set to use for validation."
  [model train-data test-data params k]
  (let [holdouts (select-folds train-data k)]
    (mapv (fn [holdouts]
            (let [train-folds (s/sel train-data (s/exclude holdouts) (s/irange))
                  test-fold (s/sel test-data holdouts (s/irange))
                  m (train-model model train-folds params)]
              (test-model m test-fold)))
          holdouts)))

(defn k-fold-cross-validation-dnn
  "Perform k-fold cross-validation on a training model. A single data
  set is provided, and a vector of error percentages is
  returned. Since training data and test data are in potentially
  different formats, pass both in for use. These should be two formats
  of the same data set to use for validation. DNNs require two
  training steps, and so require a slightly different validation
  approach."
  [model train-data test-data params k classes]
  (let [holdouts (select-folds train-data k)]
    (mapv (fn [holdouts]
            (let [train-folds (m/matrix (s/sel train-data
                                               (s/exclude holdouts) (s/irange)))
                  test-fold (m/matrix (s/sel test-data holdouts (s/irange)))
                  m (train-model model train-folds params)
                  m (dbn->dnn m classes)
                  train-folds (m/matrix (s/sel test-data (s/exclude holdouts)
                                               (s/irange)))
                  m (train-model m train-folds params)]
              (test-model m test-fold)))
          holdouts)))
