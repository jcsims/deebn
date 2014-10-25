(ns deebn.mnist
  (:require [clojure.core.matrix :as matrix]
            [clojure.core.matrix.select :as select]
            [clojure.data.csv :as csv]
            [clojure.java.io :as io]
            [deebn.util :refer [softmax-from-obv]]))

(matrix/set-current-implementation :vectorz)

(defn scale-data
  "Scale the input parameters to [0-1]. This assumes that the label is
  the first element. After scaling, the label is the last element."
  [x]
  (conj (mapv #(/ % 255.0) (rest x)) (first x)))

(defn load-data
  "Load a MNIST CSV data set."
  [filepath]
  (let [data (with-open [in-file (io/reader filepath)]
               (->> in-file
                    (csv/read-csv)
                    (matrix/emap read-string)
                    (mapv scale-data)))]
    (matrix/matrix data)))

(defn load-data-sans-label
  "Load a MNIST CSV data set without the label"
  [filepath]
  (let [data (load-data filepath)]
    (matrix/matrix (select/sel data (select/irange) (range 0 784)))))

(defn load-data-with-softmax
  "Load a dataset with the class label expanded to a softmax-appropriate form.
  Example: In the MNIST dataset, class '7' expands to -> '0 0 0 0 0 0 0 1 0 0"
  [filepath]
  (let [data (load-data filepath)
        data (mapv #(softmax-from-obv % 10) (matrix/rows data))]
    (matrix/matrix data)))
