(ns dbn-rbm.mnist
  (:require [dbn-rbm.rbm :as rbm]
            [clojure.data.csv :as csv]
            [clojure.java.io :as io]
            [clojure.core.matrix :as matrix]
            [clojure.core.matrix.operators]))

(defn load-csv
  "Load the CSV into memory"
  [filepath]
  (with-open [in-file (io/reader filepath)]
    (doall (csv/read-csv in-file))))

(defn scale-data
  "Scale the input parameters to [0-1]. This assumes that the label is
  the first element. After scaling, the label is the last element."
  [x]
  (conj (mapv #(/ % 255.0) (rest x)) (first x)))

(defn load-data
  "Load a MNIST CSV data set."
  [filepath]
  (->> filepath
       (load-csv)
       (matrix/emap read-string)
       (mapv scale-data)))

(defn load-data-sans-label
  "Load a MNIST CSV data set without the label"
  [filepath]
  (let [data (load-data filepath)]
    (matrix/select data :all (range 0 784))))

(defn load-data-with-softmax
  "Load a dataset with the class label expanded to a softmax-appropriate form.
  Example: In the MNIST dataset, class '7' expands to -> '0 0 0 0 0 0 0 1 0 0"
  [filepath]
  (let [data (load-data filepath)]
    (doall (mapv #(rbm/softmax-from-obv % 10) data))))
