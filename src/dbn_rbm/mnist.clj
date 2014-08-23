(ns dbn-rbm.mnist
  (:require [clojure.data.csv :as csv]
            [clojure.java.io :as io]
            [clojure.core.matrix :as matrix]
            [clojure.core.matrix.operators]))

(matrix/set-current-implementation :vectorz)

(defn load-csv
  "Load the CSV into memory"
  [filepath]
  (with-open [in-file (io/reader filepath)]
    (doall (csv/read-csv in-file))))

(defn scale-data
  "Scale the input parameters to [0-1]"
  [x]
  (conj (doall (mapv #(/ % 255.0) (rest x))) (first x)))

(defn load-data
  "Load a MNIST CSV data set."
  [filepath]
  (let [input (load-csv filepath)
        m (matrix/matrix input)
        read-data (matrix/emap read-string m)]
    (doall (mapv scale-data read-data))))
