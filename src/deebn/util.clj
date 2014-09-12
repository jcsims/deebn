(ns deebn.util
  (:use [clojure.core.matrix]))

(defn bernoulli
  "Take a single Bernoulli sample, given a probability"
  [p]
  (if (> (rand) p) 0 1))

(defn mean
  "Find the mean (average) value of a vector."
  [x]
  (/ (ereduce + x) (count x)))
