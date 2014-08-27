(ns dbn-rbm.util)

(defn bernoulli
  "Take a single Bernoulli sample, given a probability"
  [p]
  (if (> (rand) p) 0 1))
