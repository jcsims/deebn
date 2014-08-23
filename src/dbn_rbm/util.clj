(ns dbn-rbm.util)

(defn bernoulli
  "Take a vector of probabilities, and map a single Bernoulli sample
  over each element"
  ([] [])
  ([p]
     (mapv #(if (> (rand) %) 0 1) p)))
