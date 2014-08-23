(ns dbn-rbm.rbm
  (:require [clojure.core.matrix :as matrix]))

(defrecord RBM [w hbias vbias visible hidden]) 

(defn rand-vec 
  "Create a n-length vector of random real numbers in range [-range/2 range/2]"
  [n range]
  (vector (take n (repeatedly #(- (rand range) (/ range 2))))))

(defn build-rbm
  "Factory function to produce an RBM record."
  [visible hidden]
  (matrix/set-current-implementation :vectorz)
  (let [w (matrix/matrix (vec (take visible 
                                    (repeatedly #(rand-vec hidden 0.01)))))
        hbias (matrix/zero-vector hidden)
        vbias (matrix/zero-vector visible)]
    (->RBM w hbias vbias visible hidden)))

(defn sigmoid
  "Sigmoid function, used as an activation function for nodes in a
  network."
  [x]
  (/ (+ 1 (Math/exp (* -1 x)))))
