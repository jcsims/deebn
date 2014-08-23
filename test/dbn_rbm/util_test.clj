(ns dbn-rbm.util-test
  (:require [dbn-rbm.util :refer :all]
            [clojure.test :refer :all]
            [clojure.test.check :as tc]
            [clojure.test.check.clojure-test :refer :all]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defn avg-bernoulli
  "Run n iterations of a specific Bernoulli trial vector, and
  return the average value."
  [n p]
  (loop [i 0
         running (vec (take (count p) (repeat 0)))]
    (if (< i n)
      (recur
       (inc i)
       (mapv + running (bernoulli p)))
      (mapv #(/ % (double n)) running))))

(defspec bernoulli-distribution 100
  (prop/for-all [v (gen/such-that 
                    not-empty 
                    (gen/vector 
                     (gen/fmap #(double (/ % Long/MAX_VALUE)) gen/s-pos-int)))]
                (let [a (avg-bernoulli 10000 v)
                      close (map #(< (Math/abs (- % %2)) 0.001) v a)]
                  (every? true? close))))
