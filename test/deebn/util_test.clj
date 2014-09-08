(ns dbn-rbm.util-test
  (:require [dbn-rbm.util :refer :all]
            [clojure.test :refer :all]
            [clojure.test.check :as tc]
            [clojure.test.check.clojure-test :refer :all]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defn avg-bernoulli
  "Run n iterations of a specific Bernoulli trial, and
  return the average value."
  [n p]
  (loop [i 0
         running 0]
    (if (< i n)
      (recur (inc i) (+ running (bernoulli p)))
      (/ running (double n)))))

(defspec bernoulli-distribution 10000
  (prop/for-all [p (gen/fmap #(double (/ % Long/MAX_VALUE)) gen/s-pos-int)]
                (let [a (avg-bernoulli 10000 p)]
                  (< (Math/abs (- a p)) 0.001))))
