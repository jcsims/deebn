(ns deebn.core
  (:require [deebn.dbn :refer [build-dbn]]
            [deebn.dnn :refer [dbn->dnn]]
            [deebn.mnist :refer [load-data-sans-label
                                 load-data-with-softmax load-data]]
            [deebn.protocols :refer [train-model test-model]]
            [deebn.rbm :refer [build-rbm build-jd-rbm]]
            [clojure.core.matrix :refer [set-current-implementation]]))

(set-current-implementation :vectorz)

(comment
  (def m (build-jd-rbm 784 500 10))
  (def dataset (load-data-with-softmax "data/mnist_train.csv"))
  (def m (train-model m dataset {:batch-size 100}))
  (def test-dataset (load-data "data/mnist_test.csv"))
  (test-model m test-dataset)
  (def d (build-dbn [784 500 500 250]))
  (def dbn-data (load-data-sans-label "data/mnist_train.csv"))
  (def d (train-model d dbn-data {:batch-size 100}))
  (def d (dbn->dnn d 10))
  (def dataset (load-data "data/mnist_train.csv"))
  (def d (train-model d dataset {:batch-size 100}))
  (test-model d test-dataset)
  )
