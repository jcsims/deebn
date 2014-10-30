(ns deebn.core
  (:require [deebn.dbn :refer [build-dbn build-classify-dbn]]
            [deebn.dnn :refer [dbn->dnn]]
            [deebn.mnist :refer [load-data-sans-label
                                 load-data-with-softmax load-data]]
            [deebn.protocols :refer [train-model test-model]]
            [deebn.rbm :refer [build-rbm build-jd-rbm]]
            [clojure.core.matrix :refer [set-current-implementation]]))

(set-current-implementation :vectorz)

(comment
  ;;; Choose a model and the corresponding dataset
  ;; A single Restricted Boltzmann Machine used for classification
  (def m (build-jd-rbm 784 500 10))
  (def dataset (load-data-with-softmax "data/mnist_train.csv"))
  ;; A classification Deep Belief Network
  (def m (build-classify-dbn [784 500 500 2000] 10))
  (def dataset (load-data-with-softmax "data/mnist_train.csv"))
  ;; A Deep Neural Network backed by a pre-trained Deep Belief Network
  (def m (build-dbn [784 500 500 250]))
  (def dataset (load-data-sans-label "data/mnist_train.csv"))

  ;;; Train the model
  (def m (train-model m dataset {:batch-size 100}))
  ;; For a DNN, the DBN is converted to a DNN before fine-tuning
  (def m (dbn->dnn m 10))
  (def dataset (load-data "data/mnist_train.csv"))
  (def m (train-model m dataset {:batch-size 100 :epochs 10}))

  ;;; Test the model
  (def test-dataset (load-data "data/mnist_test.csv"))
  (test-model m test-dataset)
  )
