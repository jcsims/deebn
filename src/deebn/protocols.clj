(ns deebn.protocols)

(defprotocol Trainable
  "Protocol for models that are trainable with a dataset."
  (train-model [m dataset params]
    "Train a model, given a dataset and relevant
    hyper-parameters. Refer to individual training functions for
    hyper-parameter details."))

(defprotocol Testable
  "Protocol for models that are testable."
  (test-model [m dataset]
    "Test a trained model given a dataset. The dataset may need to be
    in a different format from that used to train."))

(defprotocol Classify
  "Protocol for models that can classify a given observation."
  (classify [m obv]
    "Use the given model to classify a single observation."))
