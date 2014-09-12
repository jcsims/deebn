(ns deebn.protocols)

(defprotocol Trainable
  "Protocol for models that are trainable with a dataset."
  (trainm [m dataset params]
    "Train a model, given a dataset and relevant
    hyper-parameters. Refer to individual training functions for
    hyper-parameter details."))

(defprotocol Testable
  "Protocol for models that are testable."
  (testm [m dataset]
    "Test a trained model given a dataset."))
