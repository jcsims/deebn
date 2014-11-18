# deebn

A Clojure library implementing a Deep Belief Network using Restricted
Boltzmann Machines, based on [Geoffery Hinton's work][work]. This
library is the result of my thesis research into deep learning methods.

## "Installation"
`deebn` is available for download or usage through your favorite dependency management tool from
Clojars: 

[![Clojars Project](http://clojars.org/deebn/latest-version.svg)](http://clojars.org/deebn)

## Capabilities

There are a few types of model that you can build and train, either
for classification or as components of other models:

- Restricted Boltzmann Machine
  - can be used as a component of a Deep Belief Network, or as a standalone discriminatory classifer  
  Hyper-parameters:
    - learning rate
    - initial momentum
    - momentum (used after 'momentum-delay' epochs)
    - momentum-delay
    - batch-size
    - epochs
    - gap-delay (epochs to wait before testing for early stopping)
    - gap-stop-delay (consecutive positive energy gap epochs that initiate
      an early stop)
- Deep Belief Network (composed of layers of RBMs)
  - Can be used to pre-train a Deep Neural Network, or as a discriminatory classifier
    (Note: a classification DBN is not fine-tuned - performance is sastifactory but not optimal)  
  Hyper-parameters:
    - whether to use activations rather than samples from hidden layers when propagating
      to the next layer
- Deep Neural Network
  - Initialized from a pre-trained DBN, with an additional logistic regression layer added
  - Network output is a softmax unit
  - Logistic regression unit is pre-trained with output from the DBN before moving to a 
    full backprop training regimen  
  Hyper-parameters:
    - batch-size
    - epochs
    - learning rate
    - lambda - L2 regularization (weight decay) parameter

## Usage

The `core` namespace aims to offer examples of using the library. The
`mnist` namespace offers examples for bringing in datasets (in this case
the [MNIST][mnist] dataset).

## License

Copyright © 2014 Chris Sims

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.

[work]: http://www.cs.toronto.edu/~hinton/absps/montrealTR.pdf
[mnist]: http://yann.lecun.com/exdb/mnist/
