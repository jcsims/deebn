# deebn

A Clojure library implementing a Deep Belief Network using Restricted
Boltzmann Machines, based on [Geoffery Hinton's work](work). This
library is the result of my thesis research into deep learning methods.

## Status

Much of the functionality is present, but the final touches are still
a work in progress. A few items are still outstanding:

- back-fitting the DBN as the final step of training
- optimizing RBM testing (it's unacceptably slow presently)
- visualization

There are a few hyper-parameters available for training:

- learning rate
- initial momentum
- momentum (used after 'momentum-delay' epochs
- momentum-delay
- batch-size
- epochs
- gap-delay (epochs to wait before testing for early stopping)
- gap-stop-delay (consecutive positive energy gap epochs that initiate
  an early stop)

## Usage

The `core` namespace aims to offer examples of using the library. The
`mnist` namespace offers examples for bring in datasets (in this case
the [MNIST](mnist) dataset).

## License

Copyright © 2014 Chris Sims

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.

(work): http://www.cs.toronto.edu/~hinton/absps/montrealTR.pdf
(mnist): http://yann.lecun.com/exdb/mnist/
