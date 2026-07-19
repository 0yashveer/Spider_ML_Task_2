## Why Residual connections help optimization
   Residual Connection helps the backpropogating by providing the network with non zero gradient which helps in avoiding the problem of vanishing gradients.

## How skip connections impove the gradient flow
   the skip connections provide a new path for the flow of gradients in backpropogation ( 1 + dL/dw ) here even if the dL/dw is unstable the plus 1 term prevents the
   unstable flow.

## Effect of Network Depth on performance
   The deepness of network increases the probability of identifying more complex pattern in a provided image , on the other hand it also increases the computation       required to process an image significantely.

## Challenges Encountered
