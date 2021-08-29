from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params['b1']=np.zeros(hidden_dim)
        self.params['b2']=np.zeros(num_classes)
        self.params['W1'] = np.array([[np.random.normal(scale=weight_scale) for i in range(hidden_dim)] for j in range(input_dim)])
        self.params['W2'] = np.array([[np.random.normal(scale=weight_scale) for i in range(num_classes)] for j in range(hidden_dim)])
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        layer_1_out, _ = affine_forward(X, self.params['W1'], self.params['b1'])
        layer_1_nonlinear, _ = relu_forward(layer_1_out)
        scores, _ = affine_forward(layer_1_nonlinear, self.params['W2'], self.params['b2'])
        # scores = softmax_forward(layer_2_out)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        loss, grads = 0, {}

        if y is None:
            return scores
        else:
            loss, softmax_grads = softmax_loss(scores, y)

        loss_regularized = loss + 0.5*self.reg*(np.linalg.norm(self.params['W1'])**2+np.linalg.norm(self.params['W2'])**2)

        #print(softmax_grads.shape)
        #print(layer_1_nonlinear.shape)
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        hidden_grad, grads['W2'], grads['b2'] = affine_backward(softmax_grads, (layer_1_nonlinear, self.params['W2'], self.params['b2'], self.reg))
        hidden_grad_relu = relu_backward(hidden_grad, layer_1_out)
        #print('Shape of hidden grad_relu ', hidden_grad_relu.shape)
        #print('Shape of X ',X.shape)
        grad_input, grads['W1'], grads['b1'] = affine_backward(hidden_grad_relu, (X, self.params['W1'], self.params['b1'], self.reg))


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss_regularized, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for randomconnected
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        current_layer_index = 1
        layer_input = input_dim
        for h in hidden_dims:
            self.params['b'+str(current_layer_index)] = np.zeros(h)
            self.params['W'+str(current_layer_index)] = np.array([[np.random.normal(scale=weight_scale) for i in range(h)] for j in range(layer_input)])
            self.params['gamma'+str(current_layer_index)] = np.ones(h)
            self.params['beta'+str(current_layer_index)] = np.zeros(h)
            layer_input = h
            current_layer_index+=1
        
        self.params['b'+str(current_layer_index)] = np.zeros(num_classes)
        self.params['W'+str(current_layer_index)] = np.array([[np.random.normal(scale=weight_scale) for i in range(num_classes)] for j in range(layer_input)])

        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        # for k, v in self.params.items():
        #     self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        layer_input = X
        layer_nonlinear_dict = {}
        layer_dict = {}
        layer_droput_dict = {}

        for layer in range(1, self.num_layers):
            if self.normalization is not None:
                layer_out, cache = affine_norm_relu_forward(layer_input, self.params['W'+str(layer)], self.params['b'+str(layer)], 
                                                            self.normalization, self.params['gamma'+str(layer)], self.params['beta'+str(layer)], self.bn_params[layer-1])
                fc_cache, norm_cache, relu_cache = cache
                fc_cache = (*fc_cache, self.reg)
                cache = (fc_cache, norm_cache, relu_cache)
                layer_dict[layer] = cache

            else:    
                layer_out, cache = affine_relu_forward(layer_input, self.params['W'+str(layer)], self.params['b'+str(layer)])
                fc_cache, relu_cache = cache
                fc_cache = (*fc_cache, self.reg)
                cache = (fc_cache, relu_cache)
                layer_dict[layer] = cache
            if self.use_dropout:
                layer_dropout, cache = dropout_forward(layer_out, self.dropout_param)
                
                layer_droput_dict[layer] = cache
                layer_input = layer_dropout
            else:
                layer_input = layer_out

        scores, _ = affine_forward(layer_input, self.params['W'+str(self.num_layers)], self.params['b'+str(self.num_layers)])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        loss, softmax_grads = softmax_loss(scores, y)
        loss_regularized = loss + 0.5*self.reg*(np.sum([np.linalg.norm(self.params['W'+str(layer)])**2 for layer in range(1, self.num_layers+1)]))
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        hidden_grad = softmax_grads
        #print(hidden_grad.shape)
        hidden_grad, grads['W'+str(self.num_layers)], grads['b'+str(self.num_layers)] = affine_backward(hidden_grad, (layer_input, self.params['W'+str(self.num_layers)], self.params['b'+str(self.num_layers)], self.reg))
        #print(hidden_grad.shape)
        for layer in range(self.num_layers, 1, -1):
            if self.use_dropout:
                hidden_grad = dropout_backward(hidden_grad, layer_droput_dict[layer-1])
            if self.normalization is not None:
                hidden_grad, grads['W'+str(layer-1)], grads['b'+str(layer-1)], grads['gamma'+str(layer-1)], grads['beta'+str(layer-1)] = affine_norm_relu_backward(hidden_grad, self.normalization, layer_dict[layer-1])
            else:
                hidden_grad, grads['W'+str(layer-1)], grads['b'+str(layer-1)] = affine_relu_backward(hidden_grad, layer_dict[layer-1])
            #print(hidden_grad.shape)
            
            # hidden_grad = hidden_grad_input
            # hidden_grad_non_linear = relu_backward(hidden_grad, layer_dict[layer-1])

        # grad_input, grads['W1'], grads['b1'] = affine_backward(hidden_grad_non_linear, (X, self.params['W1'], self.params['b1'], self.reg))
        
        #print('Shape of hidden grad_relu ', hidden_grad_relu.shape)
        #print('Shape of X ',X.shape)
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss_regularized, grads


def affine_norm_relu_forward(x, w, b, normalization, gamma, beta, bn_param):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    # print(x.shape, w.shape, b.shape)
    a, fc_cache = affine_forward(x, w, b)
    if normalization=='batchnorm':
        norm_out, norm_cache = batchnorm_forward(a, gamma, beta, bn_param)
    elif normalization=='layernorm':
        norm_out, norm_cache = layernorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(norm_out)
    cache = (fc_cache, norm_cache, relu_cache)
    return out, cache


def affine_norm_relu_backward(dout, normalization, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, norm_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    if normalization == 'batchnorm':
        dx, dgamma, dbeta = batchnorm_backward_alt(da, norm_cache)
    if normalization == 'layernorm':
        dx, dgamma, dbeta = layernorm_backward(da, norm_cache)
    dx, dw, db = affine_backward(dx, fc_cache)
    return dx, dw, db, dgamma, dbeta