import numpy as np

def im2col(X, k_height, k_width, padding=1, stride=1):
    '''
    Construct the im2col matrix of intput feature map X.
    X: 4D tensor of shape [N, C, H, W], input feature map
    k_height, k_width: height and width of convolution kernel
    return a 2D array of shape (C*k_height*k_width, H*W*N)
    The axes ordering need to be (C, k_height, k_width, H, W, N) here, while in
    reality it can be other ways if it weren't for autograding tests.
    '''
    N, C, H, W = X.shape 

    output_height = int((H + 2 * padding - k_height) / stride + 1)
    output_width = int((W + 2 * padding - k_width) / stride + 1)
    padX = np.pad(X, ((0,0), (0,0), (padding, padding), (padding, padding)), 'constant')

    patches = []

    for i in range(output_height):
        for j in range(output_width):
            patch_height_start = i
            patch_height_end = i + k_height
            patch_width_start = j 
            patch_width_end = j + k_width 

            patch = padX[:, :, patch_height_start:patch_height_end, patch_width_start:patch_width_end]
            patches.append(patch.reshape(N, -1))

    return np.concatenate(patches, axis=0).T

def im2col_bw(grad_X_col, X_shape, k_height, k_width, padding=1, stride=1):
    '''
    Map gradient w.r.t. im2col output back to the feature map.
    grad_X_col: a 2D array
    return X_grad as a 4D array in X_shape
    '''
    N, C, H, W = X_shape

    # print(grad_X_col.T)
    grad_X_col = grad_X_col.T

    padH = H + 2*padding
    padW = W + 2*padding
    X_grad = np.zeros((N, C, padH, padW))

    output_height = int((H + 2 * padding - k_height) / stride + 1)
    output_width = int((W + 2 * padding - k_width) / stride + 1)

    for i in range(output_height):
        for j in range(output_width):
            patch_start = i*output_height*N + j*N
            patch_end = patch_start + N
            patch = grad_X_col[patch_start:patch_end, :]

            im_height_start = i
            im_height_end = i+k_height
            im_width_start = j
            im_width_end = j+k_width

            X_grad[:, :, im_height_start:im_height_end, im_width_start:im_width_end] += patch.reshape(N, C, k_height, k_width)

    # Remove padding
    X_grad = X_grad[:, :, padding:(H + 2*padding - padding), padding:(W + 2*padding - padding)]

    return X_grad
    
    
class Transform:
    """
    This is the base class. You do not need to change anything.
    Read the comments in this class carefully.
    """
    def __init__(self):
        """
        Initialize any parameters
        """
        pass

    def forward(self, x):
        """
        x should be passed as column vectors
        """
        pass

    def backward(self, grad_wrt_out):
        """
        Unlike Problem 1 MLP, here we no longer accumulate the gradient values,
        we assign new gradients directly. This means we should call update()
        every time we do forward and backward, which is fine. Consequently, in
        Problem 2 zerograd() is not needed any more.
        Compute and save the gradients wrt the parameters for update()
        Read comments in each class to see what to return.
        """
        pass

    def update(self, learning_rate, momentum_coeff):
        """
        Apply gradients to update the parameters
        """
        pass


class ReLU(Transform):
    """
    Implement this class
    """
    def forward(self, x, train=True):
        """
        returns ReLU(x)
        """
        self.x = x
        return np.maximum(x, 0)

    def backward(self, dLoss_dout):
        """
        dLoss_dout is the gradients wrt the output of ReLU
        returns gradients wrt the input to ReLU
        """
        return dLoss_dout * (self.x > 0)

      
class Flatten(Transform):
    """
    Implement this class
    """
    def forward(self, x):
        """
        returns Flatten(x)
        """
        # print("x", x.shape)
        self.flattened = x.reshape(x.shape[0], -1)
        return self.flattened

    def backward(self, dloss):
        """
        dLoss is the gradients wrt the output of Flatten
        returns gradients wrt the input to Flatten
        """
        # is this how flatten backwards work?
        return dloss * self.flattened


class Conv(Transform):
    """
    Implement this class - Convolution Layer
    """
    def __init__(self, input_shape, filter_shape, rand_seed=0):
        """
        input_shape is a tuple: (channels, height, width)
        filter_shape is a tuple: (num of filters, filter height, filter width)
        weights shape (number of filters, number of input channels, filter height, filter width)
        Use Xavier initialization for weights, as instructed on handout
        Initialze biases as an array of zeros in shape of (num of filters, 1)
        """
        np.random.seed(rand_seed) # keep this line for autograding; you may remove it for training
        self.C, self.H, self.Width = input_shape
        self.num_filters, self.k_height, self.k_width = filter_shape
        b = np.sqrt(6) / np.sqrt((self.C + self.num_filters) * self.k_height * self.k_width)
        self.W = np.random.uniform(-b, b, (self.num_filters, self.C, self.k_height, self.k_width))
        self.b = np.zeros((self.num_filters, 1))

        self.w_gradient = np.zeros(self.W.shape)
        self.b_gradient = np.zeros(self.b.shape)
        self.w_update = 0.0
        self.b_update = 0.0

    def forward(self, inputs, stride=1, pad=2):
        """
        Forward pass of convolution between input and filters
        inputs is in the shape of (batch_size, num of channels, height, width)
        Return the output of convolution operation in shape (batch_size, num of filters, height, width)
        use im2col here
        """

        # Forward: input (N, C, H, W) -> im2col -> col (C x K x K, H_out x W_out x N) -> Linear forward -> linear out (C_out, H_out x W_out x N) -> reshape/transpose -> output (N, C_out, H_out, W_out)

        self.inputs = inputs
        n, c, h, w = inputs.shape
        self.batch_size = n
        
        X2col = im2col(inputs, self.k_height, self.k_width, padding=pad, stride=stride)
        self.pad = pad
        self.stride = stride
        Wcol = self.W.reshape(X2col.shape[0], -1)

        out = np.dot(Wcol.T, X2col) + self.b    
        out = out.reshape(self.num_filters, h, w, n)

        return out.transpose(3,0,1,2)

    def backward(self, dloss):
        """
        Read Transform.backward()'s docstring in this file
        dloss shape (batch_size, num of filters, output height, output width)
        Return [gradient wrt weights, gradient wrt biases, gradient wrt input to this layer]
        """
        # Backward: 
        # grad_wrt_out (N, C_out, H_out, W_out) -> reshape/transpose -> grad_wrt_out2 (C_out, H_out x W_out x N) 
        #   -> Linear backward -> grad_col (C x K x K, H_out x W_out x N) -> im2col_bw -> grad_wrt_input (N, C, H, W)
        
        X2col = im2col(self.inputs, self.k_height, self.k_width, padding=self.pad, stride=self.stride)

        batch_size, num_filters, output_height, output_width = dloss.shape 

        dloss2 = dloss.transpose(1,2,3,0).reshape(num_filters, -1)

        self.w_gradient = np.dot(dloss2, X2col.T).reshape(self.W.shape)
        self.b_gradient = np.sum(dloss, axis=(0,2,3)).reshape(num_filters, -1)

        X_gradient = np.dot(self.W.reshape(num_filters, -1).T, dloss2)
        self.X_gradient = im2col_bw(X_gradient, self.inputs.shape,
                        self.k_height, self.k_width, self.pad, self.stride)

        return [self.w_gradient, self.b_gradient, self.X_gradient]

    def update(self, learning_rate=0.001, momentum_coeff=0.5):
        """
        Update weights and biases with gradients calculated by backward()
        Use the same momentum formula as Problem1
        Here we divide gradients by batch_size (because we will be using sum Loss
        instead of mean Loss in Problem 2 during backpropogation). Do not divide
        gradients by batch_size in step() in Problem 1.
        """
        self.w_update = momentum_coeff * self.w_update + self.w_gradient / self.batch_size
        self.W = self.W - learning_rate * self.w_update

        self.b_update = momentum_coeff * self.b_update + self.b_gradient / self.batch_size
        self.b = self.b - learning_rate * self.b_update

    def get_wb_conv(self):
        """
        Return weights and biases
        """
        return self.W, self.b


class MaxPool(Transform):

    def __init__(self, filter_shape, stride):
        self.filter_shape = filter_shape
        self.filter_height, self.filter_width = filter_shape
        self.stride = stride

    def forward(self, x):
        N, C, H, W = x.shape
        stride = self.stride

        out_height = (H - self.filter_height) // stride + 1
        out_width = (W - self.filter_width) // stride + 1

        x_split = x.reshape(N * C, 1, H, W)
        x_cols = im2col(x_split, self.filter_height, self.filter_width, padding=0, stride=stride)
        x_cols_max_index = np.argmax(x_cols, axis=0)
        x_cols_max = x_cols[x_cols_max_index, np.arange(x_cols.shape[1])]
        out = x_cols_max.reshape(out_height, out_width, N, C).transpose(2, 3, 0, 1)

        self.x_cols = x_cols
        self.x_cols_max_index = x_cols_max_index
        self.x = x
        return out

    def backward(self, dout):
        x, x_cols, x_cols_max_index = self.x, self.x_cols, self.x_cols_max_index
        N, C, H, W = x.shape
        self.filter_height, self.filter_width = 2, 2
        stride = 2

        dout_reshaped = dout.transpose(2, 3, 0, 1).flatten()
        dx_cols = np.zeros_like(x_cols)
        dx_cols[x_cols_max_index, np.arange(dx_cols.shape[1])] = dout_reshaped
        dx = im2col_bw(dx_cols, (N * C, 1, H, W), self.filter_height, self.filter_width,
                    padding=0, stride=stride)
        dx = dx.reshape(x.shape)

        return dx


class MaxPoolNaive(Transform):
    """
    Implement this class - MaxPool layer
    """
    def __init__(self, filter_shape, stride):
        """
        filter_shape is (filter_height, filter_width)
        stride is a scalar
        """
        self.filter_shape = filter_shape
        self.filter_height, self.filter_width = filter_shape
        self.stride = stride
        self.max_indices = None
        

    def forward(self, inputs):
        self.inputs = inputs # save the inputs
        N, C, H, W = inputs.shape 
        # self.max_indices = np.zeros(inputs.shape)
        
        # print(self.filter_shape)
        output_height = int((H - self.filter_height)/self.stride) + 1
        output_width = int((W - self.filter_width)/self.stride) + 1
        # print(output_height, output_width)

        forward_out = np.zeros((N, C, output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):
                pool_height_start = i*self.stride
                pool_height_end = i*self.stride + self.filter_height
                pool_width_start = j*self.stride
                pool_width_end = j*self.stride + self.filter_width

                inp = inputs[:, :, pool_height_start:pool_height_end, pool_width_start:pool_width_end].reshape(N, C, -1)
                forward_out[:, :, i, j] = np.max(inp, axis=2)

        self.forward_out = forward_out

        return forward_out

    def backward(self, dloss):
        """
        dloss is the gradients wrt the output of forward()
        """
        N, C, H, W = self.inputs.shape
        backward_out = np.zeros(self.inputs.shape)
        h_strides = int(H / self.stride)
        w_strides = int(W / self.stride)

        for n in range(N):
            for c in range(C):
                for i in range(h_strides):
                    for j in range(w_strides):

                        pool_height_start = i*self.stride
                        pool_height_end = i*self.stride + self.filter_height
                        pool_width_start = j*self.stride
                        pool_width_end = j*self.stride + self.filter_width
                        inp = self.inputs[:, :, pool_height_start:pool_height_end, pool_width_start:pool_width_end]

                        inp = inp.reshape(N, C, -1)
                        max_index = np.argmax(inp, axis=2)

                        w_idx = int(((max_index % self.filter_width) + pool_width_start)[n,c])
                        h_idx = int(((max_index / self.filter_width) + pool_height_start)[n,c])

                        backward_out[n, c, h_idx, w_idx] = dloss[n, c, i, j]

        return backward_out.reshape(N, C, H, W)
    
    
class LinearLayer(Transform):
    """
    Implement this class - Linear layer
    """
    def __init__(self, indim, outdim, rand_seed=0):
        """
        indim, outdim: input and output dimensions
        weights shape (indim,outdim)
        Use Xavier initialization for weights, as instructed on handout
        Initialze biases as an array of ones in shape of (outdim,1)
        """
        np.random.seed(rand_seed) # keep this line for autograding; you may remove it for training
        b = np.sqrt(6) / np.sqrt(indim + outdim)
        self.W = np.random.uniform(-b, b, (indim, outdim))
        self.b = np.zeros((outdim, 1))

        self.w_gradient = np.zeros(self.W.shape)
        self.b_gradient = np.zeros((outdim, 1))
        self.w_update = 0.0
        self.b_update = 0.0

        self.indim = indim 
        self.outdim = outdim

    def forward(self, inputs):
        """
        Forward pass of linear layer
        inputs shape (batch_size, indim)
        """
        self.inputs = inputs
        self.batch_size, self.indim = inputs.shape 
        self.forward_out = np.dot(inputs, self.W).T + self.b 
        return self.forward_out.T

    def backward(self, dloss):
        """
        Read Transform.backward()'s docstring in this file
        dloss shape (batch_size, outdim)
        Return [gradient wrt weights, gradient wrt biases, gradient wrt input to this layer]
        """
        self.w_gradient = np.matmul(self.inputs.T, dloss)
        self.b_gradient = np.sum(dloss, axis=0).reshape(self.b.shape)
        self.backward_out = np.dot(self.W, dloss.T).T

        return self.w_gradient, self.b_gradient, self.backward_out

    def update(self, learning_rate=0.001, momentum_coeff=0.5):
        """
        Similar to Conv.update()
        """
        self.w_update = momentum_coeff * self.w_update + self.w_gradient / self.batch_size
        self.W = self.W - learning_rate * self.w_update

        self.b_update = momentum_coeff * self.b_update + self.b_gradient / self.batch_size
        self.b = self.b - learning_rate * self.b_update

    def get_wb_fc(self):
        """
        Return weights and biases
        """
        return self.W, self.b


class SoftMaxCrossEntropyLoss():
    """
    Implement this class
    """
    def forward(self, logits, labels, get_predictions=False):
        """
        logits are pre-softmax scores, labels are true labels of given inputs
        labels are one-hot encoded
        logits and labels are in the shape of (batch_size, num_classes)
        returns loss as scalar
        (your loss should just be a sum of a batch, don't use mean)
        """
        self.batch_size, _ = logits.shape 
        self.logits = logits 
        self.labels = labels 

        softmax = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        loss = -1 * np.sum((labels * np.log(softmax)), axis=1)
        self.derivative = (softmax - labels)

        if get_predictions:
            # print(logits)
            return np.sum(loss), np.argmax(logits, axis=1)

        return np.sum(loss)


    def backward(self):
        """
        return shape (batch_size, num_classes)
        (don't divide by batch_size here in order to pass autograding)
        """
        return self.derivative

    def getAccu(self):
        """
        Implement as you wish, not autograded.
        """
        pass


class ConvNet:
    """
    Class to implement forward and backward pass of the following network -
    Conv -> Relu -> MaxPool -> Linear -> Softmax
    For the above network run forward, backward and update
    """
    def __init__(self):
        """
        Initialize Conv, ReLU, MaxPool, LinearLayer, SoftMaxCrossEntropy objects
        Conv of input shape 3x32x32 with filter size of 1x5x5 (or 5x5x5)
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then initialize linear layer with output 10 neurons
        Initialize SotMaxCrossEntropy object
        """

        # Parameters
        input_shape = (3, 32, 32)
        filter_shape = (1, 5, 5)
        pool_size = (2,2)
        stride = 2
        outdim = 10
        indim = 1*16*16

        # Conv
        self.conv = Conv(input_shape=input_shape, filter_shape=filter_shape)
        # ReLU
        self.relu = ReLU()
        # MaxPool
        self.maxpool = MaxPool(filter_shape=pool_size, stride=stride)
        # Flatten
        self.flatten = Flatten()
        # LinearLayer
        self.linear = LinearLayer(indim=indim, outdim=outdim)
        # SoftMaxCrossEntropy
        self.loss = SoftMaxCrossEntropyLoss()

    def forward(self, inputs, y_labels):
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape batch x channels x height x width
        2. labels => True labels

        Return loss and predicted labels after one forward pass
        """
        self.inputs = inputs
        self.batch_size, self.channels, self.height, self.width = inputs.shape
        # print("y", y_labels)
        
        conv_out = self.conv.forward(inputs)  # use defult stride and pad?
        # print("conv_out", conv_out.shape)
        relu_out = self.relu.forward(conv_out)
        # print("relu_out", relu_out.shape)
        maxpool_out = self.maxpool.forward(relu_out)
        # print("pool_out", pool_out.shape)
        flatten_out = self.flatten.forward(maxpool_out)
        # print("flatten_out", flatten_out.shape)
        linear_out = self.linear.forward(flatten_out)
        # print("linear_out", linear_out.shape)
        loss, predictions = self.loss.forward(linear_out, y_labels, get_predictions=True)

        return loss, predictions

    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        loss_grad = self.loss.backward()
        lin_w_grad, lin_b_grad, lin_x_grad = self.linear.backward(loss_grad)
        lin_x_grad = lin_x_grad.reshape(self.batch_size, 1, 16, 16)
        maxpool_grad = self.maxpool.backward(lin_x_grad)
        relu_grad = self.relu.backward(maxpool_grad)
        conv_w_grad, conv_b_grad, conv_x_grad = self.conv.backward(relu_grad)

    def update(self, learning_rate, momentum_coeff):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        self.conv.update(learning_rate=learning_rate, momentum_coeff=momentum_coeff)
        self.linear.update(learning_rate=learning_rate, momentum_coeff=momentum_coeff)
 

class ConvNet5x5:
    """
    Class to implement forward and backward pass of the following network -
    Conv -> Relu -> MaxPool -> Linear -> Softmax
    For the above network run forward, backward and update
    """
    def __init__(self):
        """
        Initialize Conv, ReLU, MaxPool, LinearLayer, SoftMaxCrossEntropy objects
        Conv of input shape 3x32x32 with filter size of 1x5x5 (or 5x5x5)
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then initialize linear layer with output 10 neurons
        Initialize SotMaxCrossEntropy object
        """

        # Parameters
        input_shape = (3, 32, 32)
        filter_shape = (5, 5, 5)
        pool_size = (2,2)
        stride = 2
        outdim = 10
        indim = 5*16*16

        # Conv
        self.conv = Conv(input_shape=input_shape, filter_shape=filter_shape)
        # ReLU
        self.relu = ReLU()
        # MaxPool
        self.maxpool = MaxPool(filter_shape=pool_size, stride=stride)
        # Flatten
        self.flatten = Flatten()
        # LinearLayer
        self.linear = LinearLayer(indim=indim, outdim=outdim)
        # SoftMaxCrossEntropy
        self.loss = SoftMaxCrossEntropyLoss()

    def forward(self, inputs, y_labels):
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape batch x channels x height x width
        2. labels => True labels

        Return loss and predicted labels after one forward pass
        """
        self.inputs = inputs
        self.batch_size, self.channels, self.height, self.width = inputs.shape
        # print("y", y_labels)
        
        conv_out = self.conv.forward(inputs)  # use defult stride and pad?
        # print("conv_out", conv_out.shape)
        relu_out = self.relu.forward(conv_out)
        # print("relu_out", relu_out.shape)
        maxpool_out = self.maxpool.forward(relu_out)
        # print("pool_out", pool_out.shape)
        flatten_out = self.flatten.forward(maxpool_out)
        # print("flatten_out", flatten_out.shape)
        linear_out = self.linear.forward(flatten_out)
        # print("linear_out", linear_out.shape)
        loss, predictions = self.loss.forward(linear_out, y_labels, get_predictions=True)

        return loss, predictions

    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        loss_grad = self.loss.backward()
        lin_w_grad, lin_b_grad, lin_x_grad = self.linear.backward(loss_grad)
        lin_x_grad = lin_x_grad.reshape(self.batch_size, 5, 16, 16)
        maxpool_grad = self.maxpool.backward(lin_x_grad)
        relu_grad = self.relu.backward(maxpool_grad)
        conv_w_grad, conv_b_grad, conv_x_grad = self.conv.backward(relu_grad)

    def update(self, learning_rate, momentum_coeff):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        self.conv.update(learning_rate=learning_rate, momentum_coeff=momentum_coeff)
        self.linear.update(learning_rate=learning_rate, momentum_coeff=momentum_coeff)


class ConvNetThree:
    """
    Class to implement forward and backward pass of the following network -
    Conv -> Relu -> MaxPool -> Linear -> Softmax
    For the above network run forward, backward and update
    """
    def __init__(self):
        self.conv1 = Conv(input_shape=(3,32,32), filter_shape=(5,5,5))
        self.relu1 = ReLU()
        self.maxpool = MaxPool(filter_shape=(2,2), stride=2)

        self.conv2 = Conv(input_shape=(5,16,16), filter_shape=(5,5,5))
        self.relu2 = ReLU()

        self.conv3 = Conv(input_shape=(5,16,16), filter_shape=(5,5,5))
        self.relu3 = ReLU()

        self.flatten = Flatten()
        self.linear = LinearLayer(indim=1280, outdim=10)
        self.loss = SoftMaxCrossEntropyLoss()

    def forward(self, inputs, y_labels):
        self.inputs = inputs
        self.batch_size, self.channels, self.height, self.width = inputs.shape

        conv1_out = self.conv1.forward(inputs)  # use defult stride and pad?
        relu1_out = self.relu1.forward(conv1_out)
        maxpool_out = self.maxpool.forward(relu1_out)

        conv2_out = self.conv2.forward(maxpool_out)  
        relu2_out = self.relu2.forward(conv2_out)

        conv3_out = self.conv3.forward(relu2_out)  
        relu3_out = self.relu3.forward(conv3_out)
        
        flatten_out = self.flatten.forward(relu3_out)
        linear_out = self.linear.forward(flatten_out)

        loss, predictions = self.loss.forward(linear_out, y_labels, get_predictions=True)

        return loss, predictions

    def backward(self):
        loss_grad = self.loss.backward()
        _, _, lin_x_grad = self.linear.backward(loss_grad)
        lin_x_grad = lin_x_grad.reshape(self.batch_size, 5, 16, 16)

        relu3_grad = self.relu3.backward(lin_x_grad)
        _, _, conv3_x_grad = self.conv3.backward(relu3_grad)

        relu2_grad = self.relu2.backward(conv3_x_grad)
        _, _, conv2_x_grad = self.conv2.backward(relu2_grad)

        maxpool_grad = self.maxpool.backward(conv2_x_grad)
        relu1_grad = self.relu1.backward(maxpool_grad)
        _, _, conv1_x_grad = self.conv1.backward(relu1_grad)

    def update(self, learning_rate, momentum_coeff):
        self.conv1.update(learning_rate=learning_rate, momentum_coeff=momentum_coeff)
        self.conv2.update(learning_rate=learning_rate, momentum_coeff=momentum_coeff)
        self.conv3.update(learning_rate=learning_rate, momentum_coeff=momentum_coeff)
        self.linear.update(learning_rate=learning_rate, momentum_coeff=momentum_coeff)


class MLP:
    """
    Implement as you wish, not autograded
    """
    def __init__(self):
        pass

    def forward(self, inputs, y_labels):
        pass

    def backward(self):
        pass

    def update(self,learning_rate,momentum_coeff):
        pass
