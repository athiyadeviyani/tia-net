import numpy as np

def random_normal_weight_init(indim, outdim):
    return np.random.normal(0,1,(indim, outdim))

def random_weight_init(indim,outdim):
    b = np.sqrt(6)/np.sqrt(indim+outdim)
    return np.random.uniform(-b,b,(indim, outdim))

def zeros_bias_init(outdim):
    return np.zeros((outdim,1))

def labels2onehot(labels):
    return np.array([[i==lab for i in range(10)]for lab in labels],dtype=np.float32)

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
        In this function, we accumulate the gradient values instead of assigning
        the gradient values. This allows us to call forward and backward multiple
        times while only update parameters once.
        Compute and save the gradients wrt the parameters for step()
        Return grad_wrt_x which will be the grad_wrt_out for previous Transform
        """
        pass

    def step(self):
        """
        Apply gradients to update the parameters
        """
        pass

    def zerograd(self):
        """
        This is used to Reset the gradients.
        Usually called before backward()
        """
        pass

class ReLU(Transform):
    """
    relu(x) = max(x, 0)
    """
    def __init__(self):
        Transform.__init__(self)

    def forward(self, x, train=True):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, grad_wrt_out):
        # return np.maximum(grad_wrt_out, 0)
        return grad_wrt_out * (self.x > 0)

class LinearMap(Transform):
    """
    Implement this class
    feel free to use random_xxx_init() functions given on top
    """
    def __init__(self, indim, outdim, alpha=0, lr=0.01):
        Transform.__init__(self)
        """
        indim: input dimension 
        outdim: output dimension
        alpha: parameter for momentum updates
        lr: learning rate
        """
        self.alpha = alpha
        self.lr = lr
        self.W = random_weight_init(indim, outdim)  # test shape = (18, 100), (indim, outdim)
        self.b = zeros_bias_init(outdim)            # test shape = (100, 1),  (outdim, 1)

        self.indim = indim
        self.outdim = outdim 

        self.x = None 
        self.forward_out = None
        self.backward_out = None
        self.w_gradient = np.zeros(self.W.shape)
        self.b_gradient = 0.0
        self.w_update = 0.0
        self.b_update = 0.0

    def forward(self, x):
        """
        x shape (batch_size, indim)  # test shape = (1, 18)
        return shape (batch_size, outdim)
        """
        self.x = x
        self.forward_out = np.dot(x, self.W).T + self.b
        return self.forward_out.T

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (batch_size, outdim)
        return shape (batch_size, indim)
        Your backward call should Accumulate gradients.
        """
        self.w_gradient = np.matmul(self.x.T, grad_wrt_out)   # (1, 18).T (1, 100)
        self.b_gradient = np.sum(grad_wrt_out, axis=0).reshape(self.b.shape)
        self.backward_out = np.dot(self.W, grad_wrt_out.T).T

        return self.backward_out

    def step(self):
        """
        apply gradients calculated by backward() to update the parameters

        Make sure your gradient step takes into account momentum.
        Use alpha as the momentum parameter.
        """
        self.w_update = self.alpha * self.w_update + self.w_gradient 
        self.W = self.W - self.lr * self.w_update

        self.b_update = self.alpha * self.b_update + self.b_gradient
        self.b = self.b - self.lr * self.b_update

    def zerograd(self):
    # reset parameters
        self.w_gradient = np.zeros(self.W.shape)
        self.b_gradient = 0.0

    def getW(self):
    # return weights
        return self.W

    def getb(self):
    # return bias
        return self.b

    def loadparams(self, w, b):
    # Used for Autograder. Do not change.
        self.W, self.b = w, b

class SoftmaxCrossEntropyLoss:
    """
    Implement this class
    """
    def forward(self, logits, labels):
        """
        logits are pre-softmax scores, labels are true labels of given inputs
        labels are one-hot encoded
        logits and labels are in the shape of (batch_size, num_classes)
        returns loss as scalar
        (your loss should be a mean value on batch_size)
        """
        batch_size, _ = logits.shape
        self.logits = logits 
        self.labels = labels

        softmax = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        loss = -1 * np.sum((labels * np.log(softmax)), axis=1)
        self.derivative = (softmax - labels) /  batch_size
        return np.mean(loss)

    def backward(self):
        """
        return shape (batch_size, num_classes)
        (don't forget to divide by batch_size because your loss is a mean)
        """
        return self.derivative

    def getAccu(self):
        """
        return accuracy here (as you wish)
        This part is not autograded.
        """
        pass

class SingleLayerMLP(Transform):
    """
    Implement this class
    """
    def __init__(self, inp, outp, hiddenlayer=100, alpha=0.1, lr=0.01, batchnorm=False, dropout=False, p=0.5):
        Transform.__init__(self)
        self.inp = int(inp)
        self.outp = int(outp)
        self.hiddenlayer = int(hiddenlayer)
        self.alpha = alpha
        self.lr = lr

        # Set to True for the required experiments
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.p = p

        # Initialize network layers
        # layer1 (input -> hidden) | [opt] dropout | ReLU | [opt] batchnorm | layer2 (hidden -> output)
        self.layer1 = LinearMap(indim=inp, outdim=hiddenlayer, alpha=alpha, lr=lr)
        self.drop1 = Dropout(p=p)
        self.relu1 = ReLU()
        self.bn1 = BatchNorm(indim=hiddenlayer, mm=alpha, lr=lr)
        self.layer2 = LinearMap(indim=hiddenlayer, outdim=outp, alpha=alpha, lr=lr)

    def forward(self, x, train=True):
        # x shape (batch_size, indim)
        self.x = x

        # forward -> lm2(relu1(lm1(x)))
        if self.batchnorm:
            layer1_out = self.layer1.forward(x)
            bn1_out = self.bn1.forward(layer1_out, train)
            relu1_out = self.relu1.forward(bn1_out)
            layer2_out = self.layer2.forward(relu1_out)
        elif self.dropout:
            layer1_out = self.layer1.forward(x)
            drop1_out = self.drop1.forward(layer1_out, train)
            relu1_out = self.relu1.forward(drop1_out)
            layer2_out = self.layer2.forward(relu1_out)
        else:
            layer1_out = self.layer1.forward(x)
            relu1_out = self.relu1.forward(layer1_out)
            layer2_out = self.layer2.forward(relu1_out)

        return layer2_out

    def backward(self, grad_wrt_out):
        # backward -> lm1(relu1(lm2(grad_wrt_out)))
        
        if self.batchnorm:
            layer2_grad = self.layer2.backward(grad_wrt_out)
            relu1_grad = self.relu1.backward(layer2_grad)
            bn1_grad = self.bn1.backward(relu1_grad)
            layer1_grad = self.layer1.backward(bn1_grad)
        elif self.dropout:
            layer2_grad = self.layer2.backward(grad_wrt_out)
            relu1_grad = self.relu1.backward(layer2_grad)
            drop1_grad = self.drop1.backward(relu1_grad)
            layer1_grad = self.layer1.backward(drop1_grad)
        else:
            layer2_grad = self.layer2.backward(grad_wrt_out)
            relu1_grad = self.relu1.backward(layer2_grad)
            layer1_grad = self.layer1.backward(relu1_grad)

        return layer1_grad

    def step(self):
        self.layer1.step()
        self.layer2.step()

        # Update gamma and beta from BatchNorm
        if self.batchnorm:
            self.bn1.step()

    def zerograd(self):
        self.layer1.zerograd()
        self.layer2.zerograd()

        if self.batchnorm:
            self.bn1.zerograd()

    def loadparams(self, Ws, bs):
        """
        use LinearMap.loadparams() to implement this
        Ws is a list, whose element is weights array of a layer, first layer first
        bs for bias similarly
        e.g., Ws may be [layer1.W, layer2.W]
        Used for autograder.
        """
        W1, W2 = Ws 
        b1, b2 = bs
        self.layer1.loadparams(W1, b1)
        self.layer2.loadparams(W2, b2)

    def getWs(self):
        """
        Return the weights for each layer
        You need to implement this. 
        Return weights for first layer then second and so on...
        """
        return [self.layer1.getW(), self.layer2.getW()]

    def getbs(self):
        """
        Return the biases for each layer
        You need to implement this. 
        Return bias for first layer then second and so on...
        """
        return [self.layer1.getb(), self.layer2.getb()]


class TwoLayerMLP(Transform):
    """
    Implement this class
    Everything similar to SingleLayerMLP
    """
    def __init__(self, inp, outp, hiddenlayers=[100,100], alpha=0.1, lr=0.01):
        Transform.__init__(self)
        self.inp = inp
        self.outp = outp
        self.h1, self.h2 = hiddenlayers
        self.alpha = alpha
        self.lr = lr

        # Initialize network layers
        # 1. LinearMap of input -> h1
        # 2. ReLU()
        # 3. LinearMap of h1 -> h2
        # 4. ReLU()
        # 5. LinearMap of h2 -> output
        self.layer1 = LinearMap(inp, self.h1, alpha, lr) 
        self.relu1 = ReLU()
        self.layer2 = LinearMap(self.h1, self.h2, alpha, lr)
        self.relu2 = ReLU()
        self.layer3 = LinearMap(self.h2, outp, alpha, lr)

    def forward(self, x, train=True):
        # x shape (batch_size, indim)
        self.x = x

        # forward -> lm3(relu2(lm2(relu1(lm1(x)))))
        layer1_out = self.layer1.forward(x)
        relu1_out = self.relu1.forward(layer1_out)
        layer2_out = self.layer2.forward(relu1_out)
        relu2_out = self.relu2.forward(layer2_out)
        layer3_out = self.layer3.forward(relu2_out)

        return layer3_out

    def backward(self, grad_wrt_out):
        # backward -> lm1(relu1(lm2(relu2(lm3(grad_wrt_out)))))
        layer3_grad = self.layer3.backward(grad_wrt_out)
        relu2_grad = self.relu2.backward(layer3_grad)
        layer2_grad = self.layer2.backward(relu2_grad)
        relu1_grad = self.relu1.backward(layer2_grad)
        layer1_grad = self.layer1.backward(relu1_grad)
        
        return layer1_grad

    def step(self):
        self.layer1.step()
        self.layer2.step()
        self.layer3.step()

    def zerograd(self):
        self.layer1.zerograd()
        self.layer2.zerograd()
        self.layer3.zerograd()

    def loadparams(self, Ws, bs):
        W1, W2, W3 = Ws
        b1, b2, b3 = bs
        self.layer1.loadparams(W1, b1)
        self.layer2.loadparams(W2, b2)
        self.layer3.loadparams(W3, b3)

    def getWs(self):
        return [self.layer1.getW(), self.layer2.getW(), self.layer3.getW()]

    def getbs(self):
        return [self.layer1.getb(), self.layer2.getb(), self.layer3.getb()]


class Dropout(Transform):
    """
    Implement this class
    """
    def __init__(self, p=0.5):
        Transform.__init__(self)
        """
        p is the Dropout probability
        """
        self.p = p

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x, train=True):
        """
        Get and apply a mask generated from np.random.binomial during training
        Scale your output accordingly during testing
        """
        self.x = x
        if train:
            self.mask = np.random.binomial(1, self.p, x.shape) 
            return self.x * self.mask
        
        else:
            return self.x * self.p

    def backward(self, grad_wrt_out):
        """
        This method is only called during training.
        """
        return grad_wrt_out * self.mask


class BatchNorm(Transform):
    """
    Implement this class
    """
    def __init__(self, indim, alpha=0.9, lr=0.01, mm=0.01):
        Transform.__init__(self)
        """
        You shouldn't need to edit anything in init
        """
        self.alpha = alpha  # parameter for running average of mean and variance
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None
        self.lr = lr
        self.mm = mm  # parameter for updating gamma and beta

        self.indim = indim

        """
        The following attributes will be tested
        """
        self.var = np.ones((1, indim))
        self.mean = np.zeros((1, indim))

        self.gamma = np.ones((1, indim))
        self.beta = np.zeros((1, indim))

        """
        gradient parameters
        """
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

        """
        momentum parameters
        """
        self.mgamma = np.zeros_like(self.gamma)
        self.mbeta = np.zeros_like(self.beta)

        """
        inference parameters
        """
        self.running_mean = np.zeros((1, indim))
        self.running_var = np.ones((1, indim))

    def __call__(self, x, train=True):
        return self.forward(x, train)

    def forward(self, x, train=True):
        """
        x shape (batch_size, indim)
        return shape (batch_size, indim)
        """
        # Reference: https://agustinus.kristia.de/techblog/2016/07/04/batchnorm/ 
        # My interpretation of batchnorm: https://drive.google.com/file/d/1dWtmgYXvCSV_b4zSIIFKGp29_h0Gna2Q/view?usp=sharing 

        self.x = x

        if train:
            self.mean = np.mean(self.x, axis=0)
            self.var = np.var(self.x, axis=0)

            ivar = 1.0 / np.sqrt(self.var + self.eps)  # inverse variance
            self.norm = (self.x - self.mean) * ivar
            self.out = self.gamma * self.norm + self.beta

            self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * self.mean
            self.running_var = self.alpha * self.running_var + (1 - self.alpha) * self.var

        else:  # inference time
            irvar = 1.0 / np.sqrt(self.running_var + self.eps)  # inverse running variance
            self.norm = (self.x - self.running_mean) * irvar
            self.out = self.gamma * self.norm + self.beta

        return self.out 

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (batch_size, indim)
        return shape (batch_size, indim)
        """
        # Reference: https://agustinus.kristia.de/techblog/2016/07/04/batchnorm/ 
        # My interpretation of batchnorm: https://drive.google.com/file/d/1dWtmgYXvCSV_b4zSIIFKGp29_h0Gna2Q/view?usp=sharing 

        norm = self.x - self.mean
        ivar = 1.0 / np.sqrt(self.var + self.eps)

        dnorm = grad_wrt_out * self.gamma
        divar = np.sum(dnorm * norm, axis=0)
        dvar = divar * -0.5 * ivar**3

        dx_mean1 = (dnorm * ivar)
        dx_mean2 = 2 * norm * 1.0/self.x.shape[0] * np.ones(self.x.shape) * dvar
        dmean = -1 * np.sum(dx_mean1+dx_mean2, axis=0)

        dx1 = dx_mean1 + dx_mean2
        dx2 = dmean / self.x.shape[0]
        dx = dx1 + dx2

        self.dgamma = np.sum(grad_wrt_out * self.norm, axis=0)
        self.dbeta = np.sum(grad_wrt_out, axis=0)

        return dx

    def step(self):
        """
        apply gradients calculated by backward() to update the parameters
        Make sure your gradient step takes into account momentum.
        Use mm as the momentum parameter.
        """
        self.mgamma = self.mm * self.mgamma + self.dgamma
        self.gamma = self.gamma - self.lr * self.mgamma

        self.mbeta = self.mm * self.mbeta + self.dbeta
        self.beta = self.beta - self.lr * self.mbeta

    def zerograd(self):
        # reset parameters
        self.gamma = np.ones((1, self.indim))
        self.beta = np.zeros((1, self.indim))

    def getgamma(self):
        # return gamma
        return self.gamma

    def getbeta(self):
        # return beta
        return self.beta

    def loadparams(self, gamma, beta):
        # Used for Autograder. Do not change.
        self.gamma, self.beta = gamma, beta
