import numpy as np
import numpy_ml as npml
from collections import OrderedDict
from numpy_ml.neural_nets.layers import Conv2D, BatchNorm2D, Flatten, Softmax, Pool2D,FullyConnected
from numpy_ml.neural_nets.activations import ReLU
from time import time
from numpy_ml.neural_nets.utils import minibatch
from numpy_ml.neural_nets.losses import CrossEntropy

#npml.neural_nets.layers.Conv2D
test_model = npml.neural_nets.models.WGAN_GP(g_hidden=512, init='he_uniform', optimizer='RMSProp(lr=0.0001)', debug=False)


class resnet50_bottlenet(object):
    def __init__(
        self,
        init = "he_uniform",
        optimizer = "Adam",
        net_conv1_pad = 1,
        net_conv1_out_ch = 32,
        net_conv1_stride = 1,
        net_conv1_kernel_shape = (7, 7),
    ):
        self.init = init
        self.optimizer = optimizer
        self.net_conv1_pad = net_conv1_pad
        self.net_conv1_out_ch = net_conv1_out_ch
        self.net_conv1_stride = net_conv1_stride
        self.net_conv1_kernel_shape = net_conv1_kernel_shape
        self.loss = CrossEntropy()
        self._init_param()


    def _init_param(self):
        self._build_net()


    def _build_net(self):
        """
        Conv1 -> BN -> Scale -> ReLU
        """
        self.net = OrderedDict()
        self.net["Conv1"] = Conv2D(
            act_fn=ReLU(),
            init=self.init,
            pad='same',
            optimizer=self.optimizer,
            out_ch=self.net_conv1_out_ch,
            stride=self.net_conv1_stride,
            kernel_shape=self.net_conv1_kernel_shape,
        )

        #self.net["Pool1"] = Pool2D(
        #    mode="max",
        #    optimizer=self.optimizer,
        #    pad='same',
        #    stride=1,
        #    kernel_shape=self.net_conv1_kernel_shape,
        #)
        self.net["Flatten"] = Flatten(optimizer=self.optimizer)
        self.net["FC5"] = FullyConnected(
            n_out=10,
            optimizer=self.optimizer,
            act_fn=None,
            init=self.init,
        )
        self.net["Softmax"] = Softmax()



    @property
    def gradients(self):
        return {
            "components": {k: v.gradients for k, v in self.net.items()}
        }


    def forward(self, X, retain_derived=True):
        mod = self.net
        Xs = {}
        out, rd = X, retain_derived
        for k, v in mod.items():
            out = v.forward(out, retain_derived = rd)
            Xs[k] = out
        return out, Xs


    def backward(self, grad, retain_grads = True):
        mod = self.net
        out, rg = grad, retain_grads
        dXs = {}
        for k, v in reversed(list(mod.items())):
            if k == 'Softmax':
                out = out
            else:
                dXs[k] = out
                out = v.backward(out, retain_grads=rg)
        return out, dXs


    def update(self, cur_loss=None):
        """Perform gradient updates"""
        for k, v in reversed(list(self.net.items())):
            v.update(cur_loss)
        for k, v in reversed(list(self.net.items())):
            v.update(cur_loss)
        self.flush_gradients()


    def flush_gradients(self):
        """Reset parameter gradients to 0 after an update."""
        mod = self.net
        for k, v in mod.items():
            v.flush_gradients()


    def fit(
        self,
        X_train,
        y_target,
        n_epochs = 1,
        batchsize = 128,
        verbose = True,
    ):
        self.verbose = verbose
        self.n_epochs = n_epochs
        self.batchsize = batchsize

        _, self.in_rows, self.in_cols, self.in_ch = X_train.shape

        prev_loss = np.inf
        for i in range(n_epochs):
            loss, estart = 0.0, time()
            batch_generator, nb = minibatch(X_train, batchsize, shuffle=True)

            for j, b_ix in enumerate(batch_generator):
                bsize, bstart = len(b_ix), time()
                X_batch = X_train
                #X_batch_col = X_train[b_ix].reshape(bsize, -1)
                a = []
                y_predict, Xs_out = self.forward(X_batch)
                y_pred = self.loss.grad(y_target, y_predict)
                batch_loss = self.loss(y_target, y_predict)
                print("batch_loss", batch_loss)
                self.backward(y_pred/128)

                #self.update(batch_loss)

                loss += batch_loss
                a = []
                if self.verbose:
                    fstr = "\t[Batch {}/{}] Train loss: {:.3f} ({:.1f}s/batch)"
                    print(fstr.format(j + 1, nb, batch_loss, time() - bstart))

            loss /= nb
            fstr = "[Epoch {}] Avg. loss: {:.3f}  Delta: {:.3f} ({:.2f}m/epoch)"
            print(fstr.format(i + 1, loss, prev_loss - loss, (time() - estart) / 60.0))
            prev_loss = loss


#x_input = np.random.rand(128, 24, 24, 3)
#y_input = np.random.randint(0, 2, size=(128, 10))

x_input = np.load('x_input.npy')
y_input = np.load('y_input.npy')


rb = resnet50_bottlenet()
rb.fit(x_input, y_input)
aa = rb.gradients()
a = []









