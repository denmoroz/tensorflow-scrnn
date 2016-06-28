import math

from tensorflow.models.rnn.rnn_cell import RNNCell

from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs


def glorot_initializer(in_size, out_size):
    """
    Normalized initialization proposed for variance stabilization per layer

    Links:

    Understanding the difficulty of training deep feedforward neural networks
    http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    """
    width = math.sqrt(6.0 / (in_size + out_size))
    return init_ops.random_uniform_initializer(-width, width)


class SCRNNCell(RNNCell):
    """
    Tensor Flow port of Structurally Constrained Recurrent Neural Network model

    Links:

    Learning Longer Memory in Recurrent Neural Networks
    http://arxiv.org/abs/1412.7753

    Original implentation in Torch
    https://github.com/facebook/SCRNNs
    """
    def __init__(self, batch_size, input_size,
                 hidden_size, context_size, alpha):
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._context_size = context_size
        self._batch_size = batch_size

        self._alpha = alpha

    @property
    def state_size(self):
        return self._hidden_size + self._context_size

    @property
    def output_size(self):
        return self._hidden_size + self._context_size

    def __call__(self, inputs, state, scope=None):
        # state = [h_(t-1), s_(t-1)]
        hidden_state = array_ops.slice(
            state, begin=(0, 0),
            size=(self._batch_size, self._hidden_size)
        )

        context_state = array_ops.slice(
            state, begin=(0, self._hidden_size),
            size=(self._batch_size, self._context_size)
        )

        with vs.variable_scope(scope or type(self).__name__):
            B = vs.get_variable(
                'B_matrix', shape=[self._input_size, self._context_size],
                initializer=glorot_initializer(
                    in_size=self._input_size,
                    out_size=self._context_size
                )
            )

            A = vs.get_variable(
                'A_matrix', shape=[self._input_size, self._hidden_size],
                initializer=glorot_initializer(
                    in_size=self._input_size,
                    out_size=self._hidden_size
                )
            )

            R = vs.get_variable(
                'R_matrix', shape=[self._hidden_size, self._hidden_size],
                initializer=glorot_initializer(
                    in_size=self._hidden_size,
                    out_size=self._hidden_size
                )
            )

            P = vs.get_variable(
                'P_matrix', shape=[self._context_size, self._hidden_size],
                initializer=glorot_initializer(
                    in_size=self._context_size,
                    out_size=self._hidden_size
                )
            )

            bias_term = vs.get_variable(
                'Bias', shape=[self._hidden_size],
                initializer=init_ops.constant_initializer(
                    value=0.0
                )
            )

            new_context = (1.0 - self._alpha) * math_ops.matmul(inputs, B) + self._alpha * context_state

            # TODO: math_ops.batch_matmul?
            # math_ops.tanh, nn_ops.softsign
            new_hidden = nn_ops.elu(
                math_ops.matmul(new_context, P) +
                math_ops.matmul(inputs, A) +
                math_ops.matmul(hidden_state, R) +
                bias_term
            )

            new_state = array_ops.concat(
                values=[new_hidden, new_context], concat_dim=1
            )

        return new_state, new_state
