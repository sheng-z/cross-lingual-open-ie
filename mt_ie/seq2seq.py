#!/usr/bin/env python
# encoding: utf-8


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip     # pylint: disable=redefined-builtin

import tensorflow as tf
from tensorflow.python import shape
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.python.ops import seq2seq


linear = rnn_cell._linear  # pylint: disable=protected-access


def _beam_search_and_embed(embedding, beam_size, num_symbols, embedding_size,
                           output_projection=None,
                           update_embedding_for_previous=True):
    def loop_function(prev, i, beam_symbols, beam_path, beam_log_probs):
        """Get a loop_function that extract the beam_sized previous symbols
        and embeds it.

        Args:
            prev: previous decoder output of shape [batch_size * beam_size, num_symbols]
                if i > 1 else [batch_size, num_symbols].
            i: decoding step.
            beam_symbols: a (i-1)-length list of tensors in shape [batch_size, beam_size],
                which are symbols in the beam at each step.
            beam_path: a (i-1)-length list of tensors in shape [batch_size, beam_size],
                which are indices for previous symbols in the beam at each step.
            beam_log_probs: a (i-1)-length list of tensors in shape [batch_size * beam_size, 1],
                which are log probabilities in the beam at each step.

        """
        if output_projection is not None:
            prev = nn_ops.xw_plus_b(prev,
                                    output_projection[0], output_projection[1])

        log_probs = tf.log(tf.nn.softmax(prev))

        if i > 1:
            # broadcasting occurs in the add operation where beam_log_probs[-1]
            # is in shape [batch_size * beam_size, 1].
            log_probs = tf.reshape(log_probs + beam_log_probs[-1],
                                   [-1, beam_size * num_symbols])

        # Both returns are in shape [batch_size, beam_size].
        best_log_probs, best_indices = tf.nn.top_k(log_probs, beam_size)
        # Reshape best_indices to shape [batch_size * beam_size].
        best_indices = tf.stop_gradient(tf.squeeze(tf.reshape(best_indices, [-1, 1])))
        # Reshape best_log_probs to shape [batch_size * beam_size, 1].
        best_log_probs = tf.stop_gradient(tf.reshape(best_log_probs, [-1, 1]))

        symbols = best_indices % num_symbols
        parent_indices = best_indices // num_symbols

        beam_symbols.append(tf.reshape(symbols, [-1, beam_size]))
        beam_path.append(tf.reshape(parent_indices, [-1, beam_size]))
        beam_log_probs.append(best_log_probs)

        # emb_prev has shape [batch_size * beam_size, embedding_size].
        emb_prev = embedding_ops.embedding_lookup(embedding, symbols)
        if not update_embedding_for_previous:
            emb_prev = tf.stop_gradient(emb_prev)
        return tf.reshape(emb_prev, [-1, embedding_size])
    return loop_function


def attention_decoder(decoder_inputs,
                      list_of_mask,
                      initial_state,
                      attention_states,
                      cell,
                      output_size=None,
                      num_heads=1,
                      beam_size=1,
                      loop_function=None,
                      dtype=None,
                      scope=None,
                      initial_state_attention=False):
    """RNN decoder with attention for the sequence-to-sequence model.

    In this context "attention" means that, during decoding, the RNN can look up
    information in the additional tensor attention_states, and it does this by
    focusing on a few entries from the tensor. This model has proven to yield
    especially good results in a number of sequence-to-sequence tasks. This
    implementation is based on http://arxiv.org/abs/1412.7449 (see below for
    details). It is recommended for complex sequence-to-sequence tasks.

    Args:
      decoder_inputs: A list of 2D Tensors [batch_size x input_size].
      initial_state: 2D Tensor [batch_size x cell.state_size].
      attention_states: 3D Tensor [batch_size x attn_length x attn_size].
      cell: rnn_cell.RNNCell defining the cell function and size.
      output_size: Size of the output vectors; if None, we use cell.output_size.
      num_heads: Number of attention heads that read from attention_states.
      loop_function: If not None, this function will be applied to i-th output
        in order to generate i+1-th input, and decoder_inputs will be ignored,
        except for the first element ("GO" symbol). This can be used for decoding,
        but also for training to emulate http://arxiv.org/abs/1506.03099.
        Signature -- loop_function(prev, i) = next
          * prev is a 2D Tensor of shape [batch_size x output_size],
          * i is an integer, the step number (when advanced control is needed),
          * next is a 2D Tensor of shape [batch_size x input_size].
      dtype: The dtype to use for the RNN initial state (default: tf.float32).
      scope: VariableScope for the created subgraph; default: "attention_decoder".
      initial_state_attention: If False (default), initial attentions are zero.
        If True, initialize the attentions from the initial state and attention
        states -- useful when we wish to resume decoding from a previously
        stored decoder state and attention states.

    Returns:
      A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors of
          shape [batch_size x output_size]. These represent the generated outputs.
          Output i is computed from input i (which is either the i-th element
          of decoder_inputs or loop_function(output {i-1}, i)) as follows.
          First, we run the cell on a combination of the input and previous
          attention masks:
            cell_output, new_state = cell(linear(input, prev_attn), prev_state).
          Then, we calculate new attention masks:
            new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
          and then we calculate the output:
            output = linear(cell_output, new_attn).
        state: The state of each decoder cell the final time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].

    Raises:
      ValueError: when num_heads is not positive, there are no inputs, shapes
        of attention_states are not set, or input size cannot be inferred
        from the input.
    """
    if not decoder_inputs:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention decoder.")
    if not attention_states.get_shape()[1:2].is_fully_defined():
        raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                     % attention_states.get_shape())
    if output_size is None:
        output_size = cell.output_size

    with variable_scope.variable_scope(
            scope or "attention_decoder", dtype=dtype) as scope:
        dtype = scope.dtype

        batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
        attn_length = attention_states.get_shape()[1].value
        attn_size = attention_states.get_shape()[2].value

        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = array_ops.reshape(
            attention_states, [-1, attn_length, 1, attn_size])
        hidden_features = []
        v = []

        # Tile hidden states for beam search.
        if beam_size > 1:
            tiled_attention_states = tf.reshape(tf.tile(tf.reshape(
                attention_states, [-1, 1, attn_length, attn_size]),
                [1, beam_size, 1, 1]), [-1, attn_length, attn_size])
            tiled_hidden = array_ops.reshape(
                tiled_attention_states, [-1, attn_length, 1, attn_size])
            tiled_hidden_features = []

        attention_vec_size = attn_size // 2  # Size of query vectors for attention.
        for a in xrange(num_heads):
            k = variable_scope.get_variable("AttnW_%d" % a,
                                            [1, 1, attn_size, attention_vec_size])
            hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
            v.append(
                variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size]))

            if beam_size > 1:
                tiled_hidden_features.append(nn_ops.conv2d(tiled_hidden, k,
                                                           [1, 1, 1, 1], "SAME"))

        # state = (state_1(c, h), state_2(c, h))
        state = initial_state

        state_size = int(state[0].c.get_shape().with_rank(2)[1])

        def attention(query, tiled=False):
            """Put attention masks on hidden using hidden_features and query."""
            ds = []  # Results of attention reads will be stored here.
            if nest.is_sequence(query):  # If the query is a tuple, flatten it.
                query_list = nest.flatten(query)
                for q in query_list:  # Check that ndims == 2 if specified.
                    ndims = q.get_shape().ndims
                    if ndims:
                        assert ndims == 2
                query = array_ops.concat(1, query_list)
            for a in xrange(num_heads):
                with variable_scope.variable_scope("Attention_%d" % a):
                    y = linear(query, attention_vec_size, True)
                    y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                    # Use tiled hidden states and features to compute attention
                    # for beam search.
                    if tiled == True:
                        encoder_hidden_states = tiled_hidden
                        encoder_hidden_features = tiled_hidden_features
                    else:
                        encoder_hidden_states = hidden
                        encoder_hidden_features = hidden_features
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = math_ops.reduce_sum(
                        v[a] * math_ops.tanh(encoder_hidden_features[a] + y), [2, 3])
                    a = nn_ops.softmax(s)
                    # Now calculate the attention-weighted vector d.
                    d = math_ops.reduce_sum(
                        array_ops.reshape(a, [-1, attn_length, 1, 1]) * encoder_hidden_states,
                        [1, 2])
                    ds.append(array_ops.reshape(d, [-1, attn_size]))
            return ds

        outputs = []
        prev = None
        batch_attn_size = array_ops.pack([batch_size, attn_size])
        ones = array_ops.ones([batch_size, 1], dtype=dtype)
        attns = [array_ops.zeros(batch_attn_size, dtype=dtype)
                 for _ in xrange(num_heads)]
        # Properties for beam search.
        beam_symbols, beam_path, beam_log_probs = [], [], []

        for a in attns:  # Ensure the second shape of attention vectors is set.
            a.set_shape([None, attn_size])
        if initial_state_attention:
            attns = attention(initial_state)
        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            # If loop_function is set, we use it instead of decoder_inputs.
            if prev is not None:
                # if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    if beam_size > 1:
                        # Only use beam search in decoding.
                        inp = loop_function(prev, i, beam_symbols, beam_path, beam_log_probs)
                    else:
                        # Use mask to randomly select the decoder input from the gold input
                        # or the previous decoder output.
                        mask = tf.reshape(tf.cast(list_of_mask[i], dtype=dtype),
                                          [batch_size, 1])
                        prev = loop_function(prev, i)
                        inp = tf.stop_gradient((ones - mask) * inp + mask * prev)


            # Merge input and previous attentions into one vector of the right size.
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)
            x = linear([inp] + attns, input_size, True)
            # Run the RNN.
            cell_output, state = cell(x, state)
            # Run the attention mechanism.
            if i == 0 and initial_state_attention:
                with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                                   reuse=True):
                    attns = attention(state)
            else:
                if beam_size > 1 and i == 0:
                    # Save the untiled first-step attention to compute the
                    # first-step output.
                    first_step_attns = attention(state, False)
                    # Tile state for beam search
                    states = []
                    replicate = lambda x: tf.reshape(tf.tile(tf.reshape(
                        x, [-1, 1, state_size]), [1, beam_size, 1]), [-1, state_size])
                    for state_i in state:
                        c, h = state_i
                        states.append((replicate(c), replicate(h)))
                    state = states
                    # Compute attention for the next step.
                    with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                                   reuse=True):
                        attns = attention(state, True)
                else:
                    if beam_size > 1:
                        attns = attention(state, True)
                    else:
                        attns = attention(state)


            with variable_scope.variable_scope("AttnOutputProjection"):
                if beam_size > 1 and i == 0:
                    output = linear([cell_output] + first_step_attns, output_size, True)
                else:
                    output = linear([cell_output] + attns, output_size, True)

            if loop_function is not None:
                prev = output

            outputs.append(output)

    # Add the last-step beam properties.
    if beam_size > 1 and len(decoder_inputs) > 0:
        loop_function(prev, i, beam_symbols, beam_path, beam_log_probs)

    return outputs, state, [beam_symbols, beam_path, beam_log_probs]


def embedding_attention_decoder(decoder_inputs,
                                list_of_mask,
                                initial_state,
                                attention_states,
                                cell,
                                num_symbols,
                                embedding_size,
                                num_heads=1,
                                beam_size=1,
                                output_size=None,
                                output_projection=None,
                                update_embedding_for_previous=False,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False):
    """RNN decoder with embedding and attention and a pure-decoding option.

    Args:
      decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
      initial_state: 2D Tensor [batch_size x cell.state_size].
      attention_states: 3D Tensor [batch_size x attn_length x attn_size].
      cell: rnn_cell.RNNCell defining the cell function.
      num_symbols: Integer, how many symbols come into the embedding.
      embedding_size: Integer, the length of the embedding vector for each symbol.
      num_heads: Number of attention heads that read from attention_states.
      output_size: Size of the output vectors; if None, use output_size.
      output_projection: None or a pair (W, B) of output projection weights and
        biases; W has shape [output_size x num_symbols] and B has shape
        [num_symbols]; if provided and feed_previous=True, each fed previous
        output will first be multiplied by W and added B.
      feed_previous: Boolean; if True, only the first of decoder_inputs will be
        used (the "GO" symbol), and all other decoder inputs will be generated by:
          next = embedding_lookup(embedding, argmax(previous_output)),
        In effect, this implements a greedy decoder. It can also be used
        during training to emulate http://arxiv.org/abs/1506.03099.
        If False, decoder_inputs are used as given (the standard decoder case).
      update_embedding_for_previous: Boolean; if False and feed_previous=True,
        only the embedding for the first symbol of decoder_inputs (the "GO"
        symbol) will be updated by back propagation. Embeddings for the symbols
        generated from the decoder itself remain unchanged. This parameter has
        no effect if feed_previous=False.
      dtype: The dtype to use for the RNN initial states (default: tf.float32).
      scope: VariableScope for the created subgraph; defaults to
        "embedding_attention_decoder".
      initial_state_attention: If False (default), initial attentions are zero.
        If True, initialize the attentions from the initial state and attention
        states -- useful when we wish to resume decoding from a previously
        stored decoder state and attention states.

    Returns:
      A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors with
          shape [batch_size x output_size] containing the generated outputs.
        state: The state of each decoder cell at the final time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].

    Raises:
      ValueError: When output_projection has the wrong shape.
    """
    if output_size is None:
        output_size = cell.output_size
    if output_projection is not None:
        proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
        proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    with variable_scope.variable_scope(
            scope or "embedding_attention_decoder", dtype=dtype) as scope:

        embedding = variable_scope.get_variable("embedding",
                                              [num_symbols, embedding_size])
        if beam_size > 1:
            loop_function = _beam_search_and_embed(
                embedding, beam_size, num_symbols, embedding_size,
                output_projection, update_embedding_for_previous)
        else:
            loop_function = seq2seq._extract_argmax_and_embed(
                embedding, output_projection,
                update_embedding_for_previous)
        emb_inp = [
            embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]
        return attention_decoder(
            emb_inp,
            list_of_mask,
            initial_state,
            attention_states,
            cell,
            output_size=output_size,
            num_heads=num_heads,
            beam_size=beam_size,
            loop_function=loop_function,
            initial_state_attention=initial_state_attention)


def embedding_attention_bidirectional_seq2seq(encoder_inputs,
                                              decoder_inputs,
                                              encoder_input_length,
                                              list_of_mask,
                                              encoder_cell,
                                              decoder_cell,
                                              num_encoder_symbols,
                                              num_decoder_symbols,
                                              embedding_size,
                                              num_heads=1,
                                              beam_size=1,
                                              output_projection=None,
                                              dtype=None,
                                              scope=None,
                                              initial_state_attention=False):
    """Embedding sequence-to-sequence model with attention.

    This model first embeds encoder_inputs by a newly created embedding (of shape
    [num_encoder_symbols x input_size]). Then it runs an bidirectional RNN to encode
    embedded encoder_inputs into a state vector. It keeps the outputs of this
    RNN at every step to use for attention later. Next, it embeds decoder_inputs
    by another newly created embedding (of shape [num_decoder_symbols x
    input_size]). Then it runs attention decoder, initialized with the last
    encoder state, on embedded decoder_inputs and attending to encoder outputs.

    Args:
      encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
      decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
      encoder_cell: rnn_cell.RNNCell defining the cell function and size.
      decoder_cell: rnn_cell.RNNCell defining the cell function and size.
      num_encoder_symbols: Integer; number of symbols on the encoder side.
      num_decoder_symbols: Integer; number of symbols on the decoder side.
      embedding_size: Integer, the length of the embedding vector for each symbol.
      num_heads: Number of attention heads that read from attention_states.
      output_projection: None or a pair (W, B) of output projection weights and
        biases; W has shape [output_size x num_decoder_symbols] and B has
        shape [num_decoder_symbols]; if provided and feed_previous=True, each
        fed previous output will first be multiplied by W and added B.
      feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
        of decoder_inputs will be used (the "GO" symbol), and all other decoder
        inputs will be taken from previous outputs (as in embedding_rnn_decoder).
        If False, decoder_inputs are used as given (the standard decoder case).
      dtype: The dtype of the initial RNN state (default: tf.float32).
      scope: VariableScope for the created subgraph; defaults to
        "embedding_attention_seq2seq".
      initial_state_attention: If False (default), initial attentions are zero.
        If True, initialize the attentions from the initial state and attention
        states.

    Returns:
      A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors with
          shape [batch_size x num_decoder_symbols] containing the generated
          outputs.
        state: The state of each decoder_cell at the final time-step.
          It is a 2D Tensor of shape [batch_size x decoder_cell.state_size].
    """
    with variable_scope.variable_scope(
            scope or "embedding_attention_bidirectional_seq2seq", dtype=dtype) as scope:
        dtype = scope.dtype
        # Encoder.
        # encoder_cell = rnn_cell.EmbeddingWrapper(
        #     encoder_cell, embedding_classes=num_encoder_symbols,
        #     embedding_size=embedding_size)
        embedding = variable_scope.get_variable("encoder_embedding",
                                              [num_encoder_symbols, embedding_size])
        encoder_inputs = array_ops.pack([
            embedding_ops.embedding_lookup(embedding, i) for i in encoder_inputs])
        # encoder_inputs = array_ops.reshape(array_ops.pack(encoder_inputs), [50, -1, 1])
        encoder_outputs, encoder_states = rnn.bidirectional_dynamic_rnn(
            cell_fw=encoder_cell,
            cell_bw=encoder_cell,
            inputs=encoder_inputs,
            sequence_length=encoder_input_length,
            time_major=True,
            dtype=dtype)
        encoder_state_fw, encoder_state_bw = encoder_states

        # Concatenate output_fw and output_bw => [step, batch_size, cell.out_size * 2].
        concat_encoder_outputs = array_ops.concat(2, encoder_outputs)
        # Transpose to [batch_size, step, cell.out_size * 2].
        attention_states = array_ops.transpose(concat_encoder_outputs, [1, 0, 2])

        # Decoder.
        output_size = None
        if output_projection is None:
            decoder_cell = rnn_cell.OutputProjectionWrapper(decoder_cell, num_decoder_symbols)
            output_size = num_decoder_symbols

        return embedding_attention_decoder(
            decoder_inputs,
            list_of_mask,
            encoder_state_bw,
            attention_states,
            decoder_cell,
            num_decoder_symbols,
            embedding_size,
            num_heads=num_heads,
            beam_size=beam_size,
            output_size=output_size,
            output_projection=output_projection,
            update_embedding_for_previous=False,
            initial_state_attention=initial_state_attention)


def model(encoder_inputs,
          decoder_inputs,
          targets,
          weights,
          encoder_input_length,
          list_of_mask,
          encoder_cell,
          decoder_cell,
          num_encoder_symbols,
          num_decoder_symbols,
          embedding_size,
          beam_size=1,
          output_projection=None,
          softmax_loss_function=None,
          dtype=None,
          name=None):
    all_inputs = encoder_inputs + decoder_inputs + targets + weights
    with ops.name_scope(name, "seq2seq_model", all_inputs):
        with variable_scope.variable_scope("model_seq2seq"):
            outputs, _, beams = embedding_attention_bidirectional_seq2seq(
                encoder_inputs,
                decoder_inputs,
                encoder_input_length,
                list_of_mask,
                encoder_cell,
                decoder_cell,
                num_encoder_symbols,
                num_decoder_symbols,
                embedding_size,
                beam_size=beam_size,
                output_projection=output_projection,
                dtype=dtype)
            loss = None
            if beam_size == 1:
                loss = seq2seq.sequence_loss(outputs,
                                         targets,
                                         weights,
                                         softmax_loss_function=softmax_loss_function)
    return outputs, loss, beams
