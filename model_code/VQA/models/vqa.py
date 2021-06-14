"""Contains code for the VQA model based on variational autoencoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_cnn import EncoderCNN
from .encoder_rnn import EncoderRNN
from .decoder_rnn import DecoderRNN
from .mlp import MLP


class VQA(nn.Module):
    """Information Maximization question generation.
    """

    def __init__(self, vocab_size, max_len, hidden_size,
                 sos_id, eos_id,
                 num_layers=1, rnn_cell='LSTM', bidirectional=False,
                 input_dropout_p=0, dropout_p=0,
                 encoder_max_len=None, num_att_layers=2, att_ff_size=512,
                 embedding=None, z_size=20, no_question_recon=False, 
                 no_image_recon=False):
        """Constructor for VQA.

        Args:
            vocab_size: Number of words in the vocabulary.
            max_len: The maximum length of the answers we generate.
            hidden_size: Number of dimensions of RNN hidden cell.
            num_categories: The number of answer categories.
            sos_id: Vocab id for <start>.
            eos_id: Vocab id for <end>.
            num_layers: The number of layers of the RNNs.
            rnn_cell: LSTM or RNN or GRU.
            bidirectional: Whether the RNN is bidirectional.
            input_dropout_p: Dropout applied to the input question words.
            dropout_p: Dropout applied internally between RNN steps.
            encoder_max_len: Maximum length of encoder.
            num_att_layers: Number of stacked attention layers.
            att_ff_size: Dimensions of stacked attention.
            embedding (vocab_size, hidden_size): Tensor of embeddings or
                None. If None, embeddings are learned.
            z_size: Dimensions of noise epsilon.
        """
        super(VQA, self).__init__()
        self.question_recon = not no_question_recon
        self.image_recon = not no_image_recon
        self.hidden_size = hidden_size
        if encoder_max_len is None:
            encoder_max_len = max_len
        self.num_layers = num_layers

        # Setup image encoder.
        self.encoder_cnn = EncoderCNN(hidden_size)

        
        # Setup answer encoder.
        self.question_encoder = EncoderRNN(vocab_size, max_len, hidden_size,
                                         input_dropout_p=input_dropout_p,
                                         dropout_p=dropout_p,
                                         n_layers=num_layers,
                                         bidirectional=False,
                                         rnn_cell=rnn_cell,
                                         variable_lengths=True)
        

        # Setup stacked attention to combine image and answer features.
        self.question_attention = MLP(2*hidden_size, att_ff_size, hidden_size,
                                    num_layers=num_att_layers)


        # Setup answer decoder.
        self.z_decoder = nn.Linear(z_size, hidden_size)
        self.gen_decoder = MLP(hidden_size, att_ff_size, hidden_size,
                               num_layers=num_att_layers)
        self.decoder = DecoderRNN(vocab_size, max_len, hidden_size,
                                  sos_id=sos_id,
                                  eos_id=eos_id,
                                  n_layers=num_layers,
                                  rnn_cell=rnn_cell,
                                  input_dropout_p=input_dropout_p,
                                  dropout_p=dropout_p,
                                  embedding=embedding)

        # Setup encodering to z space.
        self.mu_question_encoder = nn.Linear(hidden_size, z_size)
        self.logvar_question_encoder = nn.Linear(hidden_size, z_size)
        

        # Setup image reconstruction.
        if self.image_recon:
            self.image_reconstructor = MLP(
                    z_size, att_ff_size, hidden_size,
                    num_layers=num_att_layers)

        # Setup answer reconstruction.
        if self.question_recon:
            self.question_reconstructor = MLP(
                    z_size, att_ff_size, hidden_size,
                    num_layers=num_att_layers)
        
        

    def flatten_parameters(self):
        if hasattr(self, 'decoder'):
            self.decoder.rnn.flatten_parameters()
        if hasattr(self, 'encoder'):
            self.encoder.rnn.flatten_parameters()

    def generator_parameters(self):
        params = self.parameters()
        params = filter(lambda p: p.requires_grad, params)
        return params

    def info_parameters(self):
        params = (list(self.question_attention.parameters()) +
                  list(self.mu_question_encoder.parameters()) +
                  list(self.logvar_question_encoder.parameters()))

        # Reconstruction parameters.
        if self.image_recon:
            params += list(self.image_reconstructor.parameters())
        if self.question_recon:
            params += list(self.question_reconstructor.parameters())

        params = filter(lambda p: p.requires_grad, params)
        return params

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def modify_hidden(self, func, hidden, rnn_cell):
        """Applies the function func to the hidden representation.

        This method is useful because some RNNs like LSTMs have a tuples.

        Args:
            func: A function to apply to the hidden representation.
            hidden: A RNN (or LSTM or GRU) representation.
            rnn_cell: One of RNN, LSTM or GRU.

        Returns:
            func(hidden).
        """
        if rnn_cell is nn.LSTM:
            return (func(hidden[0]), func(hidden[1]))
        return func(hidden)

    def parse_outputs_to_tokens(self, outputs):
        """Converts model outputs to tokens.

        Args:
            outputs: Model outputs.

        Returns:
            A tensor of batch_size X max_len.
        """
        # Take argmax for each timestep
        # Output is list of MAX_LEN containing BATCH_SIZE * VOCAB_SIZE.

        # BATCH_SIZE * VOCAB_SIZE -> BATCH_SIZE
        outputs = [o.max(1)[1] for o in outputs]

        outputs = torch.stack(outputs)  # Tensor(max_len, batch)
        outputs = outputs.transpose(0, 1)  # Tensor(batch, max_len)
        return outputs

    def encode_images(self, images):
        """Encodes images.

        Args:
            images: Batch of image Tensors.

        Returns:
            Batch of image features.
        """
        return self.encoder_cnn(images)

    def encode_questions(self, questions, qlengths):
        """Encodes the answers.

        Args:

        Returns:
            batch of answer features.
        """
        _, encoder_hidden = self.question_encoder(
                questions, qlengths, None)
        if self.question_encoder.rnn_cell == nn.LSTM:
            encoder_hidden = encoder_hidden[0]

        # Pick the hidden vector from the top layer.
        encoder_hidden = encoder_hidden[-1, :, :].squeeze()
        return encoder_hidden

    def encode_into_z(self, image_features, questions_features):
        """Encodes the attended features into z space.

        Args:
            image_features: Batch of image features.
            answer_features: Batch of answer features.

        Returns:
            mus and logvars of the batch.
        """
        together = torch.cat((image_features, questions_features), dim=1)
        attended_hiddens = self.question_attention(together)
        mus = self.mu_question_encoder(attended_hiddens)
        logvars = self.logvar_question_encoder(attended_hiddens)
        return mus, logvars


    def decode_answers(self, image_features, zs,
                         answers=None, teacher_forcing_ratio=0,
                         decode_function=F.log_softmax):
        """Decodes the question from the latent space.

        Args:
            image_features: Batch of image features.
            zs: Batch of latent space representations.
            questions: Batch of question Variables.
            teacher_forcing_ratio: Whether to predict with teacher forcing.
            decode_function: What to use when choosing a word from the
                distribution over the vocabulary.
        """
        batch_size = zs.size(0)
        z_hiddens = self.z_decoder(zs)
        if image_features is None:
            hiddens = z_hiddens
        else:
            hiddens = self.gen_decoder(image_features + z_hiddens)

        # Reshape encoder_hidden (NUM_LAYERS * N * HIDDEN_SIZE).
        hiddens = hiddens.view((1, batch_size, self.hidden_size))
        hiddens = hiddens.expand((self.num_layers, batch_size,
                                  self.hidden_size)).contiguous()
        if self.decoder.rnn_cell is nn.LSTM:
            hiddens = (hiddens, hiddens)
        result = self.decoder(inputs=answers,
                              encoder_hidden=hiddens,
                              function=decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return result

    def forward(self, images, questions, qlengths=None, answers=None,
                teacher_forcing_ratio=0, decode_function=F.log_softmax):
        """Passes the image and the question through a model and generates answers.

        Args:
            images: Batch of image Variables.
            answers: Batch of answer Variables.
            categories: Batch of answer Variables.
            alengths: List of answer lengths.
            questions: Batch of question Variables.
            teacher_forcing_ratio: Whether to predict with teacher forcing.
            decode_function: What to use when choosing a word from the
                distribution over the vocabulary.

        Returns:
            - outputs: The output scores for all steps in the RNN.
            - hidden: The hidden states of all the RNNs.
            - ret_dict: A dictionary of attributes. See DecoderRNN.py for details.
        """
        # features is (N * HIDDEN_SIZE)
        image_features = self.encode_images(images)

        # encoder_hidden is (N * HIDDEN_SIZE).
        question_hiddens = self.encode_questions(questions, qlengths)
        

        # Calculate the mus and logvars.
        mus, logvars = self.encode_into_z(image_features, question_hiddens)
        zs = self.reparameterize(mus, logvars)
        result = self.decode_answers(image_features, zs,
                                       answers=answers,
                                       decode_function=decode_function,
                                       teacher_forcing_ratio=teacher_forcing_ratio)

        return result

    def reconstruct_inputs(self, image_features, question_features):
        """Reconstructs the image features using the VAE.

        Args:
            image_features: Batch of image features.
            answer_features: Batch of answer features.

        Returns:
            Reconstructed image features and answer features.
        """
        recon_image_features = None
        recon_question_features = None
        mus, logvars = self.encode_into_z(image_features, question_features)
        zs = self.reparameterize(mus, logvars)
        if self.image_recon:
            recon_image_features = self.image_reconstructor(zs)
        if self.question_recon:
            recon_question_features = self.question_reconstructor(zs)
        return recon_image_features, recon_question_features


    def encode_from_question(self, images, questions, lengths=None,):
        """Encodes images and categories in t-space.

        Args:
            images: Batch of image Tensor.
            answers: Batch of answer Tensors.
            alengths: List of answer lengths.

        Returns:
            Batch of latent space encodings.
        """
        # print(questions)
        image_features = self.encode_images(images)
        question_hiddens = self.encode_questions(questions, lengths)
        if image_features.shape!=question_hiddens.shape:
            question_hiddens=question_hiddens.reshape((1,-1))
        # print("image_features.shape: ", image_features.shape)
        # print("question_hiddens.shape: ", question_hiddens.shape)
        mus, logvars = self.encode_into_z(image_features, question_hiddens)
        zs = self.reparameterize(mus, logvars)
        return image_features, zs


    def predict_from_question(self, images, questions, lengths=None,
                            answers=None, teacher_forcing_ratio=0,
                            decode_function=F.log_softmax):
        """Outputs the predicted vocab tokens for the answers in a minibatch.

        Args:
            images: Batch of image Tensors.
            answers: Batch of answer Tensors.
            alengths: List of answer lengths.
            questions: Batch of question Tensors.
            teacher_forcing_ratio: Whether to predict with teacher forcing.
            decode_function: What to use when choosing a word from the
                distribution over the vocabulary.

        Returns:
            A tensor with BATCH_SIZE X MAX_LEN where each element is the index
            into the vocab word.
        """
        image_features, zs = self.encode_from_question(images, questions, lengths=lengths)
        outputs, _, _ = self.decode_answers(image_features, zs, answers=answers,
                                              decode_function=decode_function,
                                              teacher_forcing_ratio=teacher_forcing_ratio)
        return self.parse_outputs_to_tokens(outputs)

