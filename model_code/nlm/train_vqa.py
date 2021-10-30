"""This script is used to VAE question generation.
"""
import progressbar
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
import argparse
import json
import logging
import os
import random
import time
import torch
import torch.nn as nn

from models import VQA
from process2json import process2json
from utils import Vocabulary, NLGEval, Dict2Obj
from utils import get_glove_embedding
from utils import get_loader
from utils import load_vocab
from utils import process_lengths
from utils import gaussian_KL_loss
from torchtext.vocab import Vectors

# from utils.store_dataset import save_dataset
from utils.store_dataset import save_dataset
from utils.vocab import build_vocab


def get_wv_embedding(name, embed_size, vocab):
    """Construct embedding tensor.

    Args:
        name (str): Which GloVe embedding to use.
        embed_size (int): Dimensionality of embeddings.
        vocab: Vocabulary to generate embeddings.
    Returns:
        embedding (vocab_size, embed_size): Tensor of
            GloVe word embeddings.
    """
    """    for index, w in zip(vocab.values(), vocab.keys()):
        if w in list(word_vecs.wv.vocab):
            vec = model[w]
        else:
            vec = np.random.uniform(-0.25,0.25, embed_size)
        embedding[index] = vec    
    
    glove = torchtext.vocab.GloVe(name=name,
                                  dim=str(embed_size))
    """
    # name='/home/sarroutim2/PosDoc NLM/Question Answering/Embedding and pretained models/wikipedia-pubmed-and-PMC-w2v.txt'
    w2v = Vectors(name=name)  ##cache='.vector_cache/wiki-PubMed-w2v.txt.pt'
    vocab_size = len(vocab)
    embedding = torch.zeros(vocab_size, embed_size)
    for i in range(vocab_size):
        embedding[i] = w2v[vocab.idx2word[str(i)]]

    return embedding


def create_model(args, vocab, embedding=None):
    """Creates the model.

    Args:
        args: Instance of Argument Parser.
        vocab: Instance of Vocabulary.

    Returns:
        An VQA model.
    """
    # Load GloVe embedding.
    if args.use_glove:
        embedding = get_glove_embedding(args.embedding_name,
                                        300,
                                        vocab)
    elif args.use_w2v:
        embedding = get_wv_embedding(args.embedding_name,
                                     200, vocab)
    else:
        embedding = None

    # Build the models
    # logging.info('Creating VQA model...')
    vqa = VQA(len(vocab), args.max_length, args.hidden_size,

              vocab(vocab.SYM_SOQ), vocab(vocab.SYM_EOS),
              num_layers=args.num_layers,
              rnn_cell=args.rnn_cell,
              dropout_p=args.dropout_p,
              input_dropout_p=args.input_dropout_p,
              encoder_max_len=args.encoder_max_len,
              embedding=embedding,
              num_att_layers=args.num_att_layers,
              z_size=args.z_size,
              no_question_recon=args.no_question_recon,
              no_image_recon=args.no_image_recon)
    return vqa


def evaluate(vqa, data_loader, criterion, l2_criterion, args):
    """Calculates vqa average loss on data_loader.

    Args:
        vqa: visual question answering model.
        data_loader: Iterator for the data.
        criterion: The loss function used to evaluate the loss.
        l2_criterion: The loss function used to evaluate the l2 loss.
        args: ArgumentParser object.

    Returns:
        A float value of average loss.
    """
    vqa.eval()
    total_gen_loss = 0.0
    total_kl = 0.0
    total_recon_image_loss = 0.0
    total_recon_question_loss = 0.0
    total_z_t_kl = 0.0
    total_t_kl_loss = 0.0
    total_steps = len(data_loader)
    if args.eval_steps is not None:
        total_steps = min(len(data_loader), args.eval_steps)
    start_time = time.time()
    for iterations, (images, questions, answers,
                     aindices) in enumerate(data_loader):

        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            questions = questions.cuda()
            answers = answers.cuda()
            aindices = aindices.cuda()
        qlengths = process_lengths(questions)
        qlengths.sort(reverse=True)

        # Forward, Backward and Optimize
        image_features = vqa.encode_images(images)
        question_features = vqa.encode_questions(questions, qlengths)
        mus, logvars = vqa.encode_into_z(image_features, question_features)
        zs = vqa.reparameterize(mus, logvars)
        (outputs, _, other) = vqa.decode_answers(
            image_features, zs, answers=answers,
            teacher_forcing_ratio=1.0)

        # Reorder the questions based on length.
        answers = torch.index_select(answers, 0, aindices)

        # Ignoring the start token.
        answers = answers[:, 1:]
        alengths = process_lengths(answers)
        alengths.sort(reverse=True)

        # Convert the output from MAX_LEN list of (BATCH x VOCAB) ->
        # (BATCH x MAX_LEN x VOCAB).
        outputs = [o.unsqueeze(1) for o in outputs]
        outputs = torch.cat(outputs, dim=1)
        outputs = torch.index_select(outputs, 0, aindices)

        # Calculate the loss.
        targets = pack_padded_sequence(answers, alengths,
                                       batch_first=True)[0]
        outputs = pack_padded_sequence(outputs, alengths,
                                       batch_first=True)[0]
        gen_loss = criterion(outputs, targets)
        total_gen_loss += gen_loss.data.item()

        # Get KL loss if it exists.
        kl_loss = gaussian_KL_loss(mus, logvars)
        total_kl += kl_loss.item()

        # Reconstruction.
        if not args.no_image_recon or not args.no_question_recon:
            image_targets = image_features.detach()
            question_targets = question_features.detach()
            recon_image_features, recon_question_features = vqa.reconstruct_inputs(
                image_targets, question_targets)

            if not args.no_image_recon:
                recon_i_loss = l2_criterion(recon_image_features, image_targets)
                total_recon_image_loss += recon_i_loss.item()
            if not args.no_question_recon:
                recon_q_loss = l2_criterion(recon_question_features, question_targets)
                total_recon_question_loss += recon_q_loss.item()

        # Quit after eval_steps.
        if args.eval_steps is not None and iterations >= args.eval_steps:
            break

        # Print logs
        if iterations % args.log_step == 0:
            delta_time = time.time() - start_time
            start_time = time.time()
            # logging.info('Time: %.4f, Step [%d/%d], gen loss: %.4f, '
            #              'KL: %.4f, I-recon: %.4f, Q-recon: %.4f'
            #              % (delta_time, iterations, total_steps,
            #                 total_gen_loss / (iterations + 1),
            #                 total_kl / (iterations + 1),
            #                 total_recon_image_loss / (iterations + 1),
            #                 total_recon_question_loss / (iterations + 1)))
    total_info_loss = total_recon_image_loss + total_recon_question_loss
    return total_gen_loss / (iterations + 1), total_info_loss / (iterations + 1)


def run_eval(vqa, data_loader, criterion, l2_criterion, args, epoch,
             scheduler, info_scheduler):
    # logging.info('=' * 80)
    start_time = time.time()
    val_gen_loss, val_info_loss = evaluate(
        vqa, data_loader, criterion, l2_criterion, args)
    delta_time = time.time() - start_time
    scheduler.step(val_gen_loss)
    scheduler.step(val_info_loss)
    # logging.info('Time: %.4f, Epoch [%d/%d], Val-gen-loss: %.4f, '
    #              'Val-info-loss: %.4f' % (
    #                  delta_time, epoch, args.num_epochs, val_gen_loss, val_info_loss))
    # logging.info('=' * 80)


def compare_outputs(images, answers, questions,
                    qlengths, vqa, vocab, logging,
                    args, num_show=1):
    """Sanity check generated output as we train.

    Args:
        images: Tensor containing images.
        questions: Tensor containing questions as indices.
        answers: Tensor containing answers as indices.
        alengths: list of answer lengths.
        vqa: A question answering instance.
        vocab: An instance of Vocabulary.
        logging: logging to use to report results.
    """
    vqa.eval()

    # Forward pass through the model.
    outputs = vqa.predict_from_question(images, questions, lengths=qlengths)

    for _ in range(num_show):
        # logging.info("         ")
        i = random.randint(0, images.size(0) - 1)  # Inclusive.

        # Log the outputs.
        output = vocab.tokens_to_words(outputs[i])
        answer = vocab.tokens_to_words(answers[i])
        question = vocab.tokens_to_words(questions[i])
        # logging.info('Sampled answer : %s\n'
        #              'Target  answer (%s) -> %s'
        #              % (output, answer, question))
        # logging.info("         ")


def compute_two_gaussian_loss(mu1, logvar1, mu2, logvar2):
    """Computes the KL loss between the embedding attained from the answers
    and the categories.

    KL divergence between two gaussians:
        log(sigma_2/sigma_1) + (sigma_2^2 + (mu_1 - mu_2)^2)/(2sigma_1^2) - 0.5

    Args:
        mu1: Means from first space.
        logvar1: Log variances from first space.
        mu2: Means from second space.
        logvar2: Means from second space.
    """
    numerator = logvar1.exp() + torch.pow(mu1 - mu2, 2)
    fraction = torch.div(numerator, (logvar2.exp() + 1e-8))
    kl = 0.5 * torch.sum(logvar2 - logvar1 + fraction - 1)
    return kl / (mu1.size(0) + 1e-8)


def evaluate_test(vqa, data_loader, vocab, args, params):
    """Runs BLEU, METEOR, CIDEr and distinct n-gram scores.

    Args:
        vqa: question generation model.
        data_loader: Iterator for the data.
        args: ArgumentParser object.
        params: ArgumentParser object.

    Returns:
        A float value of average loss.
    """
    vqa.eval()
    nlge = NLGEval(no_glove=True, no_skipthoughts=True)
    preds = []
    gts = []
    bar = progressbar.ProgressBar(maxval=len(data_loader))
    bar.start()
    for iterations, (images, questions, answers,
                     _) in enumerate(data_loader):

        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            questions = questions.cuda()
            answers = answers.cuda()
        qlengths = process_lengths(questions)
        qlengths.sort(reverse=True)

        # Predict.
        outputs = vqa.predict_from_question(images, questions, qlengths)

        for i in range(images.size(0)):
            # print (images[i])
            output = vocab.tokens_to_words(outputs[i])
            preds.append(output)

            question = vocab.tokens_to_words(answers[i])
            gts.append(question)
        bar.update(iterations)
    # print('=' * 80)
    # print('GROUND TRUTH')
    # print(gts[:args.num_show])
    # print('-' * 80)
    # print('PREDICTIONS')
    # print(preds[:args.num_show])
    # print('=' * 80)
    scores = nlge.compute_metrics(ref_list=[gts], hyp_list=preds)
    return scores[
        'Bleu_4']  # {'Bleu_1': 0.0, 'Bleu_2': 0.0, 'Bleu_3': 0.0, 'Bleu_4': 0.0, 'METEOR': 0.0, 'ROUGE_L': 0.0, 'CIDEr': 0.0}


def train(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Save the arguments.
    with open(os.path.join(args.model_path, 'args.json'), 'w') as args_file:
        json.dump(args.__dict__, args_file)

    # Config logging.
    log_format = '%(levelname)-8s %(message)s'
    logfile = os.path.join(args.model_path, 'train.log')
    logging.basicConfig(filename=logfile, level=logging.INFO, format=log_format)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(json.dumps(args.__dict__))

    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(args.crop_size,
                                     scale=(1.00, 1.2),
                                     ratio=(0.75, 1.3333333333333333)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    # Load vocabulary wrapper.
    vocab = load_vocab(args.vocab_path)

    # Build data loader
    # logging.info("Building data loader...")
    train_sampler = None
    val_sampler = None
    test_sampler = None
    if os.path.exists(args.train_dataset_weights):
        train_weights = json.load(open(args.train_dataset_weights))
        train_weights = torch.DoubleTensor(train_weights)
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
            train_weights, len(train_weights))
    if os.path.exists(args.val_dataset_weights):
        val_weights = json.load(open(args.val_dataset_weights))
        val_weights = torch.DoubleTensor(val_weights)
        val_sampler = torch.utils.data.sampler.WeightedRandomSampler(
            val_weights, len(val_weights))
    data_loader = get_loader(args.dataset, transform,
                             args.batch_size, shuffle=True,
                             num_workers=args.num_workers,
                             max_examples=args.max_examples,
                             sampler=train_sampler)
    val_data_loader = get_loader(args.val_dataset, transform,
                                 args.batch_size, shuffle=False,
                                 num_workers=args.num_workers,
                                 max_examples=args.max_examples,
                                 sampler=val_sampler)

    test_data_loader = get_loader(args.test_dataset, transform,
                                  args.batch_size, shuffle=False,
                                  num_workers=args.num_workers,
                                  max_examples=args.max_examples,
                                  sampler=test_sampler)
    # logging.info("Done")

    vqa = create_model(args, vocab)

    if args.load_model is not None:
        vqa.load_state_dict(torch.load(args.load_model))
    # logging.info("Done")

    # Loss criterion.
    pad = vocab(vocab.SYM_PAD)  # Set loss weight for 'pad' symbol to 0
    criterion = nn.CrossEntropyLoss(ignore_index=pad)
    l2_criterion = nn.MSELoss()

    # Setup GPUs.
    if torch.cuda.is_available():
        # logging.info("Using available GPU...")
        vqa.cuda()
        criterion.cuda()
        l2_criterion.cuda()

    # Parameters to train.
    gen_params = vqa.generator_parameters()
    info_params = vqa.info_parameters()
    learning_rate = args.learning_rate
    info_learning_rate = args.info_learning_rate
    gen_optimizer = torch.optim.Adam(gen_params, lr=learning_rate)
    info_optimizer = torch.optim.Adam(info_params, lr=info_learning_rate)
    scheduler = ReduceLROnPlateau(optimizer=gen_optimizer, mode='min',
                                  factor=0.1, patience=args.patience,
                                  verbose=False, min_lr=1e-7)
    info_scheduler = ReduceLROnPlateau(optimizer=info_optimizer, mode='min',
                                       factor=0.1, patience=args.patience,
                                       verbose=False, min_lr=1e-7)

    # Train the model.
    total_steps = len(data_loader)
    start_time = time.time()
    n_steps = 0

    # Optional losses. Initialized here for logging.
    recon_question_loss = 0.0
    recon_image_loss = 0.0
    kl_loss = 0.0
    z_t_kl = 0.0
    t_kl = 0.0
    for epoch in range(args.num_epochs):
        for i, (images, questions, answers, aindices) in enumerate(data_loader):
            n_steps += 1

            # Set mini-batch dataset.
            if torch.cuda.is_available():
                images = images.cuda()
                questions = questions.cuda()
                answers = answers.cuda()
                aindices = aindices.cuda()
            qlengths = process_lengths(questions)
            qlengths.sort(reverse=True)

            # Eval now.
            if (args.eval_every_n_steps is not None and
                    n_steps >= args.eval_every_n_steps and
                    n_steps % args.eval_every_n_steps == 0):
                run_eval(vqa, val_data_loader, criterion, l2_criterion,
                         args, epoch, scheduler, info_scheduler)
                compare_outputs(images, answers, questions,
                                qlengths, vqa, vocab, logging, args)

            # Forward.
            vqa.train()
            gen_optimizer.zero_grad()
            info_optimizer.zero_grad()
            image_features = vqa.encode_images(images)
            question_features = vqa.encode_questions(questions, qlengths)

            # Question generation.
            mus, logvars = vqa.encode_into_z(image_features, question_features)
            zs = vqa.reparameterize(mus, logvars)
            (outputs, _, _) = vqa.decode_answers(
                image_features, zs, answers=answers,
                teacher_forcing_ratio=1.0)

            # Reorder the questions based on length.
            answers = torch.index_select(answers, 0, aindices)

            # Ignoring the start token.
            answers = answers[:, 1:]
            alengths = process_lengths(answers)
            alengths.sort(reverse=True)

            # Convert the output from MAX_LEN list of (BATCH x VOCAB) ->
            # (BATCH x MAX_LEN x VOCAB).
            outputs = [o.unsqueeze(1) for o in outputs]
            outputs = torch.cat(outputs, dim=1)
            outputs = torch.index_select(outputs, 0, aindices)

            # Calculate the generation loss.
            targets = pack_padded_sequence(answers, alengths,
                                           batch_first=True)[0]
            outputs = pack_padded_sequence(outputs, alengths,
                                           batch_first=True)[0]
            gen_loss = criterion(outputs, targets)
            total_loss = 0.0
            total_loss += args.lambda_gen * gen_loss
            gen_loss = gen_loss.item()

            # Variational loss.
            kl_loss = gaussian_KL_loss(mus, logvars)
            total_loss += args.lambda_z * kl_loss
            kl_loss = kl_loss.item()

            # Generator Backprop.
            total_loss.backward()
            gen_optimizer.step()

            # Reconstruction loss.
            recon_image_loss = 0.0
            recon_question_loss = 0.0
            if not args.no_question_recon or not args.no_image_recon:
                total_info_loss = 0.0
                gen_optimizer.zero_grad()
                info_optimizer.zero_grad()
                question_targets = question_features.detach()
                image_targets = image_features.detach()

                recon_image_features, recon_question_features = vqa.reconstruct_inputs(
                    image_targets, question_targets)

                # Answer reconstruction loss.
                if not args.no_question_recon:
                    recon_q_loss = l2_criterion(recon_question_features, question_targets)
                    total_info_loss += args.lambda_a * recon_q_loss
                    recon_question_loss = recon_q_loss.item()

                # Image reconstruction loss.
                if not args.no_image_recon:
                    recon_i_loss = l2_criterion(recon_image_features, image_targets)
                    total_info_loss += args.lambda_i * recon_i_loss
                    recon_image_loss = recon_i_loss.item()

                # Info backprop.
                total_info_loss.backward()
                info_optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                delta_time = time.time() - start_time
                start_time = time.time()
                # logging.info('Time: %.4f, Epoch [%d/%d], Step [%d/%d], '
                #              'LR: %f, gen: %.4f, KL: %.4f, '
                #              'I-recon: %.4f, Q-recon: %.4f'
                #              % (delta_time, epoch, args.num_epochs, i,
                #                 total_steps, gen_optimizer.param_groups[0]['lr'],
                #                 gen_loss, kl_loss,
                #                 recon_image_loss, recon_question_loss))

            # Save the models
            if args.save_step is not None and (i + 1) % args.save_step == 0:
                torch.save(vqa.state_dict(),
                           os.path.join(args.model_path,
                                        'vqa-tf-%d-%d.pkl'
                                        % (epoch + 1, i + 1)))

        # torch.save(vqa.state_dict(),
        #            os.path.join(args.model_path,
        #                         'vqa-tf-%d.pkl' % (epoch + 1)))
        torch.save(vqa.state_dict(),
                   args.model_path+'/save.pkl')
        # Evaluation and learning rate updates.
        run_eval(vqa, val_data_loader, criterion, l2_criterion,
                 args, epoch, scheduler, info_scheduler)

    params = Dict2Obj(json.load(
        open(os.path.join(args.model_path, "args.json"), "r")))
    bleu = evaluate_test(vqa, test_data_loader, vocab, args, params)
    print(bleu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Session parameters.
    parser.add_argument('--data_dir', default='/data2/entity/bhy/VQADEMO/dataset/VQA-Med-2019', help='Original dataset position.')
    parser.add_argument('--model-path', type=str, default='weights/tf1/',
                        help='Path for saving trained models')
    parser.add_argument('--crop-size', type=int, default=224,
                        help='Size for randomly cropping images')
    parser.add_argument('--log-step', type=int, default=10,
                        help='Step size for prining log info')
    parser.add_argument('--save-step', type=int, default=None,
                        help='Step size for saving trained models')
    parser.add_argument('--eval-steps', type=int, default=100,
                        help='Number of eval steps to run.')
    parser.add_argument('--eval-every-n-steps', type=int, default=400,
                        help='Run eval after every N steps.')
    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--info-learning-rate', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--max-examples', type=int, default=None,
                        help='For debugging. Limit examples in database.')

    # Lambda values.
    parser.add_argument('--lambda-gen', type=float, default=1.0,
                        help='coefficient to be added in front of the generation loss.')
    parser.add_argument('--lambda-z', type=float, default=0.001,
                        help='coefficient to be added in front of the kl loss.')
    parser.add_argument('--lambda-t', type=float, default=0.0001,
                        help='coefficient to be added with the type space loss.')
    parser.add_argument('--lambda-a', type=float, default=0.001,
                        help='coefficient to be added with the answer recon loss.')
    parser.add_argument('--lambda-i', type=float, default=0.001,
                        help='coefficient to be added with the image recon loss.')
    parser.add_argument('--lambda-z-t', type=float, default=0.001,
                        help='coefficient to be added with the t and z space loss.')

    # Data parameters.
    parser.add_argument('--vocab-path', type=str,
                        default='/data2/entity/bhy/VQADEMO/dataset/VQA-Med-2019/vocab_vqa.json',
                        help='Path for vocabulary wrapper.')
    parser.add_argument('--dataset', type=str,
                        default='/data2/entity/bhy/VQADEMO/dataset/VQA-Med-2019/train_vqa_dataset.hdf5',
                        help='Path for train annotation json file.')
    parser.add_argument('--val-dataset', type=str,
                        default='/data2/entity/bhy/VQADEMO/dataset/VQA-Med-2019/val_vqa_dataset.hdf5',
                        help='Path for train annotation json file.')
    parser.add_argument('--test-dataset', type=str,
                        default='/data2/entity/bhy/VQADEMO/dataset/VQA-Med-2019/test_vqa_dataset.hdf5',
                        help='Location of sampling weights for training set.')
    parser.add_argument('--train-dataset-weights', type=str,
                        default='/data2/entity/bhy/VQADEMO/dataset/VQA-Med-2019/vqa_train_dataset_weights.json',
                        help='Location of sampling weights for training set.')
    parser.add_argument('--val-dataset-weights', type=str,
                        default='/data2/entity/bhy/VQADEMO/dataset/VQA-Med-2019/vqa_val_dataset_weights.json',
                        help='Location of sampling weights for training set.')
    parser.add_argument('--test-dataset-weights', type=str,
                        default='/data2/entity/bhy/VQADEMO/dataset/VQA-Med-2019/vqa_test_dataset_weights.json',
                        help='Location of sampling weights for training set.')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Location of where the model weights are.')

    # Model parameters
    parser.add_argument('--rnn-cell', type=str, default='LSTM',
                        help='Type of rnn cell (GRU, RNN or LSTM).')
    parser.add_argument('--hidden-size', type=int, default=512,
                        help='Dimension of lstm hidden states.')
    parser.add_argument('--num-layers', type=int, default=1,
                        help='Number of layers in lstm.')
    parser.add_argument('--max-length', type=int, default=20,
                        help='Maximum sequence length for outputs.')
    parser.add_argument('--encoder-max-len', type=int, default=20,
                        help='Maximum sequence length for inputs.')
    parser.add_argument('--bidirectional', action='store_true', default=False,
                        help='Boolean whether the RNN is bidirectional.')
    parser.add_argument('--use-glove', action='store_true',
                        help='Whether to use GloVe embeddings.')
    parser.add_argument('--use-w2v', action='store_true',
                        help='Whether to use W2V embeddings.')
    # parser.add_argument('--use-glove', default=False,
    #                     help='Whether to use GloVe embeddings.')
    # parser.add_argument('--use-w2v', default=True,
    #                     help='Whether to use W2V embeddings.')
    parser.add_argument('--embedding-name', type=str, default='PubMed-w2v.txt',
                        help='Name of the GloVe embedding to use. data/processed/PubMed-w2v.txt.pt')
    parser.add_argument('--num-categories', type=int, default=5,
                        help='Number of answer types we use.')
    parser.add_argument('--dropout-p', type=float, default=0.3,
                        help='Dropout applied to the RNN model.')
    parser.add_argument('--input-dropout-p', type=float, default=0.3,
                        help='Dropout applied to inputs of the RNN.')
    parser.add_argument('--num-att-layers', type=int, default=2,
                        help='Number of attention layers.')
    parser.add_argument('--z-size', type=int, default=100,
                        help='Dimensions to use for hidden variational space.')

    # Ablations.
    parser.add_argument('--no-image-recon', action='store_true', default=True,
                        help='Does not try to reconstruct image.')
    parser.add_argument('--no-question-recon', action='store_true', default=True,
                        help='Does not try to reconstruct answer.')
    parser.add_argument('--no-caption-recon', action='store_true', default=False,
                        help='Does not try to reconstruct answer.')
    parser.add_argument('--no-category-space', action='store_true', default=False,
                        help='Does not try to reconstruct answer.')

    # Inputs.
    parser.add_argument('--questions', type=str,
                        default='../dataset/train.json',
                        help='Path for train questions file.')
    parser.add_argument('--questions_val', type=str,
                        default='../dataset/val.json',
                        help='Path for train questions file.')
    parser.add_argument('--questions_test', type=str,
                        default='../dataset/test.json',
                        help='Path for train questions file.')

    # Hyperparameters.
    parser.add_argument('--threshold', type=int, default=4,
                        help='Minimum word count threshold.')

    # Hyperparameters.
    parser.add_argument('--im_size', type=int, default=299,
                        help='Size of images.')
    parser.add_argument('--max-q-length', type=int, default=20,
                        help='maximum sequence length for questions.')
    parser.add_argument('--max-a-length', type=int, default=20,
                        help='maximum sequence length for answers.')
    parser.add_argument('--max-c-length', type=int, default=20,
                        help='maximum sequence length for answers.')

    args = parser.parse_args()
    #process data
    #process txt to json
    process2json(args)
    # create vocab.json
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    vocab = build_vocab(args.data_dir+'/train.json', args.data_dir+'/val.json', args.data_dir+'/test.json', args.threshold)
    # logging.info("Total vocabulary size: %d" % len(vocab))
    vocab.save(args.vocab_path)
    # logging.info("Saved the vocabulary wrapper to '%s'" % args.vocab_path)
    #store_dataset
    save_dataset(args,args.data_dir+'/train', args.data_dir+'/train.json','train',
                 args.vocab_path,args.data_dir+'/train_vqa_dataset.hdf5', im_size=args.im_size,
                 max_q_length=args.max_q_length, max_a_length=args.max_a_length, max_c_length=args.max_c_length)
    # print('Train: Wrote dataset to %s' % args.train_output)
    # Hack to avoid import errors.
    Vocabulary()

    save_dataset(args,args.data_dir+'/val', args.data_dir+'/val.json','val',
                 args.vocab_path,args.data_dir+'/val_vqa_dataset.hdf5', im_size=args.im_size,
                 max_q_length=args.max_q_length, max_a_length=args.max_a_length, max_c_length=args.max_c_length)
    # print('Val: Wrote dataset to %s' % args.val_output)

    save_dataset(args,args.data_dir+'/test', args.data_dir+'/test.json','test',
                 args.vocab_path,args.data_dir+'/test_vqa_dataset.hdf5', im_size=args.im_size,
                 max_q_length=args.max_q_length, max_a_length=args.max_a_length, max_c_length=args.max_c_length)
    # print('Test: Wrote dataset to %s' % args.test_output)
    train(args)
    # Hack to disable errors for importing Vocabulary. Ignore this line.
    Vocabulary()
