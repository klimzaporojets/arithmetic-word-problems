import os
import time

import torch
from tensorboard_logger import log_value
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR

from models.metrics import get_classif_metrics, get_accuracy, get_predicted, get_results_info
from data.corpus.data import DataIterator
from data.dictionary import Dictionary


def execute_validation(model: nn.Module, val_iterator: DataIterator, loss_fn, limit_sigmoid: float, get_details=False,
                       dictionary: Dictionary = None, calculate_nr_order_statistic=False):
    """Executes validation"""

    val_metrics = None
    val_losses = None

    if model.gpu:
        val_gr_metrics = torch.cuda.FloatTensor()
    else:
        val_gr_metrics = torch.Tensor()

    idx = 0
    nr_negatives = 0
    nr_positives = 0
    questions = dict()
    for idx, (
            batch_questions, batch_questions_len, batch_targets_ans, batch_indexes, batch_trees,
            batch_include_in_accuracy) \
            in enumerate(val_iterator):

        questions_torch = torch.from_numpy(batch_questions).long()

        targets_torch = batch_targets_ans
        targets_torch = torch.FloatTensor([targets_torch])
        if model.gpu:
            questions_torch = questions_torch.cuda()
            targets_torch = targets_torch.cuda()

        if get_details:
            if batch_indexes.item() not in questions:
                decoded = dictionary.decode(*questions_torch.tolist())
                decoded_s = ' '.join(decoded)
                questions[batch_indexes] = {'decoded': decoded_s, 'decoded_list': decoded,
                                            'gold_tree': None, 'gold_tree_nr_vars': -1,
                                            'best_predicted': None, 'best_predicted_nr_vars': -1,
                                            'correct': -1, 'sigmoid_score': 0.0,
                                            'only_lower_equal': -1, 'only_higher_equal': -1}

        if batch_trees is not None:
            # preds = model(questions_torch, batch_trees, get_attention_weights=get_attention_weights)
            preds = model(questions_torch, batch_trees)

            predicted = get_predicted(preds, limit_sigmoid)
            loss = loss_fn(preds, targets_torch)

            if batch_targets_ans == 1:
                nr_positives += 1
            else:
                nr_negatives += 1

            if get_details:
                if batch_targets_ans == 1:
                    questions[batch_indexes]['gold_tree'] = str(batch_trees)
                    questions[batch_indexes]['gold_tree_nr_vars'] = len(batch_trees.get_values())

                sigmoid_score = torch.sigmoid(preds).item()
                if sigmoid_score > questions[batch_indexes]['sigmoid_score']:
                    questions[batch_indexes]['sigmoid_score'] = sigmoid_score
                    questions[batch_indexes]['correct'] = batch_targets_ans
                    questions[batch_indexes]['best_predicted'] = str(batch_trees)
                    questions[batch_indexes]['best_predicted_nr_vars'] = len(batch_trees.get_values())
                    # sets to then recover the attention weights, for example
                    questions[batch_indexes]['best_tree'] = batch_trees
                    if calculate_nr_order_statistic:
                        tr_indices = batch_trees.get_indices()
                        low_eq = [tr_indices[i] < tr_indices[i + 1] for i in range(len(tr_indices) - 1)]
                        low_eq_sum = sum(low_eq)
                        high_eq = [tr_indices[i] > tr_indices[i + 1] for i in range(len(tr_indices) - 1)]
                        high_eq_sum = sum(high_eq)
                        only_lower_equal = low_eq_sum / len(low_eq)
                        only_higher_equal = high_eq_sum / len(high_eq)
                        questions[batch_indexes]['only_lower_equal'] = only_lower_equal
                        questions[batch_indexes]['only_higher_equal'] = only_higher_equal

            if batch_include_in_accuracy:
                if model.gpu:
                    val_gr_metrics = torch.cat((val_gr_metrics,
                                                torch.cuda.FloatTensor(
                                                    [[batch_indexes, batch_targets_ans,
                                                      torch.sigmoid(preds).detach().item()]])), dim=0)
                else:
                    val_gr_metrics = torch.cat((val_gr_metrics,
                                                torch.FloatTensor(
                                                    [[batch_indexes, batch_targets_ans,
                                                      torch.sigmoid(preds).detach().item()]])), dim=0)
            else:
                predicted = None
                if model.gpu:
                    val_gr_metrics = torch.cat((val_gr_metrics,
                                                torch.cuda.FloatTensor(
                                                    [[batch_indexes, 0,
                                                      torch.sigmoid(torch.cuda.FloatTensor([99.9])).item()]])),
                                               dim=0)
                else:
                    val_gr_metrics = torch.cat((val_gr_metrics,
                                                torch.FloatTensor(
                                                    [[batch_indexes, 0,
                                                      torch.sigmoid(torch.FloatTensor([99.9])).item()]])),
                                               dim=0)

            if val_losses is None:
                val_losses = loss.detach().cpu()
            else:
                val_losses = torch.cat((val_losses, loss.detach().cpu()), 0)
        else:
            if model.gpu:
                val_gr_metrics = torch.cat((val_gr_metrics,
                                            torch.cuda.FloatTensor(
                                                [[batch_indexes, 0,
                                                  torch.sigmoid(torch.cuda.FloatTensor([99.9])).item()]])), dim=0)
            else:
                val_gr_metrics = torch.cat((val_gr_metrics,
                                            torch.FloatTensor(
                                                [[batch_indexes, 0, torch.sigmoid(torch.FloatTensor([99.9])).item()]])),
                                           dim=0)
            predicted = None

        metrics = get_classif_metrics(predicted, targets_torch)
        if val_metrics is None:
            val_metrics = metrics
        else:
            val_metrics = torch.cat((val_metrics, metrics), dim=0)

    # print('number of data points validation: ', idx, ' positives: ', nr_positives, ' negative: ', nr_negatives)

    if not get_details:
        return {'val_metrics': val_metrics, 'val_gr_metrics': val_gr_metrics, 'val_losses': val_losses}
    else:
        return {'val_metrics': val_metrics, 'val_gr_metrics': val_gr_metrics, 'val_losses': val_losses,
                'details': questions}


def train_module(model: nn.Module, train_iterator: DataIterator, val_iterator: DataIterator, params):
    batch_size = params['batch_size']
    epochs = params['epochs']
    continuous_lr_decrease = params['continuous_lr_decrease']
    learning_rate = params['learning_rate']
    limit_sigmoid = params['limit_sigmoid']
    save_only_last = params['save_only_last']
    variable_lr = params['variable_rate']
    scheduler_step_size = params['scheduler_step_size']
    scheduler_gamma = params['scheduler_gamma']
    save_all_models = params['save_all_models']
    end_learning_rate_ratio = params['end_lr_ratio']

    model_path = os.path.join(params['log_dir'], 'model_data')
    os.makedirs(model_path, exist_ok=True)

    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    length_train = len(train_iterator.targets)
    nr_batches_train = int(length_train / batch_size)
    if nr_batches_train == 0:
        nr_batches_train = 1
    change_per_step = 0.0
    if continuous_lr_decrease:
        total_nr_steps = nr_batches_train * epochs
        end_learning_rate = learning_rate * end_learning_rate_ratio
        change_per_step = (learning_rate - end_learning_rate) / total_nr_steps

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=learning_rate)
    optimizer.zero_grad()

    if variable_lr:
        scheduler = MultiStepLR(optimizer, scheduler_step_size, gamma=scheduler_gamma)

    best_train_loss = 99999.99
    best_train_acc = 0.0

    best_val_loss = 99999.99

    best_val_acc = -0.1

    t0 = time.time()

    for epoch_nr in range(epochs):

        train_losses = None
        train_metrics = None

        if model.gpu:
            train_gr_metrics = torch.cuda.FloatTensor()
        else:
            train_gr_metrics = torch.Tensor()

        model.train()

        cnt_iters = 0

        for batch_questions, batch_questions_len, batch_targets_ans, batch_indexes, batch_trees, \
            batch_include_in_accuracy in train_iterator:

            # ----
            cnt_iters += 1

            questions_torch = torch.from_numpy(batch_questions).long()
            targets_torch = batch_targets_ans
            targets_torch = torch.FloatTensor([targets_torch])
            if model.gpu:
                questions_torch = questions_torch.cuda()
                targets_torch = targets_torch.cuda()

            if batch_trees is not None:
                preds = model(questions_torch, batch_trees)

                if batch_include_in_accuracy:
                    if model.gpu:
                        train_gr_metrics = torch.cat((train_gr_metrics, torch.cuda.FloatTensor(
                            [[batch_indexes, batch_targets_ans, torch.sigmoid(preds).detach().item()]])), dim=0)
                    else:
                        train_gr_metrics = torch.cat((train_gr_metrics,
                                                      torch.FloatTensor(
                                                          [[batch_indexes, batch_targets_ans,
                                                            torch.sigmoid(preds).detach().item()]])), dim=0)
                    predicted = get_predicted(preds, limit_sigmoid)

                else:
                    # for some cases, such as if the equation comes from the solution of the problem, should not
                    # take it into account for accuracy calculation
                    predicted = None
                    if model.gpu:
                        train_gr_metrics = torch.cat((train_gr_metrics,
                                                      torch.cuda.FloatTensor(
                                                          [[batch_indexes, 0, torch.sigmoid(
                                                              torch.cuda.FloatTensor([99.9])).detach().item()]])),
                                                     dim=0)
                    else:
                        train_gr_metrics = torch.cat((train_gr_metrics,
                                                      torch.FloatTensor(
                                                          [[batch_indexes, 0, torch.sigmoid(
                                                              torch.FloatTensor([99.9])).detach().item()]])),
                                                     dim=0)

                loss = loss_fn(preds, targets_torch)

                loss_bck = loss.mean()
                loss_bck.backward()
                if cnt_iters % batch_size == 0 and cnt_iters > 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    if continuous_lr_decrease:
                        learning_rate = learning_rate - change_per_step
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = learning_rate

                if train_losses is None:
                    train_losses = loss.detach().cpu()
                else:
                    train_losses = torch.cat((train_losses, loss.detach().cpu()), 0)
            else:
                # if for some reason the tree could not be extracted (ex: difference between the identified numbers and
                # and the numbers in gold equation, unparseable equations, etc.), then the execution of the model is
                # not possible and predicted is set to None:
                predicted = None
                if model.gpu:
                    train_gr_metrics = torch.cat((train_gr_metrics,
                                                  torch.cuda.FloatTensor(
                                                      [[batch_indexes, 0,
                                                        torch.sigmoid(
                                                            torch.cuda.FloatTensor([99.9])).detach().item()]])), dim=0)
                else:
                    train_gr_metrics = torch.cat((train_gr_metrics,
                                                  torch.FloatTensor(
                                                      [[batch_indexes, 0,
                                                        torch.sigmoid(torch.FloatTensor([99.9])).detach().item()]])),
                                                 dim=0)

            metrics = get_classif_metrics(predicted, targets_torch)

            if train_metrics is None:
                train_metrics = metrics
            else:
                train_metrics = torch.cat((train_metrics, metrics), dim=0)

        if cnt_iters % batch_size != 0:
            optimizer.step()
            optimizer.zero_grad()
            if continuous_lr_decrease:
                learning_rate = learning_rate - change_per_step
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate

        if variable_lr:
            scheduler.step()

        train_loss = train_losses.mean().item()
        if train_loss < best_train_loss:
            best_train_loss = train_loss

        train_gr_accuracy, len_list_res, _, _ = get_results_info(train_gr_metrics)
        print('number of elements in train set for accuracy calculation: ', len_list_res)

        train_accuracy = get_accuracy(train_metrics)

        log_value('train/loss', train_loss, epoch_nr)
        log_value('train/accuracy', train_accuracy, epoch_nr)
        log_value('train/gr_accuracy', train_gr_accuracy, epoch_nr)

        # now executes for validation
        model.eval()

        res_val_stats = execute_validation(model=model, val_iterator=val_iterator, loss_fn=loss_fn,
                                           limit_sigmoid=limit_sigmoid)

        val_accuracy = get_accuracy(res_val_stats['val_metrics'])
        val_gr_accuracy, len_list_res, _, _ = get_results_info(res_val_stats['val_gr_metrics'])

        if val_gr_accuracy > best_val_acc and not save_only_last:
            best_val_acc = val_gr_accuracy
            torch.save(model, os.path.join(model_path, 'model__val_best_acc.dat'))

        if save_all_models:
            torch.save(model, os.path.join(model_path, 'model_ep_{:03d}.dat'.format(epoch_nr)))

        val_loss = res_val_stats['val_losses'].mean().item()

        log_value('val/loss', val_loss, epoch_nr)
        log_value('val/accuracy', val_accuracy, epoch_nr)
        log_value('val/gr_accuracy', val_gr_accuracy, epoch_nr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if train_gr_accuracy > best_train_acc:
            best_train_acc = train_gr_accuracy

        print('ep {}/{} - tr loss {:.4f}  - val loss {:.4f} - b tr loss {:.4f} - b val loss {:.4f} - '
              'tr acc {:.4f} - val acc {:.4f} - b tr acc {:.4f} - b val acc {:.4f} ({:.0f} min) - lr {:.8f}'
              .format(epoch_nr + 1, epochs, train_loss, val_loss, best_train_loss, best_val_loss,
                      train_gr_accuracy, val_gr_accuracy, best_train_acc, best_val_acc, (time.time() - t0) / 60,
                      learning_rate))

    # saves the last model
    torch.save(model, os.path.join(model_path, 'model_last.dat'))
