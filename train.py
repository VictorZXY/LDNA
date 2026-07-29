import os
import pickle
import time

import configargparse
import torch
import torch_geometric
import yaml
from configargparse import YAMLConfigFileParser
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch import nn
from torch_geometric.nn.resolver import optimizer_resolver, lr_scheduler_resolver

from utils import Logger, evaluator_resolver, loss_resolver, model_and_data_resolver, tee_to_file
from utils.evaluator import Code2Evaluator


def _eig_kwargs(batch):
    # DGN is the only model that consumes a precomputed Laplacian eigenvector field; the resolver
    # attaches `eig_vec` to the data only for DGN, so its presence is the switch.
    return {'eig_vec': batch.eig_vec} if hasattr(batch, 'eig_vec') else {}


def _nrank_kwargs(batch):
    # Node-level data carries a canonical rank precomputed on the full graph, since a node-level
    # batch is a slice of one graph and the model would otherwise rank the slice. The resolver
    # attaches `nrank` only for LDNA (the only model that reads it), so its presence is the switch,
    # exactly as with `eig_vec` above.
    return {'nrank': batch.nrank} if hasattr(batch, 'nrank') else {}


def _reset_encoders(model):
    # Every model's `reset_parameters()` re-initializes its conv stack, norms, readout and head but
    # NOT the node/edge encoders (`models/encoder.py: Encoder` does not even define the method), so
    # a multi-run job would warm-start each run from the previous run's TRAINED encoder and the runs
    # would not be independent. Re-initializing them here, in the caller, applies the same treatment
    # to LDNA and every baseline without touching a model file. Modules are walked by name so the
    # encoders are found through a wrapper too (`Code2Head` holds them on `.backbone`).
    def reset(module):
        # OGB's molecular encoders are third-party and define no reset method, so the walk below
        # would fall through to their `nn.Embedding` children's N(0, 1) default -- not the Xavier
        # their constructors applied. They are re-initialized here exactly as constructed instead.
        if isinstance(module, (AtomEncoder, BondEncoder)):
            for emb in module.modules():
                if isinstance(emb, nn.Embedding):
                    nn.init.xavier_uniform_(emb.weight.data)
            return
        # A module that defines `reset_parameters` owns its whole subtree's initialization, so the
        # walk stops there. Descending past it would overwrite that choice with each child's own
        # default -- `Encoder`'s Xavier replaced by `nn.Embedding`'s N(0, 1), a 12x rescale of the
        # model's input at ZINC's width.
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
            return
        for child in module.children():
            reset(child)

    for name, module in model.named_modules():
        if name.endswith(('node_encoder', 'edge_encoder')):
            reset(module)


def _lr_scheduler(train_args, optimizer):
    # The LR schedule is optional, because full-graph node-level training runs ONE optimizer step
    # per epoch while a graph-level epoch runs hundreds: an epoch-counted schedule shared between
    # them means two different things, and `ReduceLROnPlateau(patience=10)` -- ten steps at node
    # level -- collapses the LR before training starts. The reference full-graph implementations
    # (OGB's node examples, Platonov et al., tunedGNN) hold the LR constant for a fixed step budget
    # instead, so a node config simply OMITS `scheduler` and gets that.
    #
    # The key must be absent, not `scheduler: null`: configargparse stringifies the nested
    # `train_args` mapping and PyYAML's null resolver does not match `None`, so a YAML null arrives
    # here as the STRING 'None' -- the same trap `readout` carries in utils/resolver.py. That
    # string would resolve to no scheduler by accident rather than by intent, so it is rejected.
    query = train_args.get('scheduler')
    if query == 'None':
        raise ValueError("To train without an LR schedule, OMIT `scheduler` from train_args; a "
                         "YAML `scheduler: null` reaches this point as the string 'None'")
    if not query:
        return None

    scheduler_kwargs = train_args['scheduler_kwargs'] if 'scheduler_kwargs' in train_args else {}
    return lr_scheduler_resolver(query, optimizer, **scheduler_kwargs)


def _split_mask(batch, loader):
    # Node-level loaders tag themselves with the name of the boolean node mask their split uses
    # (train_mask / val_mask / test_mask), so the transductive case is the same `for batch in
    # loader` loop as the graph-level one. Graph-level loaders carry no tag and get a full slice,
    # which indexes to the identical tensor.
    return batch[loader.split] if hasattr(loader, 'split') else slice(None)


@torch.no_grad()
def evaluate(model, loader, evaluator, loss_fn, device):
    model.eval()

    y_true = []
    y_pred = []
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                    **_eig_kwargs(batch), **_nrank_kwargs(batch))

        # The split mask is applied BEFORE the `is_labelled` filter below. That filter is molpcba's
        # missing-label (NaN) guard, NOT a split selector, so leaving the split to it would score
        # and train on every node in the graph regardless of split.
        mask = _split_mask(batch, loader)
        out, y = out[mask], batch.y[mask]

        y_true.append(y)
        y_pred.append(out)
        if loss_fn is not None:
            if isinstance(loss_fn, nn.CrossEntropyLoss):
                is_labelled = (y == y).view(-1)
                loss = loss_fn(out[is_labelled], y[is_labelled].long())
            else:
                is_labelled = (y == y)
                loss = loss_fn(out[is_labelled], y[is_labelled].float())
            total_loss += loss.detach().item()

    result_dict = evaluator.eval({
        'y_true': torch.cat(y_true, dim=0),
        'y_pred': torch.cat(y_pred, dim=0)})
    if loss_fn is not None:
        result_dict['loss'] = total_loss / len(loader)

    return result_dict


def train(model, train_loader, val_loader, test_loader, train_args, device):
    # ogbg-code2 uses a sequence head (per-position CE loss + F1 eval); everything below is the
    # standard single-target path and is left unchanged for all other datasets.
    if hasattr(model, 'idx2vocab'):
        return _train_code2(model, train_loader, val_loader, test_loader, train_args, device)

    model = model.to(device)
    print(f"# Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    loss_kwargs = train_args['loss_kwargs'] if 'loss_kwargs' in train_args else {}
    loss_fn = loss_resolver(train_args['loss_fn'], **loss_kwargs)
    evaluator_kwargs = train_args['evaluator_kwargs'] if 'evaluator_kwargs' in train_args else {}
    evaluator = evaluator_resolver(train_args['evaluator'], **evaluator_kwargs)
    eval_metric = evaluator.eval_metric

    logger = Logger(train_args['runs'], eval_metric)
    run_times = []

    # EXP/CEXP 10-fold CV: the resolver rides the per-fold (train, val, test) loader triples on the
    # train_loader, and each run trains on its own fold (folds map one-to-one to runs).
    cv_folds = getattr(train_loader, 'cv_folds', None)
    if cv_folds is not None and train_args['runs'] != len(cv_folds):
        raise ValueError(f"train_args['runs'] ({train_args['runs']}) must equal the number of "
                         f"CV folds ({len(cv_folds)})")

    for run in range(train_args['runs']):
        if cv_folds is not None:
            train_loader, val_loader, test_loader = cv_folds[run]
        start_time = time.time()
        model.reset_parameters()
        _reset_encoders(model)

        optimizer_kwargs = train_args['optimizer_kwargs'] if 'optimizer_kwargs' in train_args else {}
        optimizer = optimizer_resolver(train_args['optimizer'], model.parameters(), **optimizer_kwargs)
        scheduler = _lr_scheduler(train_args, optimizer)

        for epoch in range(train_args['epochs']):
            model.train()

            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                            **_eig_kwargs(batch), **_nrank_kwargs(batch))

                # As in evaluate(): the split mask must come BEFORE the `is_labelled` NaN guard,
                # which selects labelled rows and not a split. Reusing it alone would put the
                # validation and test labels into the training loss.
                mask = _split_mask(batch, train_loader)
                out, y = out[mask], batch.y[mask]

                if isinstance(loss_fn, nn.CrossEntropyLoss):
                    is_labelled = (y == y).view(-1)
                    loss = loss_fn(out[is_labelled], y[is_labelled].long())
                else:
                    is_labelled = (y == y)
                    loss = loss_fn(out[is_labelled], y[is_labelled].float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.detach().item()

            train_result_dict = evaluate(model, train_loader, evaluator, loss_fn, device)
            val_result_dict = evaluate(model, val_loader, evaluator, loss_fn, device)
            test_result_dict = evaluate(model, test_loader, evaluator, loss_fn, device)
            logger.add_result(run, train_result_dict[eval_metric],
                              val_result_dict[eval_metric],
                              test_result_dict[eval_metric])
            if scheduler is not None:
                scheduler.step(val_result_dict[eval_metric])

            if (epoch + 1) % train_args['eval_interval'] == 0:
                print(f"Run {(run + 1):02d}, "
                      f"Epoch {(epoch + 1):3d}/{train_args['epochs']:3d}, "
                      f"Train loss: {train_result_dict['loss']:.4f}, "
                      f"Val loss: {val_result_dict['loss']:.4f}, "
                      f"Train {eval_metric}: {train_result_dict[eval_metric]:.4f}, "
                      f"Val {eval_metric}: {val_result_dict[eval_metric]:.4f}, "
                      f"Test {eval_metric}: {test_result_dict[eval_metric]:.4f}")

        end_time = time.time()
        run_times.append(end_time - start_time)
        logger.print_statistics(run)

    return model, logger, run_times


@torch.no_grad()
def _evaluate_code2(model, loader, evaluator, criterion, device):
    model.eval()

    seq_ref = []
    seq_pred = []
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        pred_list = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, **_eig_kwargs(batch))

        loss = sum(criterion(pred_list[i].float(), batch.y_arr[:, i])
                   for i in range(len(pred_list))) / len(pred_list)
        total_loss += loss.detach().item()

        mat = torch.stack([p.argmax(dim=1) for p in pred_list], dim=1)
        seq_pred.extend(evaluator.decode(mat))
        seq_ref.extend([batch.y[i] for i in range(len(batch.y))])

    result_dict = {evaluator.eval_metric: evaluator.eval(seq_ref, seq_pred)[evaluator.eval_metric]}
    result_dict['loss'] = total_loss / len(loader)

    return result_dict


def _train_code2(model, train_loader, val_loader, test_loader, train_args, device):
    model = model.to(device)
    print(f"# Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    evaluator = Code2Evaluator(name='ogbg-code2', idx2vocab=model.idx2vocab)
    criterion = nn.CrossEntropyLoss()
    eval_metric = evaluator.eval_metric

    logger = Logger(train_args['runs'], eval_metric)
    run_times = []

    for run in range(train_args['runs']):
        start_time = time.time()
        model.reset_parameters()
        _reset_encoders(model)

        optimizer_kwargs = train_args['optimizer_kwargs'] if 'optimizer_kwargs' in train_args else {}
        optimizer = optimizer_resolver(train_args['optimizer'], model.parameters(), **optimizer_kwargs)
        scheduler = _lr_scheduler(train_args, optimizer)

        for epoch in range(train_args['epochs']):
            model.train()

            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                pred_list = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, **_eig_kwargs(batch))
                loss = sum(criterion(pred_list[i].float(), batch.y_arr[:, i])
                           for i in range(len(pred_list))) / len(pred_list)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.detach().item()

            train_result_dict = _evaluate_code2(model, train_loader, evaluator, criterion, device)
            val_result_dict = _evaluate_code2(model, val_loader, evaluator, criterion, device)
            test_result_dict = _evaluate_code2(model, test_loader, evaluator, criterion, device)
            logger.add_result(run, train_result_dict[eval_metric],
                              val_result_dict[eval_metric],
                              test_result_dict[eval_metric])
            if scheduler is not None:
                scheduler.step(val_result_dict[eval_metric])

            if (epoch + 1) % train_args['eval_interval'] == 0:
                print(f"Run {(run + 1):02d}, "
                      f"Epoch {(epoch + 1):3d}/{train_args['epochs']:3d}, "
                      f"Train loss: {train_result_dict['loss']:.4f}, "
                      f"Val loss: {val_result_dict['loss']:.4f}, "
                      f"Train {eval_metric}: {train_result_dict[eval_metric]:.4f}, "
                      f"Val {eval_metric}: {val_result_dict[eval_metric]:.4f}, "
                      f"Test {eval_metric}: {test_result_dict[eval_metric]:.4f}")

        end_time = time.time()
        run_times.append(end_time - start_time)
        logger.print_statistics(run)

    return model, logger, run_times


def main(args):
    log_path = os.path.join(args.log_dir, f'{args.experiment_name}.txt') if args.log_dir is not None else None
    with tee_to_file(log_path):
        if args.cuda != -1:
            device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
        print(f"Device: {device}")

        # Cap this process's share of the (shared) GPU so a ranking run can't OOM a
        # co-located job (another user's, or another of ours). See § GPU policy.
        if device.type == 'cuda':
            torch.cuda.set_per_process_memory_fraction(args.mem_fraction, device.index)

        model, train_loader, val_loader, test_loader = model_and_data_resolver(
            args.model, args.dataset, model_args=(args.model_args or {}), data_args=(args.data_args or {})
        )
        print(f"Dataset name: {args.dataset}")
        print(f"Model name: {args.model}")

        model, logger, run_times = train(model, train_loader, val_loader, test_loader, args.train_args, device)
        print(f"Finished Training {args.model} on {args.dataset} dataset")
        logger.print_statistics()
        avg_run_time = sum(run_times) / len(run_times)
        if avg_run_time >= 3600:
            print(f"Average run time: {(avg_run_time / 3600):.3f} hours")
        elif avg_run_time >= 60:
            print(f"Average run time: {(avg_run_time / 60):.3f} minutes")
        else:
            print(f"Average run time: {avg_run_time:.3f} seconds")

        if args.checkpoint_dir is not None:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f'{args.experiment_name}.pt'))
        if args.log_dir is not None:
            with open(os.path.join(args.log_dir, f'{args.experiment_name}_logger.pickle'), 'wb') as f:
                pickle.dump(logger, f)


if __name__ == '__main__':
    parser = configargparse.ArgParser(config_file_parser_class=YAMLConfigFileParser)
    parser.add_argument('--config', is_config_file=True)
    parser.add_argument('--experiment_name', type=str, required=True)

    # Model specific arguments/hyperparameters
    parser.add_argument('--model', default='LDNA', type=str)
    parser.add_argument('--model_args', default=None, type=yaml.safe_load)

    # Dataset specific arguments/hyperparameters
    parser.add_argument('--dataset', default='ogbg-molpcba', type=str)
    parser.add_argument('--data_args', default=None, type=yaml.safe_load)

    # Training specific arguments/hyperparameters
    parser.add_argument('--cuda', default=-1, type=int)
    parser.add_argument('--mem_fraction', default=0.30, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--train_args', default=None, type=yaml.safe_load)
    parser.add_argument('--checkpoint_dir', default='out/checkpoints/', type=str)
    parser.add_argument('--log_dir', default='out/logs/', type=str)

    args = parser.parse_args()
    torch_geometric.seed_everything(args.seed)

    main(args)
