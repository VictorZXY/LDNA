import os
import pickle
import time

import configargparse
import torch
import torch_geometric
import yaml
from configargparse import YAMLConfigFileParser
from torch import nn
from torch_geometric.nn.resolver import optimizer_resolver, lr_scheduler_resolver

from utils import Logger, evaluator_resolver, loss_resolver, model_and_data_resolver, tee_to_file


@torch.no_grad()
def evaluate(model, loader, evaluator, loss_fn, device):
    model.eval()

    y_true = []
    y_pred = []
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y)
        y_pred.append(out)
        if loss_fn is not None:
            if isinstance(loss_fn, nn.CrossEntropyLoss):
                is_labelled = (batch.y == batch.y).view(-1)
                loss = loss_fn(out[is_labelled], batch.y[is_labelled].long())
            else: 
                is_labelled = (batch.y == batch.y)
                loss = loss_fn(out[is_labelled], batch.y[is_labelled].float())
            total_loss += loss.detach().item()

    result_dict = evaluator.eval({
        'y_true': torch.cat(y_true, dim=0),
        'y_pred': torch.cat(y_pred, dim=0)})
    if loss_fn is not None:
        result_dict['loss'] = total_loss / len(loader)

    return result_dict


def train(model, train_loader, val_loader, test_loader, train_args, device):
    model = model.to(device)
    print(f"# Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    loss_kwargs = train_args['loss_kwargs'] if 'loss_kwargs' in train_args else {}
    loss_fn = loss_resolver(train_args['loss_fn'], **loss_kwargs)
    evaluator_kwargs = train_args['evaluator_kwargs'] if 'evaluator_kwargs' in train_args else {}
    evaluator = evaluator_resolver(train_args['evaluator'], **evaluator_kwargs)
    eval_metric = evaluator.eval_metric

    logger = Logger(train_args['runs'], eval_metric)
    run_times = []

    for run in range(train_args['runs']):
        start_time = time.time()
        model.reset_parameters()

        optimizer_kwargs = train_args['optimizer_kwargs'] if 'optimizer_kwargs' in train_args else {}
        optimizer = optimizer_resolver(train_args['optimizer'], model.parameters(), **optimizer_kwargs)
        scheduler_kwargs = train_args['scheduler_kwargs'] if 'scheduler_kwargs' in train_args else {}
        scheduler = lr_scheduler_resolver(train_args['scheduler'], optimizer, **scheduler_kwargs)

        for epoch in range(train_args['epochs']):
            model.train()

            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                if isinstance(loss_fn, nn.CrossEntropyLoss):
                    is_labelled = (batch.y == batch.y).view(-1)
                    loss = loss_fn(out[is_labelled], batch.y[is_labelled].long())
                else: 
                    is_labelled = (batch.y == batch.y)
                    loss = loss_fn(out[is_labelled], batch.y[is_labelled].float())
                loss.backward()
                optimizer.step()
                total_loss += loss.detach().item()

            train_result_dict = evaluate(model, train_loader, evaluator, loss_fn, device)
            val_result_dict = evaluate(model, val_loader, evaluator, loss_fn, device)
            test_result_dict = evaluate(model, test_loader, evaluator, loss_fn, device)
            logger.add_result(run, train_result_dict[eval_metric],
                              val_result_dict[eval_metric],
                              test_result_dict[eval_metric])
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
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--train_args', default=None, type=yaml.safe_load)
    parser.add_argument('--checkpoint_dir', default='checkpoints/', type=str)
    parser.add_argument('--log_dir', default='logs/', type=str)

    args = parser.parse_args()
    torch_geometric.seed_everything(args.seed)

    main(args)
