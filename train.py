import argparse

from tabtransformers import (
    load_config,
    get_model,
    get_loss_function,
    get_lr_scheduler_object,
    get_optimizer_object,
    get_custom_metric,
)
from tabtransformers.tools import (
    seed_everything, 
    train, 
    inference, 
    get_data, 
    get_dataset, 
    get_data_loader, 
    plot_learning_curve, 
    to_submssion_csv
)

def main(args):
    config = load_config(args.config_path)

    seed_everything(config['seed'])

    val_params = config['train_test_split']
    val_params['random_state'] = config['seed']

    train_data, test_data, val_data = \
        get_data('data', split_val=True, 
                 val_params=val_params, 
                 index_col=config['index_col'])
    
    if config['target_mapping'] is not None:
        reverse_target_mapping = {v: k for k, v in config['target_mapping'].items()}
        train_data[config['target_name']] = train_data[config['target_name']].replace(config['target_mapping'])
        val_data[config['target_name']] = val_data[config['target_name']].replace(config['target_mapping'])
    else:
        reverse_target_mapping = None

    train_dataset, test_dataset, val_dataset = \
        get_dataset(
        train_data, test_data, val_data, config['target_name'], 
        config['model_kwargs']['output_dim'], config['categorical_features'], config['continuous_features'])
    
    train_loader, test_loader, val_loader = \
        get_data_loader(
        train_dataset, test_dataset, val_dataset, 
        train_batch_size=config['train_batch_size'], 
        inference_batch_size=config['eval_batch_size'])

    model = get_model(config, vocabulary=train_dataset.get_vocabulary(), num_continuous_features=len(config['continuous_features']))
    
    optimizer = get_optimizer_object(config)(model.parameters(), **config['optim_kwargs'])
    scheduler = get_lr_scheduler_object(config)(optimizer, **config['lr_scheduler_kwargs'])
    
    train_history, val_history = train(
        model, config['epochs'], config['model_kwargs']['output_dim'], train_loader, 
        val_loader, optimizer, get_loss_function(config), 
        scheduler=scheduler, 
        custom_metric=get_custom_metric(config), 
        maximize=config['is_greater_better'], 
        scheduler_custom_metric=config['is_greater_better'] and config['lr_scheduler_by_custom_metric'], 
        early_stopping=config['early_stopping'],
        early_stopping_patience=config['early_stopping_patience'], 
        early_stopping_start_from=config['early_stopping_start_from'])
    
    plot_learning_curve(train_history, val_history)
    predictions = inference(model, test_loader, config['model_kwargs']['output_dim'])
    if reverse_target_mapping is not None:
        predictions = [reverse_target_mapping[p] for p in predictions]
    if config['submission_file']:
        to_submssion_csv(
            predictions, test_data, 
            index_name=None, 
            target_name=config['target_name'], 
            submission_path=config['submission_file'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    main(args)