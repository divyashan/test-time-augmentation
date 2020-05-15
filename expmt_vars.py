import pickle

with open('vars.pickle', 'rb') as handle:
    vars_dict = pickle.load(handle)
    dataset = vars_dict['dataset']
    n_classes = vars_dict['n_classes']
    model_name = vars_dict['model_name']
    batch_size = vars_dict['batch_size']
    aug_order = vars_dict['aug_order_concat'].split(',')
    train_dir = vars_dict['train_dir']
    val_dir = vars_dict['val_dir']
    train_output_dir = vars_dict['train_output_dir']
    val_output_dir = vars_dict['val_output_dir']
    aggregated_outputs_dir = vars_dict['aggregated_outputs_dir']
    results_dir = vars_dict['results_dir']
    tta_policy = vars_dict['tta_policy']
    agg_models_dir = vars_dict['agg_models_dir']

