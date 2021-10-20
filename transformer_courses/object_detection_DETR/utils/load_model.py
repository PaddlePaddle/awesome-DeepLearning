import os
import numpy as np
import paddle

def match_state_dict(model_state_dict, weight_state_dict):
    """
    Match between the model state dict and pretrained weight state dict.
    Return the matched state dict.

    The method supposes that all the names in pretrained weight state dict are
    subclass of the names in models`, if the prefix 'backbone.' in pretrained weight
    keys is stripped. And we could get the candidates for each model key. Then we 
    select the name with the longest matched size as the final match result. For
    example, the model state dict has the name of 
    'backbone.res2.res2a.branch2a.conv.weight' and the pretrained weight as
    name of 'res2.res2a.branch2a.conv.weight' and 'branch2a.conv.weight'. We
    match the 'res2.res2a.branch2a.conv.weight' to the model key.
    """

    model_keys = sorted(model_state_dict.keys())
    weight_keys = sorted(weight_state_dict.keys())

    def match(a, b):
        if a.startswith('backbone.res5'):
            # In Faster RCNN, res5 pretrained weights have prefix of backbone, 
            # however, the corresponding model weights have difficult prefix,
            # bbox_head.
            b = b[9:]
        return a == b or a.endswith("." + b)

    match_matrix = np.zeros([len(model_keys), len(weight_keys)])
    for i, m_k in enumerate(model_keys):
        for j, w_k in enumerate(weight_keys):
            if match(m_k, w_k):
                match_matrix[i, j] = len(w_k)
    max_id = match_matrix.argmax(1)
    max_len = match_matrix.max(1)
    max_id[max_len == 0] = -1
    matched_keys = {}
    result_state_dict = {}
    for model_id, weight_id in enumerate(max_id):
        if weight_id == -1:
            continue
        model_key = model_keys[model_id]
        weight_key = weight_keys[weight_id]
        weight_value = weight_state_dict[weight_key]
        model_value_shape = list(model_state_dict[model_key].shape)

        if list(weight_value.shape) != model_value_shape:
            print(
                'The shape {} in pretrained weight {} is unmatched with '
                'the shape {} in model {}. And the weight {} will not be '
                'loaded'.format(weight_value.shape, weight_key,
                                model_value_shape, model_key, weight_key))
            continue

        assert model_key not in result_state_dict
        result_state_dict[model_key] = weight_value
        if weight_key in matched_keys:
            raise ValueError('Ambiguity weight {} loaded, it matches at least '
                             '{} and {} in the model'.format(
                                 weight_key, model_key, matched_keys[
                                     weight_key]))
        matched_keys[weight_key] = model_key
    return result_state_dict

def _strip_postfix(path):
    path, ext = os.path.splitext(path)
    assert ext in ['', '.pdparams', '.pdopt', '.pdmodel'], \
            "Unknown postfix {} from weights".format(ext)
    return path

def load_pretrain_weight(model, pretrain_weight):
    path = _strip_postfix(pretrain_weight)
    if not (os.path.isdir(path) or os.path.isfile(path) or
            os.path.exists(path + '.pdparams')):
        raise ValueError("Model pretrain path `{}` does not exists. "
                         "If you don't want to load pretrain model, "
                         "please delete `pretrain_weights` field in "
                         "config file.".format(path))

    model_dict = model.state_dict()

    weights_path = path + '.pdparams'
    param_state_dict = paddle.load(weights_path)
    param_state_dict = match_state_dict(model_dict, param_state_dict)

    model.set_dict(param_state_dict)
    print('Finish loading model weights: {}'.format(weights_path))

def load_weights(model, weights):
    start_epoch = 0
    load_pretrain_weight(model, weights)
    print("Load weights {} to start training".format(weights))