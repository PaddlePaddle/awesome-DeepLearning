import os
import paddle

def save_model(model, optimizers, save_dir, save_name, last_epoch):
    """
    save model into disk.

    Args:
        model (paddle.nn.Layer): the Layer instalce to save parameters.
        optimizers (paddle.optimizer.Optimizer): the Optimizer instance to
            save optimizer states.
        save_dir (str): the directory to be saved.
        save_name (str): the path to be saved.
        last_epoch (int): the epoch index.
    """
    if paddle.distributed.get_rank() != 0:
        return
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_name)
    if isinstance(model, nn.Layer):
        paddle.save(model.state_dict(), save_path + ".pdparams")
    else:
        assert isinstance(model,
                          dict), 'model is not a instance of nn.layer or dict'
        paddle.save(model, save_path + ".pdparams")
    state_dict = optimizers.state_dict()
    state_dict['last_epoch'] = last_epoch
    paddle.save(state_dict, save_path + ".pdopt")
    print("Save checkpoint: {}".format(save_dir))