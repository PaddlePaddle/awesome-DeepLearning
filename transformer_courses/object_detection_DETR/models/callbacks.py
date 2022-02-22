import os
import datetime
import paddle.distributed as dist

from .save_model import save_model
class Callback(object):
    def __init__(self, model):
        self.model = model

    def on_step_begin(self, status):
        pass

    def on_step_end(self, status):
        pass

    def on_epoch_begin(self, status):
        pass

    def on_epoch_end(self, status):
        pass
class ComposeCallback(object):
    def __init__(self, callbacks):
        callbacks = [c for c in list(callbacks) if c is not None]
        for c in callbacks:
            assert isinstance(
                c, Callback), "callback should be subclass of Callback"
        self._callbacks = callbacks

    def on_step_begin(self, status):
        for c in self._callbacks:
            c.on_step_begin(status)

    def on_step_end(self, status):
        for c in self._callbacks:
            c.on_step_end(status)

    def on_epoch_begin(self, status):
        for c in self._callbacks:
            c.on_epoch_begin(status)

    def on_epoch_end(self, status):
        for c in self._callbacks:
            c.on_epoch_end(status)


class LogPrinter(Callback):
    def __init__(self,model, batch_size=2):
        super(LogPrinter, self).__init__(model)
        self.batch_size = batch_size

    def on_step_end(self, status):
        if dist.get_world_size() < 2 or dist.get_rank() == 0:
            mode = status['mode']
            if mode == 'train':
                epoch_id = status['epoch_id']
                step_id = status['step_id']
                steps_per_epoch = status['steps_per_epoch']
                training_staus = status['training_staus']
                batch_time = status['batch_time']
                data_time = status['data_time']

                epoches = 500
                batch_size = self.batch_size             

                logs = training_staus.log()
                space_fmt = ':' + str(len(str(steps_per_epoch))) + 'd'
                if step_id % 20 == 0:
                    eta_steps = (epoches - epoch_id) * steps_per_epoch - step_id
                    eta_sec = eta_steps * batch_time.global_avg
                    eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                    ips = float(batch_size) / batch_time.avg
                    fmt = ' '.join([
                        'Epoch: [{}]',
                        '[{' + space_fmt + '}/{}]',
                        'learning_rate: {lr:.6f}',
                        '{meters}',
                        'eta: {eta}',
                        'batch_cost: {btime}',
                        'data_cost: {dtime}',
                        'ips: {ips:.4f} images/s',
                    ])
                    fmt = fmt.format(
                        epoch_id,
                        step_id,
                        steps_per_epoch,
                        lr=status['learning_rate'],
                        meters=logs,
                        eta=eta_str,
                        btime=str(batch_time),
                        dtime=str(data_time),
                        ips=ips)
                    print(fmt)
            if mode == 'eval':
                step_id = status['step_id']
                if step_id % 100 == 0:
                    print("Eval iter: {}".format(step_id))

    def on_epoch_end(self, status):
        if dist.get_world_size() < 2 or dist.get_rank() == 0:
            mode = status['mode']
            if mode == 'eval':
                sample_num = status['sample_num']
                cost_time = status['cost_time']
                print('Total sample number: {}, averge FPS: {}'.format(
                    sample_num, sample_num / cost_time))


class Checkpointer(Callback):
    def __init__(self, model, optimizers):
        super(Checkpointer, self).__init__(model)
        self.best_ap = 0.
        self.save_dir = 'output'
        self.filename = 'detr_r50_1x_coco'
        self.save_dir = os.path.join(self.save_dir, self.filename)

        self.weight = model
        self.optimizers = optimizers

    def on_epoch_end(self, status):
        # Checkpointer only performed during training
        mode = status['mode']
        epoch_id = status['epoch_id']
        weight = None
        save_name = None
        if dist.get_world_size() < 2 or dist.get_rank() == 0:
            print
            if mode == 'train':
                end_epoch = 500
                
                if (
                        epoch_id + 1
                ) % 1 == 0 or epoch_id == end_epoch - 1:
                    
                    save_name = str(
                        epoch_id) if epoch_id != end_epoch - 1 else "model_final"
                    weight = self.weight
            if weight:
                save_model(weight, self.optimizers, self.save_dir,
                           save_name, epoch_id + 1)