#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import six
if six.PY2:
    import cPickle as pickle
    import Queue
else:
    import pickle
    import queue as Queue

from collections import OrderedDict, Iterable
import numpy as np
import copy
import multiprocessing
from multiprocessing.managers import BaseManager
from threading import Thread

import paddle.fluid as fluid

from paddleslim.pantheon.utils import convert_dtype, EndSignal, SyncSignal, StartSignal, public_authkey

__all__ = ["Teacher"]

# Num of threads for post-processing, including generating and transferring 
# knowledge data
num_postprocess_threads = int(os.getenv("NUM_POSTPROCESS_THREADS", 8))
knowledge_queues = [Queue.Queue(100) for i in range(num_postprocess_threads)]

t2s_queue = Queue.Queue(100)
s2t_queue = Queue.Queue(100)
cmd_queue = Queue.Queue(5)


class MixedDataReader(object):
    """ 
    The wrapper for iterable data loader, to solve the drop problem of last 
    batches when their number is less than the number of devices in prediction. 
    It implements two data generators, one for the prediction on all devices, 
    and another one for the prediction of remained data one single device, and 
    they two should be called in order.

    Args:
        data_loader (fluid.io.DataLoader): The data loader.
        base_number (int): The base number that the number of yielded data 
                           batches for multiple devices should be its 
                           multiple times.
    """

    def __init__(self, data_loader, base_number):
        self._data_loader = data_loader
        self._base_number = base_number
        self._tail_data = []

    def multi_dev_generator(self):
        for data in self._data_loader():
            if len(self._tail_data) < self._base_number:
                self._tail_data += data
            if len(self._tail_data) == self._base_number:
                yield self._tail_data
                self._tail_data = []

    def tail_generator(self):
        for data in self._tail_data:
            yield data
        self._tail_data = []


class WorkerParallel(object):
    """
    Process data from the input queue by given worker in parallel, and put the 
    result into output queue in order.

    Args:
        num_postprocess_threads (int): Number of threads for data processing.
        in_queue (object): The input queue.
        out_queue (object|list): The output queue(s). Its length should be equal 
            to arg 'num_postprocess_threads' when it is a list.
    """

    def __init__(self, num_postprocess_threads, in_queue, out_queue):
        self._num_postprocess_threads = num_postprocess_threads
        self._in_queue = in_queue
        self._local_in_queues = [
            Queue.Queue(5) for i in range(num_postprocess_threads)
        ]
        if isinstance(out_queue, list):
            if len(out_queue) != num_postprocess_threads:
                raise ValueError("When out_queue is a list, its length must "
                                 "equal to num_postprocess_threads!")
            self._local_out_queues = out_queue
            self._out_queue = None
        else:
            self._local_out_queues = [
                Queue.Queue(5) for i in range(num_postprocess_threads)
            ]
            self._out_queue = out_queue

    def _distribute(self):
        def func():
            idx = 0
            while True:
                data = self._in_queue.get()
                self._in_queue.task_done()
                if not isinstance(data, EndSignal):
                    self._local_in_queues[
                        idx % self._num_postprocess_threads].put(data)
                    idx += 1
                else:
                    for q in self._local_in_queues:
                        q.put(EndSignal())
                    break

        t = Thread(target=func)
        t.daemon = True
        t.start()

    def _run(self, worker, args):
        for i in range(self._num_postprocess_threads):
            t = Thread(
                target=worker,
                args=(self._local_in_queues[i], self._local_out_queues[i]) +
                args)
            t.daemon = True
            t.start()

    def _gather(self):
        def func():
            end_received = False
            while True:
                for idx, q in enumerate(self._local_out_queues):
                    data = q.get()
                    q.task_done()
                    if isinstance(data, EndSignal):
                        end_received = True
                        if idx > 0:
                            continue
                    self._out_queue.put(data)
                if end_received:
                    break

        t = Thread(target=func)
        t.daemon = True
        t.start()

    def __call__(self, worker, args):
        self._distribute()
        self._run(worker, args)
        if self._out_queue:
            self._gather()


class Teacher(object):
    """
    The class defined for the teacher model. Generate knowledge data and 
    transfer them to the student model.

    Args:
        out_path (str|None): The path to dump knowledge for offline mode.
        out_port (int|None): The IP port number to send out knowledge for 
            online mode, should be unique when launching multiple teachers in 
            the same node.
    """

    def __init__(self, out_path=None, out_port=None):
        if out_path and out_port:
            raise ValueError("Out path and out port should not be set at "
                             "the same time!")

        self._out_path = out_path
        self._out_port = out_port
        # knowledge description
        self._knowledge_desc = {}

        self._sync_required = False
        self._data_required = False
        self._started = False

    def _start_manager(self):
        def get_knowledge_queue(idx):
            global knowledge_queues
            if idx < len(knowledge_queues):
                return knowledge_queues[idx]
            else:
                return None

        def get_s2t_queue():
            global s2t_queue
            return s2t_queue

        def get_t2s_queue():
            global t2s_queue
            return t2s_queue

        def get_cmd_queue():
            global cmd_queue
            return cmd_queue

        BaseManager.register(
            "get_knowledge_queue", callable=get_knowledge_queue)
        BaseManager.register("get_s2t_queue", callable=get_s2t_queue)
        BaseManager.register("get_t2s_queue", callable=get_t2s_queue)
        BaseManager.register("get_cmd_queue", callable=get_cmd_queue)
        manager = BaseManager(
            address=("", self._out_port), authkey=public_authkey.encode())
        manager.start()
        print("listen on address: {}".format(manager._address))
        print("public authkey: {}".format(public_authkey))
        return manager

    def start(self):
        """ 
        Start teacher service, sychronize with student and launch the thread 
        to monitor commands from student. 
        """
        if self._started:
            raise ValueError(
                "The teacher cannot be started more than one time.")
        self._started = True
        self._manager = self._start_manager() if self._out_port else None
        if self._manager:
            self._knowledge_queues = [
                self._manager.get_knowledge_queue(i)
                for i in range(num_postprocess_threads)
            ]
            print("Num of knowledge queues: {}".format(num_postprocess_threads))
            self._s2t_queue = self._manager.get_s2t_queue()
            self._t2s_queue = self._manager.get_t2s_queue()
            self._cmd_queue = self._manager.get_cmd_queue()
        else:
            self._knowledge_queues = None
            self._s2t_queue = None
            self._t2s_queue = None
            self._cmd_queue = None

        self._out_file = open(self._out_path, "wb") if self._out_path else None
        if self._out_file:
            return

        def wrapper():
            while True:
                if not self._cmd_queue.empty():
                    cmd = self._cmd_queue.get()
                    self._cmd_queue.task_done()
                    if isinstance(cmd, SyncSignal):
                        self._sync_required = True
                    elif isinstance(cmd, StartSignal):
                        self._data_required = True
                else:
                    time.sleep(1.0)

        t = Thread(target=wrapper)
        t.daemon = True
        t.start()

        while True:
            if self._sync_required:
                for q in self._knowledge_queues:
                    q.put(SyncSignal())
                    q.join()
                self._sync_required = False
                break

    def send(self, data):
        """
        Send one data object to student.
        
        Args:
            data (Python data): The data to be sent, can be any type of Python data object. 
        """
        if not self._started:
            raise ValueError("The method start() should be called first!")

        if not self._t2s_queue:
            raise ValueError("Cannot send data to stuent for this teacher "
                             "is offline!")
        self._t2s_queue.put(data)

    def recv(self):
        """
        Recieve one data object from student. 

        Return:
            The received data, can be any type of Python data object.
        """
        if not self._started:
            raise ValueError("The method start() should be called first!")

        if not self._s2t_queue:
            raise ValueError("Cannot receive data from stuent for this teacher "
                             "is in offline mode!")
        data = self._s2t_queue.get()
        self._s2t_queue.task_done()
        return data

    def dump(self, knowledge):
        """
        Dump one batch knowledge data into output file, only used in the 
        offline mode.

        Args:
            knowledge (dict): The knowledge data to be dumped.  
        """
        if not self._started:
            raise ValueError("The method start() should be called first!")

        if not self._out_file:
            raise ValueError("Cannot dump knowledge data in online mode!")

        if not isinstance(knowledge, dict) and not isinstance(knowledge,
                                                              OrderedDict):
            raise ValueError(
                "The knowledge data should be a dict or OrderedDict!")

        knowledge_desc = {}
        for name, value in list(knowledge.items()):
            knowledge_desc[name] = {
                "shape": [-1] + list(value.shape[1:]),
                "dtype": str(value.dtype),
                "lod_level": 0
            }
        if not self._knowledge_desc:
            self._knowledge_desc = knowledge_desc
            self._out_file.write(pickle.dumps(self._knowledge_desc))
        else:
            if self._knowledge_desc != knowledge_desc:
                raise ValueError("Current knowledge desc {} is not the same as "
                                 "historic desc {}!".format(
                                     knowledge_desc, self._knowledge_desc))

        self._out_file.write(pickle.dumps(knowledge))

    def start_knowledge_service(self,
                                feed_list,
                                schema,
                                program,
                                reader_config,
                                exe,
                                buf_size=10,
                                use_fp16=False,
                                times=1):
        """
        Start the knowledge service to generate and transfer knowledge data.
        In GPU mode, the devices to execute knowledge prediction will be 
        determined by environment variable **FLAGS_selected_gpus**, or by 
        **CUDA_VISIBLE_DEVICES** if it is not set, and by **CPU_NUM** (default 
        1) in CPU mode. Only supported in static graph. 

        Args:
            feed_list (list): A list of feed Variables or their names for the 
                              input program.
            schema (dict): A dictionary to specify names and fetched 
                           Variables of knowledge.
            program (fluid.Program): Inference program for the teacher model.
            reader_config (dict): The config for data reader. Support all the 
                three types of generators used by `fluid.io.PyReader` and 
                `fluid.io.DataLoader`, and their configs contain the key-value 
                pair of the generator type and a generator object, plus
                other necessary argument pairs. See the following: 

                    1) sample generator:
                       reader_config={"sample_generator": #some_sample_generator, 
                                  "batch_size": #batch_size, "drop_last": #drop_last},
                       'drop_last' set to True by default, 
                    2) sample list generator:
                       reader_config={"sample_list_generator": 
                                       #some_sample_list_generator},
                    3) batch generator:
                       reader_config={"batch_generator": #some_batch_genrator}.

                The trial to parse config will be in the order of 1) -> 3), and 
                any other unrelated keys in these configs will be ignored.
            exe (fluid.Executor): The executor to run the input program.
            buf_size (int): The size of buffers for data reader and knowledge 
                            writer on each device. 
            use_fp16 (bool): Whether to transfer/store knowledge data in float16 
                         if their data type is float32/float64. In the offline 
                         mode, it will reduce the size of dumped knowledge file, 
                         and in the online mode, it will speedup the online 
                         transfer, with the sacrifice in precision . Default False.
            times (int): The maximum repeated serving times. Default 1. Whenever 
                         the public method 'get_knowledge_generator()' in Student 
                         object called once, the serving times will be added one, 
                         until reaching the maximum and ending the service. Only 
                         valid in online mode, and will be ignored in offline mode.
        """
        if not self._started:
            raise ValueError("The method start() should be called first!")

        if not isinstance(program, fluid.Program):
            raise ValueError(
                "Input argument 'program' should be a fluid Program!")
        self._program = program._inference_optimize(prune_read_op=True)

        if not isinstance(feed_list, list):
            raise ValueError("Input argument 'feed_list' should be a list!")
        else:
            self._feed_list = []
            for feed in feed_list:
                if isinstance(feed, fluid.framework.Variable):
                    self._feed_list.append(feed)
                elif isinstance(feed, str) or isinstance(feed, unicode):
                    self._feed_list.append(self._program.global_block().var(
                        feed))
                else:
                    raise ValueError("Input 'feed_list' should consist of feed "
                                     "Variables or their names!")

        if not isinstance(schema, dict) and not isinstance(schema, OrderedDict):
            raise ValueError(
                "Input argument 'schema' should be a dict or OrderedDict!")
        self._schema = schema

        if not isinstance(reader_config, dict):
            raise ValueError("The reader config must be a dictionary!")

        if not isinstance(exe, fluid.Executor):
            raise ValueError("Input argument should be a fluid Executor!")
        self._exe = exe

        self._use_fp16 = use_fp16

        if not buf_size > 0:
            raise ValueError("The buffer size should be positive!")
        self._buf_size = buf_size

        if not times > 0:
            raise ValueError("Repeated serving times should be positive!")
        self._times = times
        if self._times > 1 and self._out_file:
            self._times = 1
            print("WARNING: args 'times' will be ignored in offline mode")

        desc = {}
        for name, var in list(schema.items()):
            if not isinstance(var, fluid.framework.Variable):
                raise ValueError("The member of schema must be fluid Variable.")
            desc[name] = {
                "shape": var.shape,
                "dtype": convert_dtype(var.dtype),
                "lod_level": var.lod_level
            }
        if not self._knowledge_desc:
            self._knowledge_desc = desc
        else:
            if self._out_file and not self._knowledge_desc == desc:
                raise ValueError("The knowledge description should be kept "
                                 "consistent in offline mode!")

        if isinstance(self._exe.place, fluid.CUDAPlace):
            places = fluid.cuda_places()
        else:
            places = fluid.cpu_places()
        dev_count = len(places)

        data_loader = fluid.io.DataLoader.from_generator(
            feed_list=self._feed_list,
            capacity=self._buf_size * dev_count,
            use_double_buffer=(dev_count == 1),
            iterable=True)

        places = [fluid.CPUPlace()] if dev_count > 1 else [self._exe.place]
        if "sample_generator" in reader_config:
            if "batch_size" not in reader_config:
                raise ValueError("batch size must be specified when using "
                                 "sample generator!")
            sample_generator = reader_config["sample_generator"]
            batch_size = reader_config["batch_size"]
            drop_last = reader_config[
                "drop_last"] if "drop_last" in reader_config else True

            data_loader.set_sample_generator(
                reader=sample_generator,
                batch_size=batch_size,
                drop_last=drop_last,
                places=places)
        elif "sample_list_generator" in reader_config:
            sample_list_generator = reader_config["sample_list_generator"]
            data_loader.set_sample_list_generator(
                reader=sample_list_generator, places=places)
        elif "batch_generator" in reader_config:
            batch_generator = reader_config["batch_generator"]
            data_loader.set_batch_generator(
                reader=batch_generator, places=places)
        else:
            raise ValueError(
                "The reader config doesn't contain any valid "
                "generator type, which should be one of 'sample_generator', "
                "'sample_list_generator', and 'batch_generator'.")

        def cast2fp16(know):
            for k, v in list(know.items()):
                if not isinstance(v, np.ndarray):
                    break
                if v.dtype == np.float32 or v.dtype == np.float64:
                    v = v.astype("float16")
                    know[k] = v
            return know

        feed_var_names = [var.name for var in self._feed_list]
        schema_in_feed, schema_in_fetch = {}, {}
        for k, v in list(self._schema.items()):
            if k in feed_var_names:
                schema_in_feed[k] = v
            else:
                schema_in_fetch[k] = v
        schema_in_fetch_keys, schema_in_fetch_vars = zip(
            *list(schema_in_fetch.items()))

        def know_maker(in_queue, out_queue, use_fp16):
            while True:
                data = in_queue.get()
                in_queue.task_done()
                if isinstance(data, tuple):
                    dev_batches, outputs = data
                    know = {}
                    for k in schema_in_feed.keys():
                        batch_know = [
                            np.array(batch[k]) for batch in dev_batches
                        ]
                        know[k] = np.concatenate(batch_know)
                    know.update(dict(zip(schema_in_fetch_keys, outputs)))
                    if use_fp16:
                        know = cast2fp16(know)
                    out_queue.put(know)
                else:
                    # forward other types of data directly (maybe knowledge desc or EndSignal)
                    out_queue.put(data)
                    if isinstance(data, EndSignal):
                        break

        know_make_queue = Queue.Queue(self._buf_size)
        if self._out_file:
            # For offline dump, write the knowledge description to the head of file
            self._out_file.write(pickle.dumps(self._knowledge_desc))
            print("output path: %s" % self._out_path)
            offline_write_queue = Queue.Queue(self._buf_size)

            def offline_write(queue):
                while True:
                    know = queue.get()
                    queue.task_done()
                    if not isinstance(know, EndSignal):
                        self._out_file.write(pickle.dumps(know))
                    else:
                        # should close file in child thread to wait for all 
                        # writing finished
                        self._out_file.close()

            t = Thread(target=offline_write, args=(offline_write_queue, ))
            t.daemon = True
            t.start()
            make_knowledge = WorkerParallel(
                num_postprocess_threads, know_make_queue, offline_write_queue)

        if self._knowledge_queues:
            make_knowledge = WorkerParallel(num_postprocess_threads,
                                            know_make_queue,
                                            self._knowledge_queues)

        compiled_program = fluid.compiler.CompiledProgram(
            self._program).with_data_parallel()

        print("Knowledge description {}".format(self._knowledge_desc))
        print(
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) +
            "  Teacher begins to serve ...")

        data_reader = MixedDataReader(data_loader, dev_count)
        for repeated in range(self._times):
            make_knowledge(worker=know_maker, args=(self._use_fp16, ))
            if self._knowledge_queues:
                # wait for the accessing of knowledge desc and data
                while True:
                    if self._sync_required:
                        for q in self._knowledge_queues:
                            q.put(SyncSignal())
                        # For online mode, send knowledge description every sync
                        know_make_queue.put(self._knowledge_desc)
                        self._sync_required = False
                    if self._data_required:
                        self._data_required = False
                        break
                for q in self._knowledge_queues:
                    q.join()

            print("No.{} time serving ... ".format(repeated))
            num_batches_sent = 0
            for index, dev_batches in enumerate(data_reader.multi_dev_generator(
            )):
                if self._sync_required:
                    break
                outputs = self._exe.run(compiled_program,
                                        feed=dev_batches,
                                        fetch_list=schema_in_fetch_vars)
                know_make_queue.put((dev_batches, outputs))

                num_batches_sent += dev_count
                if num_batches_sent % (100 * dev_count) == 0:
                    log = "Processed {} batch samples.".format(num_batches_sent)
                    if self._knowledge_queues:
                        qsize = 0
                        for q in self._knowledge_queues:
                            qsize += q.qsize()
                        log += " Knowledge queue size {}.".format(qsize)
                    print(log)

            dev_batches, outputs = [], []
            for index, batch in enumerate(data_reader.tail_generator()):
                if self._sync_required:
                    break
                dev_batches.append(batch)
                output = self._exe.run(self._program,
                                       feed=batch,
                                       fetch_list=schema_in_fetch_vars)
                if outputs:
                    outputs = [
                        np.concatenate(
                            (outs, out), axis=0)
                        for (outs, out) in zip(outputs, output)
                    ]
                else:
                    outputs = copy.deepcopy(output)
            if dev_batches or outputs:
                know_make_queue.put((dev_batches, outputs))
                num_batches_sent += (index + 1)

            print("Processed {} batch samples in total.".format(
                num_batches_sent))
            know_make_queue.put(EndSignal())
            know_make_queue.join()

            if self._knowledge_queues:
                for q in self._knowledge_queues:
                    q.join()
            if self._out_file:
                offline_write_queue.join()
        print(
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) +
            "  Teacher ends serving.")

    def __del__(self):
        if self._manager:
            self._manager.shutdown()
