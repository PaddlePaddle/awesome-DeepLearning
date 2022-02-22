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

import six
import time
if six.PY2:
    import cPickle as pickle
    import Queue
else:
    import pickle
    import queue as Queue

import numpy as np
from collections import OrderedDict
from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager

from threading import Thread

from paddleslim.pantheon.utils import EndSignal, SyncSignal, StartSignal, public_authkey, convert_dtype

__all__ = ["Student"]


class Student(object):
    """
    The class defined for the student model. Receive knowledge data from 
    teacher model and carry out knowledge merging.    

    Args:
        merge_strategy (dict|None): A dictionary whose keys are common 
            schemas shared by different teachers, and each corresponding 
            value specifies the merging strategy for different schemas 
            respectively, supporting 'sum' and 'mean' now.
    """

    def __init__(self, merge_strategy=None):
        if merge_strategy:
            for strategy in merge_strategy.values():
                if strategy not in ["sum", "mean"]:
                    raise ValueError(
                        "Merging strategy must be 'sum' or 'mean'!")

        self._merge_strategy = merge_strategy
        self._common_schema = merge_strategy.keys() if merge_strategy else []

        self._knowledge_desc = OrderedDict()
        self._knowledge_queue = Queue.Queue(100)
        self._teacher_knowledge_queues = []
        self._t2s_queues = []
        self._s2t_queues = []
        self._cmd_queues = []

        self._num_teachers = 0

        self._in_paths = []
        self._in_addresses = []

        self._started = False
        self._is_knowledge_desc_ready = False
        self._is_knowledge_gen_locked = False

    def register_teacher(self, in_path=None, in_address=None):
        """Register one teacher model and assign the order number to it as 
           its id, with the file path (offline mode) or IP address (online 
           mode) that the teacher model wrote knowledge data to.

        Args:
            in_path (str|None): The input file path. Default None.
            in_address (str|None): The input IP address, in the format 
                "<IP address>:<IP port>" (e.g. "127.0.0.1:8080"). Default None.
        """
        if self._started:
            raise ValueError(
                "The student has been started and cannot register "
                "teacher no longer!")
        if in_path and in_address:
            raise ValueError("Input path and input address should not "
                             "be given at the same time!")
        if not in_path and not in_address:
            raise ValueError("One of input path and input address should "
                             "be given when registering teacher!")
        if in_address:
            if in_address in self._in_addresses:
                print("WARNING: the teacher with input address {} has been "
                      "registered, and ignored this time!".format(in_path))
                return
            ip, port = in_address.strip().split(":")
            BaseManager.register("get_knowledge_queue")
            BaseManager.register("get_s2t_queue")
            BaseManager.register("get_t2s_queue")
            BaseManager.register("get_cmd_queue")
            manager = BaseManager(
                address=(ip, int(port)), authkey=public_authkey.encode())

            print("Connecting to {}, with public key {} ...".format(
                in_address, public_authkey))
            # Wait for teacher model started to establish connection
            while True:
                try:
                    manager.connect()
                    break
                except:
                    time.sleep(1.0)

            def merge(knowledge_queues):
                num = len(knowledge_queues)
                if num == 1:
                    return knowledge_queues[0]
                local_queues = [Queue.Queue(100) for _ in range(num)]

                def receive(queue, local_queue):
                    while True:
                        try:
                            data = queue.get()
                            queue.task_done()
                            local_queue.put(data)
                        except EOFError:
                            break

                knowledge_queue = Queue.Queue(100)

                def gather(local_queues, knowledge_queue):
                    num = len(local_queues)
                    end_received = [0] * num
                    while True:
                        try:
                            for i in range(num):
                                data = local_queues[i].get()
                                local_queues[i].task_done()

                                if isinstance(data, SyncSignal):
                                    if i == 0:
                                        knowledge_queue.put(data)
                                elif isinstance(data, EndSignal):
                                    end_received[i] = 1
                                    if i == 0:
                                        knowledge_queue.put(data)
                                    if sum(end_received) == num:
                                        end_received = [0] * num
                                        break
                                else:
                                    knowledge_queue.put(data)
                        except EOFError:
                            break

                # threads to receive knowledge from the online teacher
                for i in range(num):
                    p = Thread(
                        target=receive,
                        args=(knowledge_queues[i], local_queues[i]))
                    p.daemon = True
                    p.start()
                # thread to gather data from different local queues
                p = Thread(target=gather, args=(local_queues, knowledge_queue))
                p.daemon = True
                p.start()
                return knowledge_queue

            # get knowledge queues
            knowledge_queues, idx = [], 0
            while True:
                q = manager.get_knowledge_queue(idx)
                if hasattr(q, "get"):
                    knowledge_queues.append(q)
                    idx += 1
                else:
                    break
            knowledge_queue = merge(knowledge_queues)
            self._t2s_queues.append(manager.get_t2s_queue())
            self._s2t_queues.append(manager.get_s2t_queue())
            self._cmd_queues.append(manager.get_cmd_queue())
            self._in_addresses.append(in_address)
            self._in_paths.append(None)
            print("Registered teacher {} with input address {}.".format(
                self._num_teachers, in_address))
        else:
            if in_path in self._in_paths:
                print("WARNING: th teacher with input path {} has been "
                      "registered, and ignored this time!".format(in_path))
                return

            def read_offline(in_path, cmd_queue, out_queue):
                end_recved = False

                def get_cmd():
                    cmd, end_recved = None, False
                    try:
                        if not cmd_queue.empty():
                            cmd = cmd_queue.get()
                            cmd_queue.task_done()
                            if isinstance(cmd, EndSignal):
                                end_recved = True
                    except IOError:
                        end_recved = True
                    return cmd, end_recved

                # wait for the sync in start
                while not end_recved:
                    cmd, end_recved = get_cmd()
                    if isinstance(cmd, SyncSignal):
                        out_queue.put(SyncSignal())
                        break
                # for multiple-times offline serving
                while not end_recved:
                    # wait for the sync in get_knowledge_desc()
                    while not end_recved:
                        cmd, end_recved = get_cmd()
                        if isinstance(cmd, SyncSignal):
                            out_queue.put(SyncSignal())
                            break

                    if end_recved:
                        break
                    with open(in_path, 'rb') as fin:
                        # get knowledge desc
                        desc = pickle.load(fin)
                        out_queue.put(desc)
                        # wait for the data accessing signal
                        while not end_recved:
                            cmd, end_recved = get_cmd()
                            if isinstance(cmd, StartSignal):
                                break
                        # get knowledge data
                        while not end_recved:
                            try:
                                data = pickle.load(fin)
                                out_queue.put(data)
                                _, end_recved = get_cmd()
                            except EOFError:
                                break
                    if end_recved:
                        break
                    out_queue.put(EndSignal())
                    out_queue.join()

            knowledge_queue = Queue.Queue(100)
            cmd_queue = Queue.Queue(5)
            p = Thread(
                target=read_offline,
                args=(in_path, cmd_queue, knowledge_queue))
            p.daemon = True
            p.start()

            self._t2s_queues.append(None)
            self._s2t_queues.append(None)
            self._cmd_queues.append(cmd_queue)
            self._in_addresses.append(None)
            self._in_paths.append(in_path)
            print("Registered teacher {} with input path {}.".format(
                self._num_teachers, in_path))

        self._teacher_knowledge_queues.append(knowledge_queue)
        self._num_teachers += 1

    def _sync(self):
        for i, queue in enumerate(self._cmd_queues):
            if queue:
                queue.put(SyncSignal())
                while True:
                    cmd = self._teacher_knowledge_queues[i].get()
                    self._teacher_knowledge_queues[i].task_done()
                    if isinstance(cmd, SyncSignal):
                        break
                queue.join()

    def start(self):
        """
        End teachers' registration and synchronize with all of them.
        """

        if self._started:
            raise ValueError(
                "The student cannot be started more than one time.")
        self._sync()
        self._started = True

    def _merge_knowledge(self, knowledge):
        for k, tensors in list(knowledge.items()):
            if len(tensors) == 0:
                del knowledge[k]
            elif len(tensors) == 1:
                knowledge[k] = tensors[0]
            else:
                result = 0
                for tensor in tensors:
                    result += tensor
                if self._merge_strategy[k] == "sum":
                    knowledge[k] = result
                elif self._merge_strategy[k] == "mean":
                    knowledge[k] = result / len(tensors)
            # cast back to original data type if necessary
            tgt_dtype = self._knowledge_desc[k]["dtype"]
            if str(knowledge[k].dtype) != tgt_dtype:
                knowledge[k] = knowledge[k].astype(tgt_dtype)
        return knowledge

    def send(self, data, teacher_ids=None):
        """ 
        Send data to teachers.

        Args:
            data: A Python data object.
            teacher_ids (list|None): A list of teacher ids to send data. If 
                set to None, send the data to all teachers. Default None.
        """
        if not self._started:
            raise ValueError("The method start() should be called first!")

        if teacher_ids is None:
            teacher_ids = range(self._num_teachers)

        for i in teacher_ids:
            if self._s2t_queues[i]:
                self._s2t_queues[i].put(data)
            else:
                print("Warning: didn't send data to teacher {} for it is in "
                      "offline mode.".format(i))

    def recv(self, teacher_id):
        """
        Receive data from one teacher.
       
        Args:
            teacher_id (int): The id of teacher that receives data from.

        Return:
            The received data object.
        """
        if not self._started:
            raise ValueError("The method start() should be called first!")

        if self._t2s_queues[teacher_id]:
            data = self._t2s_queues[teacher_id].get()
            self._t2s_queues[teacher_id].task_done()
            return data
        else:
            raise ValueError("Cannot receive data from teacher {} for it is "
                             "offline.".format(teacher_id))

    def get_knowledge_desc(self):
        """ 
        Get description for knowledge, including shape, data type and lod 
        level for each schema.

        Return:
            dict: Knowledge description.
        """
        if not self._started:
            raise ValueError("The method start() should be called first!")

        if self._is_knowledge_desc_ready == False:
            self._sync()
            # get knowledge description
            knowledge_desc = OrderedDict()
            for idx, queue in enumerate(self._teacher_knowledge_queues):
                desc = queue.get()
                queue.task_done()
                inter_desc = set(knowledge_desc.keys()) & set(desc.keys())
                if idx > 0 and (
                        not inter_desc.issubset(set(self._common_schema))):
                    raise ValueError(
                        "Teacher {} has the same schema with other existed "
                        "teachers not in the merge_strategy.".format(idx))
                knowledge_desc.update(desc)

            print("Knowledge merging strategy: {}".format(
                self._merge_strategy))
            print("Knowledge description after merging:")
            for schema, desc in list(knowledge_desc.items()):
                print("{}: {}".format(schema, desc))

            self._knowledge_desc = knowledge_desc
            self._is_knowledge_desc_ready = True
        return self._knowledge_desc

    def get_knowledge_qsize(self):
        """
        Get the real-time size of knowledge queue. If this size is denoted as 
        **qsize**, it means that there are **qsize** batch knowledge data 
        already pushed into knowledge queue and waiting for the knowledge 
        generator to pop out. It's dynamic and limited up to 100, the capacity 
        of the knowledge queue.
        
        Return:
            int: The real-time size of knowledge queue.
        """
        if not self._started:
            raise ValueError("The method start() should be called first!")

        return self._knowledge_queue.qsize()

    def get_knowledge_generator(self, batch_size, drop_last=False):
        """ 
        Get the generator for knowledge data, return None if last generator 
        doesn't finish yet.

        Args:
            batch_size (int): The batch size of returned knowledge data.
            drop_last (bool): Whether to drop the last batch if its size is less 
                              than batch size.

        Return:
            func: The wrapper of knowledge data generator.
        """
        if not self._started:
            raise ValueError("The method start() should be called first!")

        if batch_size <= 0:
            raise ValueError("batch size must be positive!")
        self._batch_size = batch_size
        self._drop_last = drop_last

        # make sure only one generator is available at the same time
        if self._is_knowledge_gen_locked:
            print("WARNING: new knowledge generator is not available for the "
                  "last generator hasn't finished yielding all data yet! "
                  "Return None.")
            return None
        self._is_knowledge_gen_locked = True
        self.get_knowledge_desc()

        def split_batch(batch, num):
            keys = batch.keys()
            first, second = {}, {}
            for key in keys:
                first[key] = batch[key][0:num]
                second[key] = batch[key][num:]
            return first, second

        def concat_batches(batches):
            if len(batches) == 1:
                return batches[0]
            keys = batches[0].keys()
            ret_batch = {}
            for key in keys:
                ret_batch[key] = np.concatenate(
                    [batches[i][key] for i in range(len(batches))])
            return ret_batch

        def listen(knowledge_queue, out_queue):
            """
            listen on the knowledge queue for one teacher, get knowledge data
            and put it into a local queue (out_queue). 
            """
            while True:
                data = knowledge_queue.get()
                knowledge_queue.task_done()
                out_queue.put(data)
                if isinstance(data, EndSignal):
                    break

        def make_new_batch(in_queue, out_queue, batch_size):
            """ 
            Get knowledge data from a local queue and make a new batch data in 
            the batch size of student, then put it into the intermediate 
            queue (out_queue).
            """
            batches, num_samples = [], 0
            while True:
                batch_samples = in_queue.get()
                in_queue.task_done()
                if not isinstance(batch_samples, EndSignal):
                    cur_num_samples = list(batch_samples.values())[0].shape[0]
                    if num_samples + cur_num_samples < batch_size:
                        batches.append(batch_samples)
                        num_samples += cur_num_samples
                    elif num_samples + cur_num_samples == batch_size:
                        batches.append(batch_samples)
                        out_queue.put(concat_batches(batches))
                        batches, num_samples = [], 0
                    else:
                        num_splited = batch_size - num_samples
                        first, second = split_batch(batch_samples, num_splited)
                        batches.append(first)
                        out_queue.put(concat_batches(batches))
                        num_left = cur_num_samples - num_splited
                        while num_left > batch_size:
                            first, second = split_batch(second, batch_size)
                            out_queue.put(first)
                            num_left -= batch_size

                        if num_left == batch_size:
                            out_queue.put(second)
                            batches, num_samples = [], 0
                        else:
                            batches, num_samples = [second], num_left
                else:
                    if len(batches) > 0:
                        out_queue.put(concat_batches(batches))
                    out_queue.put(EndSignal())
                    break

        def gather_and_merge(in_queues, out_queue):
            """ 
            Gather knowledge from all intermediate queues, merge them 
            and put the final knowledge into the knowledge queue to 
            student (out_queue).
            """

            def data_receiver(queue):
                while True:
                    batch = queue.get()
                    queue.task_done()
                    yield batch
                    if isinstance(batch, EndSignal):
                        break

            data_receivers = [data_receiver(queue) for queue in in_queues]

            end_received = [0] * len(in_queues)
            while True:
                knowledge = OrderedDict(
                    [(k, []) for k, v in list(self._knowledge_desc.items())])
                for idx, receiver in enumerate(data_receivers):
                    if not end_received[idx]:
                        batch_samples = receiver.next(
                        ) if six.PY2 else receiver.__next__()
                        if not isinstance(batch_samples, EndSignal):
                            for k, v in list(batch_samples.items()):
                                knowledge[k].append(v)
                        else:
                            end_received[idx] = 1
                if sum(end_received) == len(in_queues):
                    break
                knowledge = self._merge_knowledge(knowledge)
                out_queue.put(knowledge)
            out_queue.put(EndSignal())
            out_queue.join()

        # acquire data from teachers
        for i, queue in enumerate(self._cmd_queues):
            if queue:
                queue.put(StartSignal())
                queue.join()

        local_queues = [Queue.Queue(100) for i in range(self._num_teachers)]
        # launch threads to listen on all knowledge queues
        for i in range(self._num_teachers):
            listen_thread = Thread(
                target=listen,
                args=(self._teacher_knowledge_queues[i], local_queues[i]))
            listen_thread.dameon = True
            listen_thread.start()

        med_queues = [Queue.Queue(100) for i in range(self._num_teachers)]
        # launch threads to make new batch for student
        for i in range(self._num_teachers):
            listen_thread = Thread(
                target=make_new_batch,
                args=(local_queues[i], med_queues[i], self._batch_size))
            listen_thread.dameon = True
            listen_thread.start()

        # launch another thread to merge knowledge from different teachers.
        merge_thread = Thread(
            target=gather_and_merge, args=(med_queues, self._knowledge_queue))
        merge_thread.dameon = True
        merge_thread.start()

        def wrapper():
            while True:
                knowledge = self._knowledge_queue.get()
                self._knowledge_queue.task_done()
                if not isinstance(knowledge, EndSignal):
                    batch_size = list(knowledge.values())[0].shape[0]
                    if (batch_size < self._batch_size) and drop_last:
                        continue
                    yield knowledge
                else:
                    break
            # After all knowledge data yielded, make current knowledge desc invalid.
            self._is_knowledge_desc_ready = False
            self._is_knowledge_gen_locked = False

        return wrapper

    def __del__(self):
        for i, path in enumerate(self._in_paths):
            if path:
                try:
                    self._cmd_queues[i].put(EndSignal())
                    self._cmd_queues[i].join()
                except:
                    pass
