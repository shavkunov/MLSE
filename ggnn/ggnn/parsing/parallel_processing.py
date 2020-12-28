import time

from os import path
from enum import Enum
from typing import Generator, Any, List
from multiprocessing import cpu_count, Manager
from multiprocessing.pool import Pool
from ctypes import c_bool, c_int

from .utils import batch_generator


class DataLoader:
    def load(self) -> Generator[Any, None, None]:
        raise NotImplemented


class DataProcessor:
    def process(self, content, meta):
        raise NotImplemented

    def init_meta(self):
        raise NotImplemented


class DataSaver:
    def save(self, result, file_path: str, filename: str):
        raise NotImplemented


class FileFilter:
    def accept(self, file_path: str):
        raise NotImplemented


class SequentialProcessors(DataProcessor):
    def __init__(self, processors: List[DataProcessor]):
        self.processors = processors

    def init_meta(self):
        return [processor.init_meta() for processor in self.processors]

    def process(self, content, metas):
        for processor, meta in zip(self.processors, metas):
            content = processor.process(content, meta)
        return content


def log(text, f):
    import datetime
    with open(f, 'a', encoding='utf-8') as file:
        file.write(str(datetime.datetime.now()))
        file.write('\t')
        file.write(text)
        file.write('\n')


class PushToClosedQueueException(Exception):
    pass


class QueueWrapper:
    def __init__(self, manager: Manager, max_size=0):
        self.lock = manager.Lock()
        self.not_full = manager.Condition(self.lock)
        self.not_empty = manager.Condition(self.lock)
        self.closed = manager.Value(c_bool, False)
        self.queue = manager.Queue(max_size)

    def push(self, value):
        with self.lock:
            while self.queue.full() and not self.closed.value:
                self.not_full.wait()
            if self.closed.value:
                raise PushToClosedQueueException()
            self.queue.put(value)
            self.not_empty.notify()

    def pop(self):
        with self.lock:
            while self.queue.empty() and not self.closed.value:
                self.not_empty.wait()
            if self.queue.empty():
                return None
            self.not_full.notify()
            return self.queue.get()

    def close(self):
        with self.lock:
            self.closed.value = True
            self.not_empty.notify_all()
            self.not_full.notify_all()


class MessageType(Enum):
    LOADER_DIED = 0
    PROCESSOR_DIED = 1
    SAVER_DIED = 2
    LOADER_FINISHED = 3
    PROCESSOR_FINISHED = 4
    SAVER_FINISHED = 5
    PROCESSOR_CATCH_EXCEPTION = 6


class Message:
    def __init__(self, message_type: MessageType, data=None):
        self.message_type = message_type
        self.data = data


class ParallelContext:
    def __init__(self, num_processors, queues_max_size):
        manager = Manager()

        self.content_queue = QueueWrapper(manager, queues_max_size)
        self.result_queue = QueueWrapper(manager, queues_max_size)
        self.message_queue = QueueWrapper(manager)

        self.num_processors_alive = manager.Value(c_int, num_processors)
        self.num_processors_alive_lock = manager.Lock()

    def report_loader_die(self, exception):
        self.message_queue.push(Message(MessageType.LOADER_DIED, exception))

    def report_processor_die(self, exception):
        self.message_queue.push(Message(MessageType.PROCESSOR_DIED, exception))

    def report_saver_die(self, exception):
        self.message_queue.push(Message(MessageType.SAVER_DIED, exception))


def __load_data(loader, context, batch_size):
    for batch in batch_generator(loader.load(), batch_size):
        context.content_queue.push(batch)
    context.content_queue.close()
    context.message_queue.push(Message(MessageType.LOADER_FINISHED))


def __process_data(processor, context):
    meta = processor.init_meta()
    while True:
        content_batch = context.content_queue.pop()
        if not content_batch:
            break
        result_batch = []
        for relative_path, filename, content in content_batch:
            try:
                result = processor.process(content, meta)
                result_batch.append((relative_path, filename, result))
            except Exception as e:
                context.message_queue.push(Message(MessageType.PROCESSOR_CATCH_EXCEPTION,
                                                   (path.join(relative_path, filename), e)))

        if result_batch:
            context.result_queue.push(result_batch)
    with context.num_processors_alive_lock:
        context.num_processors_alive.value -= 1
        if not context.num_processors_alive.value:
            context.result_queue.close()
    context.message_queue.push(Message(MessageType.PROCESSOR_FINISHED))


def __save_data(saver, context):
    with saver:
        while True:
            result_batch = context.result_queue.pop()
            if not result_batch:
                break
            for relative_path, filename, result in result_batch:
                saver.save(result, relative_path, filename)
    context.message_queue.push(Message(MessageType.SAVER_FINISHED))


class DataProcessingException(Exception):
    def __init__(self, message: str):
        super().__init__(message)


def run_parallel(loader: DataLoader, processor: DataProcessor, saver: DataSaver,
                 batch_size=10, queues_max_size=20,
                 num_processors=max(cpu_count() - 2, 1), output=None):
    with Pool(num_processors + 2) as pool:
        context = ParallelContext(num_processors, queues_max_size)
        pool.apply_async(__load_data, args=(loader, context, batch_size,),
                         error_callback=context.report_loader_die)
        print(f"[{time.asctime()}] Loader was spawned", file=output)
        pool.apply_async(__save_data, args=(saver, context,),
                         error_callback=context.report_saver_die)
        print(f"[{time.asctime()}] Saver was spawned", file=output)
        for _ in range(num_processors):
            pool.apply_async(__process_data, args=(processor, context,),
                             error_callback=context.report_processor_die)
            print(f"[{time.asctime()}] New processor was spawned", file=output)
        while True:
            message = context.message_queue.pop()
            if message.message_type is MessageType.LOADER_DIED:
                print(f"[{time.asctime()}] Fatal error: loaded fell with error: {str(message.data)}", file=output)
                pool.terminate()
                raise DataProcessingException("Fatal error: loaded fell") from message.data
            elif message.message_type is MessageType.SAVER_DIED:
                print(f"[{time.asctime()}] Fatal error: saver fell with error: {str(message.data)}", file=output)
                pool.terminate()
                raise DataProcessingException("Fatal error: saver fell") from message.data
            elif message.message_type is MessageType.PROCESSOR_DIED:
                print(f"[{time.asctime()}] Error: processor fell with error: {str(message.data)}", file=output)
                pool.apply_async(__process_data, args=(processor, context,),
                                 error_callback=context.report_processor_die)
                print(f"[{time.asctime()}] New processor was spawned", file=output)
            elif message.message_type is MessageType.PROCESSOR_CATCH_EXCEPTION:
                data_path, e = message.data
                print(f"[{time.asctime()}] Fail to process {data_path}, "
                      f"cause: {str(e.__class__.__name__)} {str(e)}", file=output)
            elif message.message_type is MessageType.LOADER_FINISHED:
                print(f"[{time.asctime()}] Loader successfully finished", file=output)
            elif message.message_type is MessageType.PROCESSOR_FINISHED:
                print(f"[{time.asctime()}] Processor successfully finished", file=output)
            elif message.message_type is MessageType.SAVER_FINISHED:
                print(f"[{time.asctime()}] Saver successfully finished", file=output)
                break


def run_sequential(loader: DataLoader, processor: DataProcessor, saver: DataSaver):
    meta = processor.init_meta()
    with saver:
        print("WHAT!")
        for relative_path, filename, content in loader.load():
            try:
                print(relative_path)
                result = processor.process(content, meta)
                saver.save(result, relative_path, filename)
            except Exception as e:
                raise e
                pass
