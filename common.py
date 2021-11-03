from multiprocessing import Array, Process, Queue, Value
import numpy as np
import queue
import tensorflow.keras as keras

class MultiprocessMnistGenerator:
    def __init__(self, batch_size=32, num_points=500, buffer_size=5, num_workers=1, threshold=50, test=False):
        self.batch_size = batch_size
        self.num_points = num_points
        self.is_running = Value('b', False)
        self.num_workers = num_workers
        self.workers = []
        self.digits = self.load_data(threshold, test)
        self.batches_x = np.frombuffer(Array('f', buffer_size*batch_size*num_points*2, lock=False), dtype=np.float32) \
                           .reshape((buffer_size, batch_size, num_points, 2))
        self.batches_y = np.frombuffer(Array('i', buffer_size*batch_size, lock=False), dtype=np.int32) \
                           .reshape((buffer_size, batch_size))
        self.batches = (self.batches_x, self.batches_y)
        self.ready_batches = Queue(buffer_size)
        self.stale_batches = Queue(buffer_size)
        
        for i in range(buffer_size):
            self.stale_batches.put(i)
        
        
    def load_data(self, threshold, test):
        """
        Loads information into shared memory
        """
        # Load MNIST
        training, testing = keras.datasets.mnist.load_data()
        x, y = testing if test else training
        
        # Extract valid pixels for points
        img_ids, y_pixels, x_pixels = np.nonzero(x > threshold)
        
        # Create shared pixel array
        pixel_shared_arr = Array('f', 2*len(img_ids), lock=False)
        pixels = np.frombuffer(pixel_shared_arr, dtype=np.float32).reshape((-1, 2))
        pixels[:] = np.column_stack((x_pixels, 28 - y_pixels))
        
        # Standardize the pixels
        mean = np.mean(pixels, axis=0)
        std = np.std(pixels, axis=0)
        pixels[:] = (pixels - mean) / std
        
        indices = np.frombuffer(Array('i', len(x), lock=False), dtype=np.int32)
        indices[:], pixel_counts = np.unique(img_ids, return_counts=True)
        pixel_offsets = np.concatenate([[0], np.cumsum(pixel_counts)])
        
        return indices, pixels, pixel_offsets, std, y
        
    def start(self):
        if self.is_running.value:
            raise Exception("Workers are already running")
        args = (self.is_running, self.batches, self.stale_batches, self.ready_batches, self.digits, self.batch_size, self.num_points)
        self.is_running.value = True
        for _ in range(self.num_workers):
            worker = Process(target=MultiprocessMnistGenerator.worker, args=args)
            worker.start()
            self.workers.append(worker)
        
    def stop(self):
        if not self.is_running.value:
            raise Exception("Workers are already stopped")
        self.is_running.value = False
        for worker in self.workers:
            worker.join()
        self.workers = []
            
    def terminate(self):
        self.is_running.value = False
        for worker in self.workers:
            worker.terminate()
        self.workers = []
        
    def generator(self):
        while True:
            yield next(self)
        
    def __next__(self):
        batch_id = self.ready_batches.get()
        self.stale_batches.put(batch_id)
        return self.batches_x[batch_id], self.batches_y[batch_id]
        
    @staticmethod
    def generate_batch(batches, batch_id, digits, batch_size, num_points):
        batches_x, batches_y = batches
        x, y = batches_x[batch_id], batches_y[batch_id]
        
        indices, pixels, offsets, std, labels = digits
        indices = np.random.choice(indices, size=batch_size, replace=False)
        y[:] = labels[indices]
        
        for i, digit in enumerate(indices):
            px = pixels[offsets[digit]:offsets[digit+1]]
            x[i] = px[np.random.randint(0, len(px), size=num_points)]
            x[i] += np.random.uniform(0.0, 1/std, size=x[i].shape)
            
    @staticmethod
    def worker(is_running, batches, stale_batches, ready_batches, digits, batch_size, num_points):
        while is_running.value:
            try:
                batch_id = stale_batches.get(timeout=1.0)
                MultiprocessMnistGenerator.generate_batch(batches, batch_id, digits, batch_size, num_points)
                ready_batches.put(batch_id)
            except queue.Empty:
                continue
            except KeyboardInterrupt:
                continue