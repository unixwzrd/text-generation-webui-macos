import math
import os
import psutil
import re
import torch
import unittest


# from modules import shared
import modules.shared as shared
from modules.logging_colors import logger

class ComputeDevice:
    '''
    Keep a list of all instances so we can use class methods for operating on all of them at once, like resetting, re-initiailixzing or anything else we might wat to do.
    '''
    devices = []

    def __init__(self, device_type=None):
        if device_type and ':' in device_type:
            self.device_type, self.local_rank = device_type.split(':')
            self.local_rank = int(self.local_rank)
        else:
            self.device_type = device_type if device_type else self.select_device()
            self.local_rank = self.get_local_rank()

        self.device = torch.device(self.device_type, self.local_rank)
        ComputeDevice.devices.append(self)

        # Initialize memory attributes
        self.system_memory = None
        self.gpu_memory = None
        self.cpu_memory = None
        # Calculate memory
        self.total_mem = self.calculate_memory()
        # Call the methods to set the device and memory attributes
        self.select_device()
        self.calculate_memory()

    @classmethod
    def clear_all_cache(cls):
        '''
        This frees all cache space used by every device of ComputeDevice class we have created.
        '''
        for device in cls.devices:
            device.clear_cache()

    def clear_cache(self):
        '''
        This clears the cache for the torch device passeed to us.
        '''
        if self.device_type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device_type == 'mps':
            torch.mps.empty_cache()

        # Remove the device from the list
        ComputeDevice.devices.remove(self)

    def get_local_rank(self):
        '''
        Get local renk is assigned in config or as environment variable.
        '''
        try:
            local_rank = shared.args.local_rank
        except TypeError:
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
        return local_rank

    def select_device(self):
        '''
        This will contain the logic to select the appropriate device (CUDA, MPS, CPU) 
        
        Default is CPU

        Local rank is just an index of the torch device.
        
        The statement: torch.device('cuds:0')
        Is identical to: torch.device('cuda', 0)

        '''
        local_rank = self.get_local_rank()
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

    def calculate_memory(self):
        '''
        Perform all memory calculations to determine total system memory, total GPU memory, and CPU memory available for use by the application.  Some of these are adjusted by amounts for reservations specified in the config files.
        '''
        self.system_memory = math.floor(psutil.virtual_memory().total / (1024 * 1024))

        # Check for MPS, CUDA, or CPU and calculate total memory accordingly
        if torch.backends.mps.is_available():
            self.gpu_memory = [self.system_memory]
        elif torch.cuda.is_available():
            self.gpu_memory = [math.floor(torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)) for i in range(torch.cuda.device_count())]
        else:
            self.gpu_memory = [self.system_memory]

        # Calculate default reserved GPU memory
        self.default_gpu_mem = []
        if shared.args.gpu_memory is not None and len(shared.args.gpu_memory) > 0:
            for i in shared.args.gpu_memory:
                if 'mib' in i.lower():
                    self.default_gpu_mem.append(int(re.sub('[a-zA-Z ]', '', i)))
                else:
                    self.default_gpu_mem.append(int(re.sub('[a-zA-Z ]', '', i)) * 1000)
        while len(self.default_gpu_mem) < len(self.gpu_memory):
            self.default_gpu_mem.append(0)

        # Calculate default reserved CPU memory
        if shared.args.cpu_memory is not None:
            self.cpu_memory = int(re.sub('[a-zA-Z ]', '', shared.args.cpu_memory))
        else:
            self.cpu_memory = 0

        # Calculate the total available memory for the application
        self.total_mem = [gm - dgm for gm, dgm in zip(self.gpu_memory, self.default_gpu_mem)]
        self.total_mem.append(self.system_memory - self.cpu_memory)



# Unit testing for this class.
class TestComputeDevice(unittest.TestCase):
    def setUp(self):
        self.device = ComputeDevice('cpu')

    def test_device_type(self):
        self.assertEqual(self.device.device_type, 'cpu')

    def test_local_rank(self):
        self.assertEqual(self.device.local_rank, 0)

    def test_device(self):
        self.assertEqual(self.device.device.type, 'cpu')

    def test_memory_calculation(self):
        self.assertIsNotNone(self.device.system_memory)
        self.assertIsNotNone(self.device.gpu_memory)
        self.assertIsNotNone(self.device.cpu_memory)

    def test_clear_cache(self):
        # This is a bit tricky to test as it doesn't return anything
        # But at least we can check it doesn't raise an error
        try:
            self.device.clear_cache()
        except Exception as e:
            self.fail(f"clear_cache raised an exception: {e}")

    def test_clear_all_cache(self):
        # Similar to test_clear_cache
        try:
            ComputeDevice.clear_all_cache()
        except Exception as e:
            self.fail(f"clear_all_cache raised an exception: {e}")

# If this is run directly from the command line, rather than imported, it willr 
# run the unit tests
if __name__ == '__main__':
    unittest.main()
