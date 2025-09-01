import numpy as np
import os
import time

data_dir = r"C:\Users\Moritz\Desktop\Pixis_data\data_tests"
spectrum_length = 2048  # Example length
max_spectra = 10000

# Individual .npy file saving
start_time = time.time()
for i in range(max_spectra):
    intensities = np.random.random(spectrum_length)
    np.save(os.path.join(data_dir, f"intensities_{i}.npy"), intensities)
end_time = time.time()
print(f"Saving 10000 individual .npy files took: {end_time - start_time:.2f} sec")

# Using np.memmap (flushed every iteration)
memmap_path = os.path.join(data_dir, "spectra_memmap.npy")
mmap = np.memmap(memmap_path, dtype=np.float32, mode="w+", shape=(max_spectra, spectrum_length))

start_time = time.time()
for i in range(max_spectra):
    intensities = np.random.random(spectrum_length)
    mmap[i] = intensities
    mmap.flush()  # Flushing after every write
end_time = time.time()
print(f"Using np.memmap (flushed every iteration) took: {end_time - start_time:.2f} sec")

# Using np.memmap (batch flushing every 100 iterations)
batch_size = 100
start_time = time.time()
for i in range(max_spectra):
    intensities = np.random.random(spectrum_length)
    mmap[i] = intensities
    if i % batch_size == 0 or i == 999:
        mmap.flush()  # Flush only every batch
end_time = time.time()
print(f"Using np.memmap (batch flushed every 100 iterations) took: {end_time - start_time:.2f} sec")
print(rf"Time per flush: {(end_time - start_time) / (max_spectra / batch_size):.2f} sec")
