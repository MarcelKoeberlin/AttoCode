import numpy as np
import json

# Load metadata
json_path = r'c:\Users\Moritz\Desktop\TESTDATA\2025\STRA\250904\250904_001\250904_001.json'
with open(json_path, 'r') as f:
    metadata = json.load(f)

# Get parameters
npy_path = r'c:\Users\Moritz\Desktop\TESTDATA\2025\STRA\250904\250904_001\250904_001.npy'
images_acquired = metadata['images_acquired']
spectra_shape = metadata['spectra_shape']

print(f'Trying to load structured array with {images_acquired} records...')

# Create the structured dtype based on metadata
dtype_info = metadata['dtype']
dt = np.dtype([
    ('spectrum', dtype_info['spectrum'], tuple(spectra_shape)),
    ('timestamp_us', dtype_info['timestamp_us'])
])

print(f'Expected dtype: {dt}')
print(f'Expected single record size: {dt.itemsize} bytes')
print(f'Expected total size: {dt.itemsize * images_acquired} bytes')
print(f'Actual file size: 13585152 bytes')

try:
    # Try to load as memory-mapped structured array
    data = np.memmap(npy_path, dtype=dt, mode='r', shape=(images_acquired,))
    print('✓ Successfully loaded as memory-mapped structured array!')
    print(f'Data shape: {data.shape}')
    print(f'Data dtype: {data.dtype}')
    print(f'First timestamp: {data[0]["timestamp_us"]}')
    print(f'First spectrum shape: {data[0]["spectrum"].shape}')
    print(f'First few spectrum values: {data[0]["spectrum"].flatten()[:10]}')
    
except Exception as e:
    print(f'✗ Memory map failed: {e}')
    
    # Try loading as raw binary and reshaping
    try:
        print('Trying alternative approach...')
        # Calculate expected shape
        spectrum_size = np.prod(spectra_shape)
        record_size = spectrum_size * 2 + 8  # uint16 spectrum + uint64 timestamp
        n_records = 13585152 // record_size
        print(f'Calculated records from file size: {n_records}')
        
        with open(npy_path, 'rb') as f:
            raw_data = f.read()
        
        print(f'Read {len(raw_data)} bytes')
        print(f'First 32 bytes as hex: {raw_data[:32].hex()}')
        
        # Try to find data pattern by skipping potential header
        # Look for non-zero data after the initial zeros/0xff pattern
        for offset in [0, 128, 256, 512, 1024]:
            chunk = raw_data[offset:offset+32]
            print(f'Bytes at offset {offset}: {chunk.hex()}')
            # Check if this looks like reasonable data
            if not all(b in [0x00, 0xff] for b in chunk):
                print(f'Found potential data at offset {offset}')
                break
        
    except Exception as e2:
        print(f'✗ Raw binary read failed: {e2}')
