def to_bytes(binary_str):
    """Convert a binary string to bytes."""
    # Ensure the length is a multiple of 8 by padding
    while len(binary_str) % 8 != 0:
        binary_str += '0'  # Add padding bits

    # Convert to bytes
    byte_array = bytearray()
    for i in range(0, len(binary_str), 8):
        byte_array.append(int(binary_str[i:i+8], 2))
    
    return byte_array


def from_bytes(byte_array):
    """Convert bytes back to a binary string."""
    binary_str = ''.join(f'{byte:08b}' for byte in byte_array)
    return binary_str


def save_to_file(binary_str, filename):
    """Save binary string to a .bin file with padding information."""
    # print(binary_str)
    # Calculate the padding needed
    padding_needed = (8 - (len(binary_str)+3) % 8) % 8
    padded_binary_str = binary_str + '0' * padding_needed # Pad with zeros 

    # Add 3 bits at the start to store the padding information
    padding_info = f"{padding_needed:03b}"  # Convert padding to 3 bits binary
    final_binary_str = padding_info + padded_binary_str

    # Convert the final binary string to bytes
    byte_data = to_bytes(final_binary_str)

    # Write the bytes to the file
    with open(filename, 'wb') as f:
        f.write(byte_data)


def load_from_file(filename):
    """Load binary string from a .bin file and remove padding."""
    # Read the byte data from the file
    with open(filename, 'rb') as f:
        byte_data = f.read()

    # Convert bytes back to a binary string
    binary_str = from_bytes(byte_data)

    # Extract the padding information (first 3 bits)
    padding_info = binary_str[:3]
    padding_needed = int(padding_info, 2)
    # print(padding_needed)
    # Remove the padding
    # print(binary_str)
    actual_binary_str = binary_str[3:-padding_needed] if padding_needed > 0 else binary_str[3:]
    # print(actual_binary_str)
    return actual_binary_str


# # Test the functions
# data = "1000001"  # Example binary string
# filename = "output.bin"

# # Save to file
# save_to_file(data, filename)
# # print(f"Saved binary data to {filename}")

# # Load from file and remove padding
# loaded_data = load_from_file(filename)
# print(f"Loaded binary data: {loaded_data}")

# # Check if the loaded data matches the original (before padding)
# assert data == loaded_data
