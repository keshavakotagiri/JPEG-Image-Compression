import heapq
from collections import defaultdict, Counter

class HuffmanNode:
    def __init__(self, value=None, freq=0, left=None, right=None):
        self.value = value
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


def to_twos_complement(value, bits):
    """Convert an integer to a two's complement binary string with fixed bits."""
    if value < 0:
        value = (1 << bits) + value  # Add 2^bits to handle negative values
    return f"{value:0{bits}b}"

def from_twos_complement(binary, bits):
    """Convert a two's complement binary string to an integer."""
    value = int(binary, 2)
    if value >= (1 << (bits - 1)):  # Check if the number is negative
        value -= (1 << bits)       # Subtract 2^bits to convert back
    return value

def build_huffman_tree(freq_table):
    heap = [HuffmanNode(value, freq) for value, freq in freq_table.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq, left, right)
        heapq.heappush(heap, merged)

    return heap[0]  # Root of the tree


def build_huffman_table(root):
    table = {}

    def traverse(node, path=""):
        if node.value is not None:
            table[node.value] = path
            return
        if node.left:
            traverse(node.left, path + "0")
        if node.right:
            traverse(node.right, path + "1")

    traverse(root)
    return table

def encode_data(data, huffman_table):
    return ''.join(huffman_table[value] for value in data)


def encode_huffman_table(huffman_table):
    # Serialize the Huffman table as a string
    serialized_table = []
    for key, value in huffman_table.items():
        serialized_table.append(f"{key}:{value}")
    return '|'.join(serialized_table)


def decode_huffman_table(encoded_table):
    huffman_table = {}
    entries = encoded_table.split('|')
    for entry in entries:
        key, value = entry.split(':')
        huffman_table[int(key)] = value
    return huffman_table

def decode_data(encoded_data, huffman_table):
    reverse_table = {v: k for k, v in huffman_table.items()}
    decoded_list = []

    temp = ""
    for bit in encoded_data:
        temp += bit
        if temp in reverse_table:
            decoded_list.append(reverse_table[temp])
            temp = ""

    return decoded_list

def huffman_compress(data):
    freq_table = Counter(data)
    huffman_tree = build_huffman_tree(freq_table)
    huffman_table = build_huffman_table(huffman_tree)

    encoded_data = encode_data(data, huffman_table)
    encoded_table = encode_huffman_table(huffman_table)

    # Return the bit string of encoded table + data
    return f"{encoded_table}||{encoded_data}"


def huffman_decompress(bit_string):
    encoded_table, encoded_data = bit_string.split('||')
    huffman_table = decode_huffman_table(encoded_table)
    return decode_data(encoded_data, huffman_table)


def to_binary_string(value, bits):
    """Convert an integer to a binary string with fixed bits."""
    return f"{value:0{bits}b}"


def from_binary_string(binary):
    """Convert a binary string to an integer."""
    return int(binary, 2)

def encode_huffman_table_binary(huffman_table):
    """Encode the Huffman table as a binary string."""
    binary_table = ""
    for key, value in huffman_table.items():
        key_binary = to_twos_complement(key, 16)  # Assume keys fit in 16 bits (two's complement)
        length_binary = to_binary_string(len(value), 8)  # Length of bit string
        binary_table += key_binary + length_binary + value  # Append key, length, and bit string
    return binary_table

def decode_huffman_table_binary(binary_table):
    """Decode the binary-encoded Huffman table."""
    huffman_table = {}
    i = 0
    while i < len(binary_table):
        key = from_twos_complement(binary_table[i:i + 16], 16)  # 16 bits for the key (two's complement)
        i += 16
        length = from_binary_string(binary_table[i:i + 8])  # 8 bits for the length
        i += 8
        value = binary_table[i:i + length]  # Variable length for the bit string
        i += length
        huffman_table[key] = value
    return huffman_table

def huffman_compress_binary(data):
    freq_table = Counter(data)
    huffman_tree = build_huffman_tree(freq_table)
    huffman_table = build_huffman_table(huffman_tree)

    # Encode the Huffman table and data
    binary_table = encode_huffman_table_binary(huffman_table)
    encoded_data = encode_data(data, huffman_table)

    # Calculate the size of the binary_table and encode it as 20-bit binary
    table_size_in_bits = len(binary_table)
    table_size_binary = to_binary_string(table_size_in_bits, 20)

    # Combine the table size (20 bits), the binary table, and the encoded data
    return table_size_binary + binary_table + encoded_data

def huffman_decompress_binary(bit_string):
    # Extract the first 20 bits for the size of the binary table
    table_size_binary = bit_string[:20]
    table_size_in_bits = from_binary_string(table_size_binary)

    # Extract the Huffman table and the encoded data
    binary_table = bit_string[20:20 + table_size_in_bits]
    encoded_data = bit_string[20 + table_size_in_bits:]

    # Decode the Huffman table and the data
    huffman_table = decode_huffman_table_binary(binary_table)
    return decode_data(encoded_data, huffman_table)


# # Test case
# data = [10, 5, 0, 0, 3, 32767, -20, 32767]

# # Compress to a binary string
# compressed_binary = huffman_compress_binary(data)
# print(f"Compressed binary string: {compressed_binary}")

# # Decompress from the binary string
# decompressed_data = huffman_decompress_binary(compressed_binary)
# print(f"Decompressed data: {decompressed_data}")

# # Verify correctness
# assert data == decompressed_data
