def read_neuron_data(file_path):
    """Read neuron position data from a file and return as a set."""
    with open(file_path, 'r') as file:
        data = file.readlines()
        neuron_positions = {line.strip() for line in data}
    return neuron_positions

# Read neuron data from both files
neurons_email = read_neuron_data('filtered-pn-gpt2-EMAIL.txt')
neurons_tel = read_neuron_data('filtered-pn-gpt2-TEL.txt')

# Calculate intersection and union
intersection = neurons_email.intersection(neurons_tel)
union = neurons_email.union(neurons_tel)

# Compute the overlap ratio
overlap_ratio = len(intersection) / len(union) if union else 0

# Print the overlap ratio
print(f'The overlap ratio of special neurons is: {overlap_ratio:.4f}')
