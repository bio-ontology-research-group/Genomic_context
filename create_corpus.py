import os
import random
import click
from tqdm import tqdm 

def create_sentences(directory, step_size):
    sentences = []
    
    # Iterate over the text files in the directory
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            
            # Read the contents of the file
            with open(file_path, 'r') as file:
                content = file.read()
                words = content.split()
                
                # Generate sentences of length 9 with specified step size
                for i in range(0, len(words) - 8, step_size):
                    sentence = ' '.join(words[i:i+9])
                    sentences.append(sentence)
    
    # Shuffle the sentences randomly
    random.shuffle(sentences)
    
    # Split the sentences into train and test sets
    train_size = int(0.9 * len(sentences))
    train_sentences = sentences[:train_size]
    test_sentences = sentences[train_size:]
    
    # Write the train sentences to train.txt
    with open('train.txt', 'w') as train_file:
        train_file.write('\n'.join(train_sentences))
    
    # Write the test sentences to test.txt
    with open('test.txt', 'w') as test_file:
        test_file.write('\n'.join(test_sentences))
    
    print(f"Created {len(train_sentences)} sentences in train.txt")
    print(f"Created {len(test_sentences)} sentences in test.txt")

@click.command()
@click.option('--directory', 
              type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help='Directory containing the text files')
@click.option('--step-size', default=3, type=int, help='Step size for generating sentences')
def main(directory, step_size):
    """Create sentences from text files in the specified directory with given step size."""
    create_sentences(directory, step_size)

if __name__ == '__main__':
    main()
