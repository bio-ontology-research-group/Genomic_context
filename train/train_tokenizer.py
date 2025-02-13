import click as ck
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer
from pathlib import Path
import glob

@ck.command()
@ck.argument('directory', type=ck.Path(exists=True, file_okay=False, dir_okay=True), help="Directory containing '.txt' formatted corpus files")
@ck.option('--vocab-size', type=int, default=545000, help='Size of the vocabulary')
@ck.option('--output-file', type=str, default='interpro_tokenizer.json', help='Filename to save tokenizer vocabulary')
def main(directory, vocab_size, output_file):
    files = glob.glob(str(Path(directory) / '*.txt'))
    tokenizer = Tokenizer(WordLevel())
    tokenizer.pre_tokenizer = WhitespaceSplit()
    trainer = WordLevelTrainer(
        vocab_size=vocab_size,
        special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>"
        ]
    )
    
    tokenizer.train(files=files, trainer=trainer)
    tokenizer.save(output_file)
    print(f"Tokenizer saved to {output_file}")

if __name__ == '__main__':
    main()
