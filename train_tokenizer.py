import click as ck
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer
from pathlib import Path
import glob


@ck.command()
def main():
    files = glob.glob('/ibex/user/toibazd/InterPro_annotated_genomes/*.txt')
    tokenizer = Tokenizer(WordLevel())
    tokenizer.pre_tokenizer = WhitespaceSplit()
    trainer = WordLevelTrainer(vocab_size=545000,
        special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>"
        ])
    
    tokenizer.train(files=files, trainer=trainer)
    tokenizer.save('interpro_tokenizer.json')

if __name__ == '__main__':
    main()
