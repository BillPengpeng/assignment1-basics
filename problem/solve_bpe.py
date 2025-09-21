import json
import time

import json
import pathlib
from functools import lru_cache
from tests.adapters import run_train_bpe
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode

DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"

def test_train_bpe_TinyStories():
    """
    Ensure that the special tokens are added to the vocabulary and not
    merged with other tokens.
    """
    start_time = time.time()
    # input_path = DATA_PATH / "TinyStoriesV2-GPT4-valid.txt"
    input_path = DATA_PATH / "TinyStoriesV2-GPT4-train.txt"
    print("process:", input_path)
    # input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )

    # Check that the special token is not in the vocab
    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    for word_bytes in vocabs_without_specials:
        assert b"<|" not in word_bytes

    # snapshot.assert_match(
    #     {
    #         "vocab_keys": set(vocab.keys()),
    #         "vocab_values": set(vocab.values()),
    #         "merges": merges,
    #     },
    # )
    end_time = time.time()
    # 20250920 TinyStoriesV2-GPT4-valid.txt 耗时 83.24239468574524
    # 20250921 TinyStoriesV2-GPT4-valid.txt 耗时 85.79178786277771  TinyStoriesV2-GPT4-train.txt 耗时 571.4011189937592
    print("test_train_bpe_TinyStories:", end_time - start_time)

    output_vocab_json = DATA_PATH / "TinyStoriesV2-GPT4-vacab.json"
    output_merge = DATA_PATH / "TinyStoriesV2-GPT4-merge.txt"
    bytes_to_unicode_dict = gpt2_bytes_to_unicode()

    vocab_index_dict = dict()
    for k, v in vocab.items():
        # import pdb;pdb.set_trace()
        unicode_v = [bytes_to_unicode_dict[ch] for ch in v]
        vocab_index_dict[''.join(unicode_v)] = k
        # import pdb;pdb.set_trace()

    json_str = json.dumps(vocab_index_dict, indent=4)
    with open(output_vocab_json, 'w') as fp:
        fp.write(json_str)
    with open(output_merge, 'w') as fp:
        for pair in merges:
            unicode_pair_0 = "".join([bytes_to_unicode_dict[ch] for ch in pair[0]])
            unicode_pair_1 = "".join([bytes_to_unicode_dict[ch] for ch in pair[1]])
            fp.write(unicode_pair_0 + ' ' + unicode_pair_1 + '\n')

def test_train_bpe_owe():
    """
    Ensure that the special tokens are added to the vocabulary and not
    merged with other tokens.
    """
    start_time = time.time()
    # input_path = DATA_PATH / "owt_valid.txt"
    input_path = DATA_PATH / "owt_train.txt"
    print("process:", input_path)
    # input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
    )

    # Check that the special token is not in the vocab
    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    for word_bytes in vocabs_without_specials:
        assert b"<|" not in word_bytes

    # snapshot.assert_match(
    #     {
    #         "vocab_keys": set(vocab.keys()),
    #         "vocab_values": set(vocab.values()),
    #         "merges": merges,
    #     },
    # )
    end_time = time.time()
    # 20250921 owt_valid.txt 耗时 85.79178786277771  owt_train.txt 耗时 571.4011189937592
    print("test_train_bpe_owe:", end_time - start_time)

    output_vocab_json = DATA_PATH / "owt-vacab.json"
    output_merge = DATA_PATH / "owt-merge.txt"
    bytes_to_unicode_dict = gpt2_bytes_to_unicode()

    vocab_index_dict = dict()
    for k, v in vocab.items():
        # import pdb;pdb.set_trace()
        unicode_v = [bytes_to_unicode_dict[ch] for ch in v]
        vocab_index_dict[''.join(unicode_v)] = k
        # import pdb;pdb.set_trace()

    json_str = json.dumps(vocab_index_dict, indent=4)
    with open(output_vocab_json, 'w') as fp:
        fp.write(json_str)
    with open(output_merge, 'w') as fp:
        for pair in merges:
            unicode_pair_0 = "".join([bytes_to_unicode_dict[ch] for ch in pair[0]])
            unicode_pair_1 = "".join([bytes_to_unicode_dict[ch] for ch in pair[1]])
            fp.write(unicode_pair_0 + ' ' + unicode_pair_1 + '\n')


if __name__ == "__main__":
    test_train_bpe_TinyStories()
    # test_train_bpe_owe()

