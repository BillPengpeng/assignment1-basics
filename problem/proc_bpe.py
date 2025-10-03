import json
import time
import numpy as np

from tqdm import tqdm
import cProfile
import pstats
from memory_profiler import profile

import json
import pathlib
from functools import lru_cache
from tests.adapters import run_train_bpe, BPETokenizer, find_chunk_boundaries
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode

DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"

# @profile
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
    max_len_line = ""
    with open(output_merge, 'w') as fp:
        for pair in merges:
            unicode_pair_0 = "".join([bytes_to_unicode_dict[ch] for ch in pair[0]])
            unicode_pair_1 = "".join([bytes_to_unicode_dict[ch] for ch in pair[1]])
            line = unicode_pair_0 + ' ' + unicode_pair_1
            if len(line) > len(max_len_line):
                max_len_line = line
            fp.write(line + '\n')
    print("max_len_line:", max_len_line)

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


def test_bpe_tokenize_TinyStories_sample():
    vocab_json = DATA_PATH / "TinyStoriesV2-GPT4-vacab.json"
    merge = DATA_PATH / "TinyStoriesV2-GPT4-merge.txt"
    
    tokenizer = BPETokenizer.from_files(
        vocab_filepath=vocab_json,
        merges_filepath=merge,
        special_tokens=["<|endoftext|>"]
    )
    subproc_cnt = 1
    chunk_per_subproc = 1000
    num_samples = 5

    # TinyStoriesV2-GPT4-valid.txt
    sample_len = 0
    encoded_ids_len = 0
    input_path = DATA_PATH / "TinyStoriesV2-GPT4-valid.txt"

    with open(input_path, "rb") as f:
        # 获取分块边界
        boundaries = find_chunk_boundaries(
            f, subproc_cnt * chunk_per_subproc, "<|endoftext|>".encode("utf-8")
        )

        start_time = time.time()
        for idx in range(num_samples):
            sample_idx = (chunk_per_subproc * subproc_cnt // num_samples) * idx
            start, end = boundaries[sample_idx], boundaries[sample_idx + 1]
            f.seek(start)
            content = f.read(end - start).decode("utf-8", errors="ignore")
            encoded_ids = tokenizer.encode(content)
            num_bytes = len(bytes(content, encoding="utf-8"))  # @inspect num_bytes
            num_tokens = len(encoded_ids)                      # @inspect num_tokens
            sample_len += num_bytes
            encoded_ids_len += num_tokens
        end_time = time.time()

    # sample TinyStoriesV2-GPT4-valid.txt throughput 5904.679196840561 bytes/second
    # sample TinyStoriesV2-GPT4-valid.txt compression_ratio: 4.11243820589615
    print("sample TinyStoriesV2-GPT4-valid.txt throughput {} bytes/second".format(num_bytes / (end_time - start_time)))
    print("sample TinyStoriesV2-GPT4-valid.txt compression_ratio:", sample_len / encoded_ids_len)

    # owt_valid.txt
    sample_len = 0
    encoded_ids_len = 0
    input_path = DATA_PATH / "owt_valid.txt"
    with open(input_path, "rb") as f:
        # 获取分块边界
        boundaries = find_chunk_boundaries(
            f, subproc_cnt * chunk_per_subproc, "<|endoftext|>".encode("utf-8")
        )

        start_time = time.time()
        for idx in range(num_samples):
            sample_idx = (chunk_per_subproc * subproc_cnt // num_samples) * idx
            start, end = boundaries[sample_idx], boundaries[sample_idx + 1]
            f.seek(start)
            content = f.read(end - start).decode("utf-8", errors="ignore")
            encoded_ids = tokenizer.encode(content)
            num_bytes = len(bytes(content, encoding="utf-8"))  # @inspect num_bytes
            num_tokens = len(encoded_ids)                      # @inspect num_tokens
            sample_len += num_bytes
            encoded_ids_len += num_tokens
        end_time = time.time()

    # sample owt_valid.txt throughput 2825.87244634013 bytes/second
    # sample owt_valid.txt compression_ratio: 3.218193049535109
    print("sample owt_valid.txt throughput {} bytes/second".format(num_bytes / (end_time - start_time)))
    print("sample owt_valid.txt compression_ratio:", sample_len / encoded_ids_len)

def test_bpe_tokenize_TinyStories_proc():
    vocab_json = DATA_PATH / "TinyStoriesV2-GPT4-vacab.json"
    merge = DATA_PATH / "TinyStoriesV2-GPT4-merge.txt"
    
    tokenizer = BPETokenizer.from_files(
        vocab_filepath=vocab_json,
        merges_filepath=merge,
        special_tokens=["<|endoftext|>"]
    )
    subproc_cnt = 1
    chunk_per_subproc = 1000
    num_samples = 5

    # TinyStoriesV2-GPT4-valid.txt
    sample_len = 0
    encoded_ids_len = 0
    input_path = DATA_PATH / "TinyStoriesV2-GPT4-valid.txt"
    output_path = DATA_PATH / "TinyStoriesV2-GPT4-valid-encoded2.npy"
    # input_path = DATA_PATH / "TinyStoriesV2-GPT4-train.txt"
    # output_path = DATA_PATH / "TinyStoriesV2-GPT4-train-encoded2.npy"

    # result_list
    result_list = list()
    with open(input_path, "rb") as f:
        # 获取分块边界
        boundaries = find_chunk_boundaries(
            f, subproc_cnt * chunk_per_subproc, "<|endoftext|>".encode("utf-8")
        )      
        start_time = time.time()
        for idx in tqdm(range(subproc_cnt * chunk_per_subproc)):
            start, end = boundaries[idx], boundaries[idx + 1]
            f.seek(start)
            content = f.read(end - start).decode("utf-8", errors="ignore")
            encoded_ids = tokenizer.encode(content)
            result_list.extend(encoded_ids)
            # import pdb;pdb.set_trace()
            # break
        end_time = time.time()
        
        print("test_bpe_tokenize_TinyStories_proc {} second".format(end_time - start_time))

    # np.save
    arr = np.array(result_list, dtype=np.uint16)
    print("arr:", arr.shape)
    np.save(output_path, arr)

def test_bpe_tokenize_owt_proc():
    vocab_json = DATA_PATH / "owt-vacab.json"
    merge = DATA_PATH / "owt-merge.txt"
    
    tokenizer = BPETokenizer.from_files(
        vocab_filepath=vocab_json,
        merges_filepath=merge,
        special_tokens=["<|endoftext|>"]
    )
    subproc_cnt = 1
    chunk_per_subproc = 1000 #50000

    # TinyStoriesV2-GPT4-valid.txt
    sample_len = 0
    encoded_ids_len = 0
    # input_path = DATA_PATH / "owt_valid.txt"
    # output_path = DATA_PATH / "owt_valid-encoded.npy"
    input_path = DATA_PATH / "owt_train.txt"
    output_path = DATA_PATH / "owt_train-encoded.npy"

    # result_list
    result_list = list()
    with open(input_path, "rb") as f:
        # 获取分块边界
        boundaries = find_chunk_boundaries(
            f, subproc_cnt * chunk_per_subproc, "<|endoftext|>".encode("utf-8")
        )      
        # print(len(boundaries), subproc_cnt * chunk_per_subproc)
        boundaries_len = len(boundaries)
        start_time = time.time()
        for idx in tqdm(range(boundaries_len - 1)):
            start, end = boundaries[idx], boundaries[idx + 1]
            f.seek(start)
            content = f.read(end - start).decode("utf-8", errors="ignore")
            encoded_ids = tokenizer.encode(content)
            # result_list.extend(encoded_ids)
            arr = np.array(encoded_ids, dtype=np.uint16)
            save_path = output_path.with_suffix(f'.{idx}.npy')
            np.save(save_path, arr)
            # import pdb;pdb.set_trace()
            # break
        end_time = time.time()
        print("test_bpe_tokenize_owt_proc {} second".format(end_time - start_time))

    # np.save
    # arr = np.array(result_list, dtype=np.uint16)
    # print("arr:", arr.shape)
    # np.save(output_path, arr)

def test_bpe_tokenize_owt_merge_proc():
    chunk_per_subproc = 1000
    output_path = DATA_PATH / "owt_train-encoded.npy"

    # 第一步：预先计算最终数组的总长度和数据类型
    total_length = 0
    for idx in range(chunk_per_subproc - 1):
        save_path = output_path.with_suffix(f'.{idx}.npy')
        cur_arr = np.load(save_path)
        total_length += len(cur_arr) # 累加所有数组的长度
        del cur_arr # 立即释放当前数组的内存

    dtype = np.uint16
    print(f"Final array will be of length: {total_length}")

    # 第二步：创建一个内存映射文件作为输出数组
    # 这里先创建一个空文件并设置大小，‘w+’模式表示可读写
    final_arr = np.lib.format.open_memmap(output_path, 
                                         mode='w+', 
                                         dtype=dtype, 
                                         shape=(total_length,))

    # 第三步：增量地将数据写入内存映射文件
    current_index = 0
    for idx in tqdm(range(chunk_per_subproc - 1)):
        save_path = output_path.with_suffix(f'.{idx}.npy')
        cur_arr = np.load(save_path) # 加载一个小块

        chunk_length = len(cur_arr)
        # 将当前小块的数据写入最终数组的对应位置
        final_arr[current_index:current_index + chunk_length] = cur_arr
        current_index += chunk_length

        del cur_arr # 立即释放当前小块的内存

    # 注意：memmap数组（final_arr）会在被删除或程序结束时自动将数据刷写到磁盘
    # 你也可以手动刷新： final_arr.flush()
    print(f"Merge completed. Final array shape: {final_arr.shape}")

if __name__ == "__main__":
    # cProfile.run('test_train_bpe_TinyStories()', 'profile_results')

    # # 使用pstats查看结果
    # p = pstats.Stats('profile_results')
    # p.sort_stats('cumulative').print_stats(10)  # 按累计时间排序，打印前10行
    # test_train_bpe_TinyStories()
    # test_train_bpe_owe()
    # test_bpe_tokenize_TinyStories_sample()
    # test_bpe_tokenize_TinyStories_proc() 
    # test_bpe_tokenize_owt_proc()
    test_bpe_tokenize_owt_merge_proc()

