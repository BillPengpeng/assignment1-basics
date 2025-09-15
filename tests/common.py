from __future__ import annotations

import pathlib
from functools import lru_cache

FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent) / "fixtures"


# gpt2_bytes_to_unicode函数通过将 256 个字节值（0-255）映射到可打印的 Unicode 字符，解决了不可打印字符在词表存储中的问题。其核心逻辑是：
# 直接保留部分可打印字节的原样；
# 为不可打印字节分配高位的 Unicode 码点（256 及以上），生成唯一的可打印字符；
# 最终生成字节到可打印字符的字典，供分词器在加载词表（vocab.json）时还原原始字节

@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.

    # range(ord("!"), ord("~") + 1)：覆盖 ASCII 中可打印的标点、数字、字母（!"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_abcdefghijklmnopqrstuvwxyz{|}~`），对应字节值 33-126（共 94 个）。
    # range(ord("¡"), ord("¬") + 1)：覆盖部分扩展拉丁字符（如 ¡倒置感叹号，Unicode 码点 241；¬逻辑非符号，码点 172），对应字节值 161-172（共 12 个）。
    # range(ord("®"), ord("ÿ") + 1)：覆盖更多扩展拉丁字符（如 ®注册商标，码点 174；ÿ带分音符的 y，码点 255），对应字节值 174-255（共 82 个）。
    # 三者合计：94 + 12 + 82 = 188 个字节值，这些字节本身已是可打印字符，无需额外处理。
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    # 为剩余 68 个不可打印的字节（256 - 188 = 68）分配可打印的 Unicode 字符。
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d
