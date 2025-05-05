from typing import Literal


DEDUPLICATION_METHODS = Literal["exact", "minhash", "bloomfilter", "minhash_bloom_hybrid"]
