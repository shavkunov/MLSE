import re
from typing import List, Generator, Any, Dict


def batch_generator(data_generator: Generator[Any, None, None], batch_size: int) -> Generator[List[Any], None, None]:
    while True:
        buffer = []
        try:
            for _ in range(batch_size):
                buffer.append(next(data_generator))
            yield buffer
        except StopIteration:
            if buffer:
                yield buffer
            break


def save_id_storage(storage: Dict[str, int], dst):
    with open(dst, 'w', encoding='utf-8') as file:
        file.write("id,value\n")
        for token, token_id in storage.items():
            file.write(f"{token_id},\"{token}\"\n")


def load_id_storage(src_path):
    result = dict()
    with open(src_path, 'r', encoding='utf-8') as file:
        file.readline()
        for line in file:
            line = line.strip()
            if line:
                token_id, token = line.split(',', maxsplit=1)
                result[token[1:-1]] = int(token_id)
    return result


class Tokenizer:
    def __init__(self):
        self.normalization_regexp = re.compile("[^a-zA-Z]")
        self.split_regexp = re.compile("(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|\\s+")

    def tokenize(self, token: str) -> List[str]:
        token = self.normalization_regexp.sub(" ", token).strip()
        buffer = map(str.strip, self.split_regexp.split(token))
        buffer = map(str.lower, buffer)
        buffer = filter(len, buffer)
        return list(buffer)
