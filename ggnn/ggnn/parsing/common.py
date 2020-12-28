import hashlib
import os
import re
from os import path
from random import Random
from typing import Generator

from .parallel_processing import DataLoader


class FilesLoader(DataLoader):

    def __init__(self, src, file_size_limit=-1, limit=-1,
                 folder_pattern='.*', file_pattern='.*', seed=0):
        self.src = src
        self.file_size_limit = file_size_limit
        self.limit = limit
        self.folder_pattern = folder_pattern
        self.file_pattern = file_pattern
        self.seed = seed

    def load(self) -> Generator[str, None, None]:
        for folder, _, files in os.walk(self.src):
            relative_path = os.path.relpath(folder, self.src)
            if not re.fullmatch(self.folder_pattern, relative_path):
                continue
            for filename in self.__select_files(relative_path, files):
                with open(os.path.join(folder, filename), 'r', encoding='utf-8') as file:
                    yield relative_path, filename, file.read()

    def __select_files(self, relative_path, files):
        files = [file for file in files if self.__check_file(relative_path, file)]
        files.sort()
        if self.limit == -1 or len(files) <= self.limit:
            return files
        local_seed = int(hashlib.sha1(relative_path.encode('utf-8')).hexdigest(), 16) % (2 ** 64)
        return Random(self.seed ^ local_seed).sample(files, self.limit)

    def __check_file(self, relative_path, filename):
        file_path = path.join(self.src, relative_path, filename)
        if not path.isfile(file_path):
            return False
        if not re.fullmatch(self.file_pattern, filename):
            return False
        if self.file_size_limit != -1 and path.getsize(file_path) > self.file_size_limit:
            return False
        return True


class PathsLoader(DataLoader):

    def __init__(self, src, file_size_limit=-1, limit=-1,
                 folder_pattern='.*', file_pattern='.*', seed=0):
        self.src = src
        self.file_size_limit = file_size_limit
        self.limit = limit
        self.folder_pattern = folder_pattern
        self.file_pattern = file_pattern
        self.seed = seed

    def load(self) -> Generator[str, None, None]:
        for folder, _, files in os.walk(self.src):
            relative_path = os.path.relpath(folder, self.src)
            if not re.fullmatch(self.folder_pattern, relative_path):
                continue
            print(relative_path)
            for filename in self.__select_files(relative_path, files):
                yield relative_path, filename, os.path.join(folder, filename)

    def __select_files(self, relative_path, files):
        files = [file for file in files if self.__check_file(relative_path, file)]
        files.sort()
        if self.limit == -1 or len(files) <= self.limit:
            return files
        local_seed = int(hashlib.sha1(relative_path.encode('utf-8')).hexdigest(), 16) % (2 ** 64)
        return Random(self.seed ^ local_seed).sample(files, self.limit)

    def __check_file(self, relative_path, filename):
        file_path = path.join(self.src, relative_path, filename)
        if not path.isfile(file_path):
            return False
        if not re.fullmatch(self.file_pattern, filename):
            return False
        if self.file_size_limit != -1 and path.getsize(file_path) > self.file_size_limit:
            return False
        return True
