import sys

from ggnn.utils import build_vocabulary


def main():
    src, dst, subtokens_bound, types_bound = sys.argv[1:]
    subtokens_bound = int(subtokens_bound)
    types_bound = int(types_bound)

    build_vocabulary(src, src, dst, subtokens_bound, types_bound)


if __name__ == '__main__':
    main()
