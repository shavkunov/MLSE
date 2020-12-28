import sys

from ggnn.utils import prepare_data


def main():
    src, dst, labels = sys.argv[1:]

    prepare_data(src, labels, dst, dst)


if __name__ == '__main__':
    main()
