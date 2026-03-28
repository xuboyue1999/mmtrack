import argparse
import os


def get_folder_names(folder_path):
    return sorted(
        name for name in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, name))
    )


def main():
    parser = argparse.ArgumentParser(description='List sequence folder names under a dataset root.')
    parser.add_argument('folder_path', help='Dataset root directory')
    args = parser.parse_args()

    file_names = get_folder_names(args.folder_path)
    print(file_names)
    print(len(file_names))


if __name__ == '__main__':
    main()
