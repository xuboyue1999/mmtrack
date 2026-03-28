import lmdb
import numpy as np
import cv2
import json
import argparse

LMDB_ENVS = dict()
LMDB_HANDLES = dict()
LMDB_FILELISTS = dict()


def get_lmdb_handle(name):
    global LMDB_HANDLES, LMDB_FILELISTS
    item = LMDB_HANDLES.get(name, None)
    if item is None:
        env = lmdb.open(name, readonly=True, lock=False, readahead=False, meminit=False)
        LMDB_ENVS[name] = env
        item = env.begin(write=False)
        LMDB_HANDLES[name] = item

    return item


def decode_img(lmdb_fname, key_name):
    handle = get_lmdb_handle(lmdb_fname)
    binfile = handle.get(key_name.encode())
    if binfile is None:
        print("Illegal data detected. %s %s" % (lmdb_fname, key_name))
    s = np.frombuffer(binfile, np.uint8)
    x = cv2.cvtColor(cv2.imdecode(s, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return x


def decode_str(lmdb_fname, key_name):
    handle = get_lmdb_handle(lmdb_fname)
    binfile = handle.get(key_name.encode())
    string = binfile.decode()
    return string


def decode_json(lmdb_fname, key_name):
    return json.loads(decode_str(lmdb_fname, key_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read a key from an LMDB dataset.")
    parser.add_argument("lmdb_path", help="Path to the LMDB directory")
    parser.add_argument("key", help="Key to decode")
    parser.add_argument(
        "--type",
        choices=("str", "json"),
        default="str",
        help="Decode value as plain text or JSON",
    )
    args = parser.parse_args()

    if args.type == "json":
        print(decode_json(args.lmdb_path, args.key))
    else:
        print(decode_str(args.lmdb_path, args.key))
