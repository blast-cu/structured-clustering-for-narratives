import lmdb
import struct


class ConstraintDB:
    def __init__(self, db_path='./tuple_db'):
        self.env = lmdb.open(db_path, map_size=10 * 1024 ** 3)

    def tuple_to_bytes(self, t):
        return struct.pack('2i', *t)  # Assumes 2-integer tuples

    def add_batch(self, tuple_list):
        """Add multiple tuples in one transaction"""
        with self.env.begin(write=True) as txn:
            for key_tuple in tuple_list:
                txn.put(self.tuple_to_bytes(key_tuple), b'')
        return len(tuple_list)

    def add_single(self, key_tuple):
        """Add single tuple"""
        with self.env.begin(write=True) as txn:
            txn.put(self.tuple_to_bytes(key_tuple), b'')

    def exists(self, key_tuple):
        """Check if tuple exists"""
        with self.env.begin() as txn:
            return txn.get(self.tuple_to_bytes(key_tuple)) is not None

    def exists_batch(self, tuple_list):
        """Check multiple tuples in one transaction"""
        results = {}
        with self.env.begin() as txn:
            for key_tuple in tuple_list:
                results[key_tuple] = txn.get(self.tuple_to_bytes(key_tuple)) is not None
        return results

    def close(self):
        self.env.close()