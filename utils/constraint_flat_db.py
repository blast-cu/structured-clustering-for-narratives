import lmdb
import struct


class ConstraintFlatDB:
    def __init__(self, db_path='./constraint_flat_db'):
        self.env = lmdb.open(db_path, map_size=100 * 1024 ** 3)

    def _make_key(self, dict_key, set_member):
        """Convert (dict_key, set_member) to bytes"""
        return struct.pack('2i', dict_key, set_member)

    def add_tuple(self, tuple_pair):
        """Needs write=True for modifications"""
        dict_key, set_member = tuple_pair
        with self.env.begin(write=True) as txn:
            txn.put(self._make_key(dict_key, set_member), b'')

    def add_tuples(self, tuple_list):
        """Add multiple tuples in batch"""
        with self.env.begin(write=True) as txn:
            for dict_key, set_member in tuple_list:
                txn.put(self._make_key(dict_key, set_member), b'')

    def contains_tuple(self, tuple_pair):
        """Read-only - no write=True needed"""
        dict_key, set_member = tuple_pair
        with self.env.begin() as txn:  # Faster, allows concurrency
            return txn.get(self._make_key(dict_key, set_member)) is not None

    def read_all_tuples(self):
        """Memory-efficient iterator over all tuples"""
        with self.env.begin() as txn:
            cursor = txn.cursor()
            cursor.first()
            while True:
                key = cursor.key()
                if len(key) == 8:
                    dict_key, set_member = struct.unpack('2i', key)
                    yield (dict_key, set_member)
                if not cursor.next():
                    break

    def close(self):
        self.env.close()