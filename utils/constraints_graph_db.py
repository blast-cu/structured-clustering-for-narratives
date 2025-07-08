import pickle

import lmdb
import struct


class ConstraintGraphDB:
    def __init__(self, db_path='./constraint_graph_db'):
        self.env = lmdb.open(db_path, map_size=10 * 1024 ** 3)

    def _key_to_bytes(self, dict_key):
        return struct.pack('i', dict_key)

    def add_set(self, dict_key, members):
        """Store entire set for dict_key (replaces existing)"""
        with self.env.begin(write=True) as txn:
            txn.put(self._key_to_bytes(dict_key), pickle.dumps(set(members)))

    def add_sets(self, dict_of_sets):
        """Add multiple sets (replaces existing)"""
        with self.env.begin(write=True) as txn:
            for dict_key, members in dict_of_sets.items():
                txn.put(self._key_to_bytes(dict_key), pickle.dumps(set(members)))

    def update_sets(self, dict_of_sets):
        """Add if key doesn't exist, merge if key exists"""
        with self.env.begin(write=True) as txn:
            for dict_key, new_members in dict_of_sets.items():
                existing_data = txn.get(self._key_to_bytes(dict_key))
                existing_set = pickle.loads(existing_data) if existing_data else set()
                merged_set = existing_set | set(new_members)
                txn.put(self._key_to_bytes(dict_key), pickle.dumps(merged_set))

    def add_to_set(self, dict_key, member):
        """Add single member to set"""
        members = self.get_set(dict_key)
        members.add(member)
        self.add_set(dict_key, members)

    def contains(self, dict_key, member):
        """Check if member is in set for dict_key"""
        members = self.get_set(dict_key)
        return member in members

    def get_set(self, dict_key):
        """Get set for dict_key"""
        with self.env.begin() as txn:
            data = txn.get(self._key_to_bytes(dict_key))
            return pickle.loads(data) if data else set()

    def read_all_sets(self):
        """Memory-efficient iterator over all (dict_key, set) pairs"""
        with self.env.begin() as txn:
            cursor = txn.cursor()
            cursor.first()
            while True:
                key_bytes = cursor.key()
                value_bytes = cursor.value()
                dict_key = struct.unpack('i', key_bytes)[0]
                members = pickle.loads(value_bytes)
                yield (dict_key, members)
                if not cursor.next():
                    break

    def close(self):
        self.env.close()