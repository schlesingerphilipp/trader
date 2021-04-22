import redis
import json

class DataProvider:
    def __init__(self, db):
        redis_config = {
            "host": 'localhost',
            "port": 6379,
            "db": db
        }
        self.redis_conn = redis.Redis(**redis_config)
        self.stocks = self.get_stock_size()

    def load(self, key):
        raw_value = self.redis_conn.get(key).decode("utf-8")
        loaded_value = json.loads(raw_value)
        loaded_value["owning"] = [0 for _ in range(self.stocks)]
        #loaded_value["fake"] = [0]
        return loaded_value

    def close(self):
        del self.redis_conn

    def get_stock_size(self):
        raw_value = self.redis_conn.get(1).decode("utf-8")
        loaded_value = json.loads(raw_value)
        return len(loaded_value["open"])
