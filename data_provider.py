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
        self.stocks = self.get_number_of_stocks()
        print(self.load(1))

    def load(self, key):
        raw_value = self.redis_conn.get(key).decode("utf-8")
        loaded_value = json.loads(raw_value)
        loaded_value["valid_action"] = [1 for _ in range(self.stocks + 1)]
        return loaded_value

    def close(self):
        del self.redis_conn

    def get_number_of_stocks(self):
        raw_value = self.redis_conn.get(1).decode("utf-8")
        loaded_value = json.loads(raw_value)
        return len(loaded_value["open"])
