import redis

def pretty_print_key_value(r, key):
    key_type = r.type(key).decode('utf-8')
    if key_type == 'string':
        value = r.get(key)
        print(f"{key}: {value}")
    elif key_type == 'hash':
        value = r.hgetall(key)
        print(f"{key}: {value}")
    elif key_type == 'list':
        value = r.lrange(key, 0, -1)
        print(f"{key}: {value}")
    elif key_type == 'set':
        value = r.smembers(key)
        print(f"{key}: {value}")
    elif key_type == 'zset':
        value = r.zrange(key, 0, -1, withscores=True)
        print(f"{key}: {value}")
    else:
        print(f"{key}: (type {key_type}) not handled")

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Get all keys
keys = r.keys('*')
print("Keys:", keys)

# Get values of keys
for key in keys:
    pretty_print_key_value(r, key)
