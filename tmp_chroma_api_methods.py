import chromadb
from chromadb.config import Settings

settings = Settings(persist_directory='vectorstore', is_persistent=True)
client = chromadb.Client(settings=settings)
print('client type:', type(client))
print('client methods:', [m for m in dir(client) if not m.startswith('_')])
try:
    coll = client.get_collection('test_debug_collection')
except Exception as e:
    print('get_collection error', repr(e))
    coll = None
if coll is not None:
    print('collection methods:', [m for m in dir(coll) if not m.startswith('_')])
    try:
        print('supports persist?', hasattr(coll, 'persist'))
    except Exception as e:
        print('persist check error', repr(e))
client.shutdown()
