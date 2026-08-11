import inspect
import chromadb
from chromadb.api.client import Client
from chromadb.config import Settings

print('chromadb version:', chromadb.__version__)
print('Client init signature:', inspect.signature(Client.__init__))
print('Client class doc first line:', Client.__doc__.strip().splitlines()[0] if Client.__doc__ else 'None')
print('Settings params:', inspect.signature(Settings))
print('Trying default client creation...')
try:
    client = Client()
    print('Client default created successfully')
    client.shutdown()
except Exception as e:
    print('Default client error:', repr(e))

print('Trying persistent client with Settings...')
try:
    settings = Settings(persist_directory='vectorstore', is_persistent=True)
    client = Client(settings=settings)
    print('Client with persistent settings created successfully')
    client.shutdown()
except Exception as e:
    print('Persistent settings error:', repr(e))
