import chromadb
import inspect
from chromadb.config import Settings

print('chromadb version:', chromadb.__version__)
print('Settings signature:', inspect.signature(Settings))
print('Settings repr:', Settings)
print('Settings doc sample:\n', Settings.__doc__[:800])
