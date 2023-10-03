import os
import re
import codecs
import pinecone
import time
from sentence_transformers import SentenceTransformer
import torch

class Buffer:
    CAPACITY = 6000

    def __init__(self) -> None:
        self.__data = []
        self.__data_len = 0
        self.id = ""
        self.page = 0
        self.paragraph = 0

    def reset(self, page: int, paragraph: int) -> None:
        self.__data = []
        self.__data_len = 0
        self.id = f"{page}-{paragraph}"
        self.page = page
        self.paragraph = paragraph

    def could_be_appended(self, s: str) -> bool:
        return self.__data_len + len(s) <= Buffer.CAPACITY

    def append(self, s: str) -> None:
        if not self.could_be_appended(s):
            raise ValueError(f"Failed to append {self.id} with len={len(s)}")
        
        self.__data.append(s)
        self.__data_len += len(s)

    @property
    def data_len(self) -> int:
        return self.__data_len

    def to_str(self) -> str:
        return "\n".join(self.__data)


class Batch:
    def __init__(self, size: int) -> None:
        self.__size = size
        self.__records = []
    
    def add(self, id, embedding, metadata):
        if len(self.__records) == self.__size:
            raise ValueError(f"Failed to append {id} into this batch")
        self.__records.append((id, embedding, metadata))

    def is_empty(self) -> bool:
        return len(self.__records) == 0

    def is_full(self):
        return len(self.__records) == self.__size

    def to_records(self):
        return self.__records

class Collection:
    BATCH_SIZE = 128

    def __init__(self, em_model, index: pinecone.GRPCIndex) -> None:
        self.__embed_model = em_model
        self.__index = index
        self.__batch = Batch(Collection.BATCH_SIZE)

    def add(
        self,
        ids,
        embedding = None,
        metadata = None,
        document = None):
        if not embedding:
            embedding = self.__embed_model.encode(document)
        if self.__batch.is_full():
            self.flush()
        self.__batch.add(ids, embedding, {**metadata, "text": document})

    def flush(self):
        if not self.__batch.is_empty():
            self.__index.upsert(self.__batch.to_records())
            self.__batch = Batch(Collection.BATCH_SIZE)

    def query(
        self,
        query_texts,
        n_results: int = 10):
        embedding = self.__embed_model.encode(query_texts)
        return self.__index.query(embedding, top_k=n_results, include_metadata=True)

def build_db(collection: Collection, input_path: str):
    buf = Buffer()
    for file_name in os.listdir(input_path):
        m = re.match(r"(\d+)\.txt", file_name)
        if not m:
            continue
        page = int(m.group(1))
        file_path = os.path.join(input_path, file_name)
        if not os.path.isfile(file_path):
            continue
        print(f"Process file {file_name}...")
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            paragraph = 0
            buf.reset(page, paragraph)
            for p in content.split("【PG】"):
                s = p.rstrip()
                if s == '':
                    continue
                #if not buf.could_be_appended(s):
                if buf.data_len > 0:
                    if (buf.data_len > 0):
                        collection.add(ids=buf.id, metadata={"page": buf.page, "paragraph": buf.paragraph}, document= buf.to_str())
                    buf.reset(page, paragraph)
                try:
                    buf.append(s)
                except Exception as e:
                    print(str(e))
                paragraph += 1

        if (buf.data_len > 0):
            collection.add(ids=buf.id, metadata={"page": buf.page, "paragraph": buf.paragraph}, document= buf.to_str())

    collection.flush()

def open_db(name: str, em_model):
    if name not in pinecone.list_indexes():
        pinecone.create_index(
            name=name,
            dimension=em_model.get_sentence_embedding_dimension(),
            metric='cosine'
        )

    # now connect to the index
    return Collection(em_model, pinecone.GRPCIndex(name))

def query(collection: Collection, question: str):
    results = collection.query(
        query_texts=question,
        n_results=5)
    import pprint
    pprint.pprint(results)

em_model = SentenceTransformer('all-MiniLM-L6-v2')
print(em_model)

print(em_model.encode("猜数游戏"))
s = ['检查按键', '滚动场景', '打印表头', '嵌套循环', '全局变量']
for i in s:
    print(em_model.encode(i))
exit(0)

api_key = 'b2cbe122-5bcd-4a36-807d-4d495f681fd1'
env = 'gcp-starter'

pinecone.init(api_key=api_key, environment=env)

collection = open_db("tutorial1", em_model)
#build_db(collection, "./preprocessed/tutorial1.pdf")

query(collection, "python有几个版本")
query(collection, "猜数游戏")
