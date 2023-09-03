import os
import re
import codecs
from pypdf import PdfReader
import chromadb
import g4f

print(os.getcwd())

def pdf2txts(input: str, output: str):
    reader = PdfReader(input)
    os.makedirs(output, exist_ok=True)
    for page in reader.pages:
        print(f"Process page {page.page_number}...") 
        with open(os.path.join(output, f"{page.page_number}.txt"), 'w', encoding="utf-8") as f:
            f.write(page.extract_text())


def count_char(input: str):
    for file_name in os.listdir(input):
        file_path = os.path.join(input, file_name)
        if os.path.isfile(file_path):
            with codecs.open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                num_char = len(content)
                print(f"{file_path}: {num_char}")

def preprocess(text: str):
    response = g4f.ChatCompletion.create(
        model=g4f.models.gpt_35_turbo,
        provider=g4f.Provider.DeepAi,
        #messages=[{"role": "user", "content": f"Split the following text into paragraphes, and output the original text of each paragraph beginning with a '【PG】' label: \n\n{text}"}],
        messages=[{"role": "user", "content": f"将以下文字拆分为段落, 并在输出每一个段落原文前加上 '【PG】' 标签: \n\n{text}"}],
    )
    return response

def preprocess_texts(input: str, output: str):
    os.makedirs(output, exist_ok=True)
    for file_name in os.listdir(input):
        input_path = os.path.join(input, file_name)
        if os.path.isfile(input_path):
            print(f"Process file {file_name}...")
            output_path = os.path.join(output, file_name)
            if (os.path.exists(output_path)):
                print(f"Output file {file_name} exists, skip it.")
                continue
            with open(input_path, 'r', encoding='utf-8') as file:
                content = file.read()
                result = preprocess(content)
                with open(output_path, 'w', encoding="utf-8") as f:
                    f.write(result)

def open_db(db_path: str, db_name: str):
    client = chromadb.PersistentClient(db_path)
    return client.get_or_create_collection(db_name)

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
        return self.__data_len + 1 + len(s) <= Buffer.CAPACITY

    def append(self, s: str) -> None:
        if not self.could_be_appended(s):
            raise ValueError(f"Fail to append {self.id} with len={len(s)}")
        
        self.__data.append("\n")
        self.__data.append(s)
        self.__data_len += 1 + len(s)

    @property
    def data_len(self) -> int:
        return self.__data_len

    def to_str(self) -> str:
        return "".join(self.__data)
    
def build_db(collection: chromadb.Collection, input_path: str):
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
                if p.strip() == '':
                    continue
                if not buf.could_be_appended(p):
                    if (buf.data_len > 0):
                        collection.add(ids=buf.id, metadatas={"page": buf.page, "paragraph": buf.paragraph}, documents= buf.to_str())
                    buf.reset(page, paragraph)
                try:
                    buf.append(p)
                except Exception as e:
                    print(str(e))
                paragraph += 1

        if (buf.data_len > 0):
            collection.add(ids=buf.id, metadatas={"page": buf.page, "paragraph": buf.paragraph}, documents= buf.to_str())


#pdf2txts("./data/tutorial2.pdf", "./tmp/tutorial2.pdf")
#count_char("./tmp/tutorial2.pdf")
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:8118"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:8118"
print(os.getenv("HTTPS_PROXY"))
print(os.getenv("HTTP_PROXY"))



#preprocess_texts("./tmp/tutorial1.pdf", "./preprocessed/tutorial1.pdf")

db_client = chromadb.PersistentClient("./chroma")
collection = db_client.get_or_create_collection("tutorial1")
build_db(collection, "./preprocessed/tutorial1.pdf")


# # Query/search 2 most similar results. You can also .get by id
# results = collection.query(
#     query_texts=["This is a query document"],
#     n_results=2,
#     # where={"metadata_field": "is_equal_to_this"}, # optional filter
#     # where_document={"$contains":"search_string"}  # optional filter
# )