import os
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
        messages=[{"role": "user", "content": f"Split the following text into paragraphes, and output each paragraph beginning with a '【PG】' identifier: \n\n{text}"}],
    )
    return response

def preprocess_texts(input: str, output: str):
    os.makedirs(output, exist_ok=True)
    for file_name in os.listdir(input):
        input_path = os.path.join(input, file_name)
        output_path = os.path.join(output, file_name)
        if os.path.isfile(input_path):
            print(f"Process file {file_name}...")
            if (os.path.exists(output_path)):
                print(f"Output file {file_name} exists, skip it.")
                continue
            with open(input_path, 'r', encoding='utf-8') as file:
                content = file.read()
                result = preprocess(content)
                with open(output_path, 'w', encoding="utf-8") as f:
                    f.write(result)

#pdf2txts("./data/tutorial2.pdf", "./tmp/tutorial2.pdf")
#count_char("./tmp/tutorial2.pdf")
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:8118"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:8118"
print(os.getenv("HTTPS_PROXY"))
print(os.getenv("HTTP_PROXY"))

preprocess_texts("./tmp/tutorial2.pdf", "./preprocessed/tutorial2.pdf")

# def fill_db():
#     reader = PdfReader("./data/tutorial1.pdf")
#     for page in reader.pages:
#         if page.page_number % 10 == 0:
#             print(f"Process page {page.page_number}...") 
#         text = page.extract_text()

#         collection.add(ids=[], metadatas=[], documents=[])


# db_client = chromadb.PersistentClient("./chroma")
# collection = db_client.get_or_create_collection("tutorial1")

# # Query/search 2 most similar results. You can also .get by id
# results = collection.query(
#     query_texts=["This is a query document"],
#     n_results=2,
#     # where={"metadata_field": "is_equal_to_this"}, # optional filter
#     # where_document={"$contains":"search_string"}  # optional filter
# )