import concurrent.futures
import base64
import datetime
import hashlib
import hmac
import json
import os
import re
import codecs
import ssl
from time import mktime
from urllib.parse import urlencode, urlparse
from wsgiref.handlers import format_date_time
from pypdf import PdfReader
import chromadb
import g4f
from chromadb.utils import embedding_functions
import websockets
import SparkApi
import numpy as np
import sentence_transformers

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

class SparkModel:
    SPARK_URL = "ws://spark-api.xf-yun.com/v2.1/chat"  # v2.0环境的地址
    DOMAIN = "generalv2"    # v2.0版本
    def __init__(self, app_id, app_key, api_secret):
        self.__app_id = app_id
        self.__app_key = app_key
        self.__api_secret = api_secret
        self.__host = urlparse(SparkModel.SPARK_URL).netloc
        self.__path = urlparse(SparkModel.SPARK_URL).path

    def __gen_url(self):
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + self.__host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.__path + " HTTP/1.1"

        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": self.__host
        }
        # 拼接鉴权参数，生成url
        url = self.Spark_url + '?' + urlencode(v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        return url

    def __gen_params(self, question):
        return {
            "header": {
                "app_id": self.__app_id,
                "uid": "1234"
            },
            "parameter": {
                "chat": {
                    "domain": self.DOMAIN,
                    "random_threshold": 0.5,
                    "max_tokens": 2048,
                    "auditing": "default"
                }
            },
            "payload": {
                "message": {
                    "text": question
                }
            }
        }
    
    def invoke(self, question: str) -> str:
        future = concurrent.futures.Future()
        response = []

        def success():
            if not future.done: future.set_result("".join(response))
        
        def fail(e):
            if not future.done: future.set_exception(e)

        def on_message(ws, msg):
            data = json.loads(msg)
            code = data['header']['code']
            if code != 0:
                fail(IOError(f'Error: {code}, {data}'))
                ws.close()
            else:
                choices = data["payload"]["choices"]
                status = choices["status"]
                content = choices["text"][0]["content"]
                response.append(content)
                if status == 2:
                    future.set_result("".join(response))
                    ws.close()

        ws = websockets.WebSocketApp(
            self.__gen_url(),
            on_message=on_message,
            on_error = lambda ws, e: fail(e),
            on_close = lambda ws: fail(IOError("Expect the last msg")))
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        return future.result(10)

class ReAct:
    def __init__(self, model) -> None:
        self.__model = model
        self.__action_list = []
        self.__action_map = {}

    def add_action(self, name:str, description, func):
        self.__action_list.append((name, description, func))
        self.__action_map[name] = func

    TEMPLATE = """通过交错使用【思考】、【行动】、【观察】这三种步骤来解决回答问题的任务。
【思考】用来推理当前的状况，说明行动的动机；
【行动】可以有如下类型：
{action_list}
2. finish(回答)：返回回答并结束这个任务
【观察】则用于给出上一次【行动】的结果。

问题：{question}

"""
    ACTION_PATTERN = re.compile("【行动】：\\(.*?\\)\\(.*?\\)")
    def invoke(self, question, max_steps=8):
        al = ""
        for i in range(len(self.__action_list)):
            a = self.__action_list[i]
            al += f"{i+1}. {a[0]}{a[1]}\n"
        al += f"{len(self.__action_list)+1}. finish(回答)：返回回答并结束这个任务\n"
        prompt = self.TEMPLATE.format(al, question)

        for i in range(max_steps):
            prompt += f"\nRound {i}\n"
            r = self.__model.invoke(prompt)
            m = self.ACTION_PATTERN.search(r)
            if not m:
                raise Exception("No action in answer: {r}")
            if m[1] == "finish":
                return f"最终答案是：{m[2]}"
            if m[1] not in self.__action_map:
                raise Exception("Unknown action in answer: {r}")

            o = self.__action_map[m[1]](m[2])
            prompt += r
            prompt += f"\n【观察】：{o}\n"
        return "我尽力了..."

class Assistant:
    def __init__(self, db_path: str, db_name: str, embedding_model_name: str) -> None:
        self.__db_client = chromadb.PersistentClient(db_path)
        self.__embedding_model = sentence_transformers.SentenceTransformer(embedding_model_name)
        self.__db_collection = self.__db_client.get_or_create_collection(
            name=db_name,
            metadata={"hnsw:space": "ip"},
            embedding_function=None)

    def add_page_deprecated(self, page_no: int, page_text: str):
        buf = Buffer()
        paragraph = 0
        buf.reset(page_no, paragraph)
        for p in page_text.split("【PG】"):
            s = p.rstrip()
            if len(s) == 0:
                continue
            #if not buf.could_be_appended(s):
            if buf.data_len > 0:
                if (buf.data_len > 0):
                    self.__db_collection.add(ids=buf.id, metadatas={"page": buf.page, "paragraph": buf.paragraph}, documents= buf.to_str())
                buf.reset(page_no, paragraph)
            try:
                buf.append(s)
            except Exception as e:
                print(str(e))
            paragraph += 1

        if (buf.data_len > 0):
            self.__db_collection.add(ids=buf.id, metadatas={"page": buf.page, "paragraph": buf.paragraph}, documents= buf.to_str())


    def add_page0(self, page_no: int, page_text: str):
        """
        对每个 page 处理的算法是A：
        1. 将 page 分割为多个 segment（可以是 sentence，paragraph 等），并计算 embedding；
        2. 将 segment embedding 单位化，此时 cosine similarity 值不变，且其计算过程等效于 inner-product，从而具有了可加性；
        3. 将 page 中所有单位化的 segment embedding 相加，作为该 page 对应的 embedding；
        4. 使用 inner-product 计算 query 和 page embedding 的相似度，由于 page embedding 未做单位化，其结果等效于 query 与每个 segment 的 cosine similarity 之和；
            * 这种做法，会让相关 segment 较多的 page 有更大优势，避免文本较少的 page 排在前面；
            * 实际使用中，由于不相关的向量也会贡献一定的相似度值，效果不理想
        """

        e = np.zeros(self.__embedding_model.get_sentence_embedding_dimension())
        n = 0
        splits = [p.rstrip() for p in page_text.split("【PG】")]
        for p in splits:
            if len(p) == 0:
                continue
            t = self.__embedding_model.encode(p)
            t /= np.linalg.norm(t)
            e += t
            n += 1
        if n > 0:
            self.__db_collection.upsert(ids=str(page_no), embeddings=e.tolist(), metadatas={"page": page_no, "n_paragraph": n}, documents="\n".join(splits))

    __SPLIT_PATTERN = re.compile("[;!?；！？。\n]|\\.\\s")
    def add_page(self, page_no: int, page_text: str):
        """
        对每个 page 处理的算法是A：
        1. 将 page 按标点分割为 sentence，并计算 embedding
        """

        n = 0
        splits = Assistant.__SPLIT_PATTERN.split(page_text)
        for s in splits:
            p = s.rstrip()
            if len(p) == 0:
                continue
            e = self.__embedding_model.encode(p)
            e /= np.linalg.norm(e)
            self.__db_collection.upsert(ids=f"{page_no}-{n}", embeddings=e.tolist(), metadatas={"page": page_no, "paragraph": n}, documents=p)
            n += 1

    def build_db(self, input_path: str):
        for file_name in os.listdir(input_path):
            m = re.match(r"(\d+)\.txt", file_name)
            if not m:
                continue
            page_no = int(m.group(1))
            file_path = os.path.join(input_path, file_name)
            if not os.path.isfile(file_path):
                continue
            print(f"Process file {file_name}...")
            with open(file_path, 'r', encoding='utf-8') as file:
                self.add_page(page_no, file.read())

    DOMAIN = "generalv2"    # v2.0版本
    SPARK_URL = "ws://spark-api.xf-yun.com/v2.1/chat"  # v2.0环境的地址

    def query(self, question: str, n_results=5):
        e = self.__embedding_model.encode(question)
        results = self.__db_collection.query(query_embeddings=e.tolist(), n_results=n_results, include=["metadatas", "documents", "embeddings", "distances"])
        keys = results.keys()
        values_list = [v[0] for v in results.values()]
        return [dict(zip(keys, values)) for values in zip(*values_list)]

if __name__ == "__main__":

    #pdf2txts("./data/tutorial2.pdf", "./tmp/tutorial2.pdf")
    #count_char("./tmp/tutorial2.pdf")
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:8118"
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:8118"


    #preprocess_texts("./tmp/tutorial1.pdf", "./preprocessed/tutorial1.pdf")

    db_client = chromadb.PersistentClient("./db")
    collection = db_client.get_or_create_collection(
        name="tutorial1_t2v_cn",
        #name="tutorial1_paraphrase",
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction('ryanlai/shibing624_text2vec-base-chinese-fine-tune-insurance_data')
        #embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction('AIDA-UPM/MSTSb_paraphrase-multilingual-MiniLM-L12-v2')
        )
    
    #build_db(collection, "./preprocessed/tutorial1.pdf")

    #query(collection, "pygame是什么")
    #query(collection, "python有几个版本")
