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
import websocket
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
    URL = "ws://spark-api.xf-yun.com/v2.1/chat"  # v2.0环境的地址
    DOMAIN = "generalv2"    # v2.0版本
    # URL = "ws://spark-api.xf-yun.com/v1.1/chat"  # v1.5环境的地址
    # DOMAIN = "general"   # v1.5版本
    def __init__(self, app_id, api_key, api_secret):
        self.__app_id = app_id
        self.__api_key = api_key
        self.__api_secret = api_secret
        self.__host = urlparse(SparkModel.URL).netloc
        self.__path = urlparse(SparkModel.URL).path

    def __gen_url(self):
        # 生成RFC1123格式的时间戳
        now = datetime.datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + self.__host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.__path + " HTTP/1.1"

        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.__api_secret.encode('utf-8'), signature_origin.encode('utf-8'), digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{self.__api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": self.__host
        }
        # 拼接鉴权参数，生成url
        url = self.URL + '?' + urlencode(v)
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
                    "temperature": 0.01,
                    "max_tokens": 2048,
                    "top_k": 1
                }
            },
            "payload": {
                "message": {
                    "text": [{"role": "user", "content": question}]
                }
            }
        }
    
    def invoke(self, question: str) -> str:
        future = concurrent.futures.Future()
        response = []

        def success():
            if not future.done(): future.set_result("".join(response))
        
        def fail(e):
            if not future.done(): future.set_exception(e)

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
                    success()
                    ws.close()

        ws = websocket.WebSocketApp(
            self.__gen_url(),
            on_message=on_message,
            on_open=lambda ws: ws.send(json.dumps(self.__gen_params(question))),
            on_error = lambda _, e: fail(e),
            on_close = lambda _: fail(Exception("Expect the last msg")))
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

    TEMPLATE = """通过交错使用【思考】、【行动】、【观察】这三个步骤来解决回答问题的任务。
【思考】 根据当前的状况进行推理，给出推理过程；
【行动】 选择如下类型中的一种：
{action_list}

问题：{question}
"""
    ACTION_PATTERN = re.compile("【行动】: (.*?)\\((.*?)\\)")
    def invoke(self, question, max_steps=8):
        al = ""
        for i in range(len(self.__action_list)):
            a = self.__action_list[i]
            al += f"{i+1}. {a[0]}{a[1]}\n"
        al += f"{len(self.__action_list)+1}. finish(答案)，用于返回答案并结束这个任务"
        prompt = self.TEMPLATE.format(action_list=al, question=question)

        for i in range(max_steps):
            prompt += f"\n【思考】{i+1}："
            prompt = """你是一个任务分解器, 你需要将用户的问题拆分成多个彼此之间互不依赖的子任务。
请拆分出多个子任务项，从而能够得到充分的信息以解决问题, 返回格式如下：
```
Plan: 当前子任务要解决的问题描述
#E[i] = 工具名称[工具参数]
Plan: 当前子任务要解决的问题描述
#E[i] = 工具名称[工具参数]
```
其中
1. #E[i] 用于存储Plan id的执行结果, 可被用作占位符。
2. 每个 #E[i] 所执行的内容应与当前Plan解决的问题严格对应。
3. 工具名称必须从以下工具中选择：
  search: 用于从《学习Python》这本书中搜索相关内容
  llm: 基于一段文字描述进行推理
注意: 每个Plan后有且仅能跟随一个#E[i]。
开始！

根据《学习Python》这本书的内容，编写一个猜数游戏？
"""
            prompt = """将下述任务分解为多个互不依赖的子任务。对每个子任务，指定要使用的外部工具及工具输入以获得所需要的证据。返回格式如下：
```
Task1
TaskDescription: 子任务的描述，应对其要解决的问题提供足够丰富的细节
ToolName: 工具名称
ToolInput: 工具输入
Task2
TaskDescription: 子任务的描述，应对其要解决的问题提供足够丰富的细节
ToolName: 工具名称
ToolInput: 工具输入
···
```
其中
1. 每个子任务只能使用一个工具，工具名称必须从以下工具中选择：
  Searcher: 可用于从《学习Python》这本书中搜索相关内容
  Calculator: 可用于解决算术计算问题
  Planner: 可用于解决需要进一步推理的问题
2. 工具名称只能是以下几个之一：Searcher, Calculator, Planner。
3. 在能够解决问题的前提下，分解的子任务应尽量少。
4. 子任务之间不能互相依赖。
开始！

任务：如果你出生于闵桥镇成立的那一年，那么你现在退休了吗？
"""
            prompt = """根据下述给定的问题和已知条件，推理并判断要解决该问题还需要解决哪些未知条件。对于每个未知条件，需要指定用于解决该条件的工具。以如下的JSON格式返回：
```
{
input: {
question: <<要解决的问题>>,
known: [
{
id: <<用于唯一地标识该条件的简短字符串>>,
result: <<该已知条件的值>>,
},
{
id: <<用于唯一地标识该条件的简短字符串>>,
result: <<该已知条件的值>>,
},
...
],
},
output: {
reason: <<根据当前的状况进行推理的思考内容，尽量简洁>>,
unknown: [ 
{
id: <<用于唯一地标识该条件的简短字符串>>,
desc: <<对该条件的描述和相关的推理，尽量简洁>>,
tool_name: <<工具名称>>,
tool_input: <<工具输入>>,
},
{
id: <<用于唯一地标识该条件的简短字符串>>,
desc: <<对该条件的描述和相关的推理，尽量简洁>>,
tool_name: <<工具名称>>,
tool_input: <<工具输入>>,
},
...
],
},
}
```
其中，
1. 字段名后由 '<<' 和 '>>' 括起来的部分是对字段值的说明，输出时需要替换为实际值
2. known 字段的取值是所有已知条件的列表
3. unknown 字段的取值是所有未知条件的列表。这些未知条件是解决问题所必需的
4. 对于每个未知条件，只能使用一个工具，tool_name 必须从以下工具中选择：
  "Searcher": 可用于从知识库中搜索信息，tool_input 必须是一个实体名
  "Calculator": 可用于解决算术计算问题，tool_input 必须是一个合法的算术表达式，只能由数值和运算符组成
5. tool_name 只能是以下几个之一："Searcher", "Calculator"。

其中，
1. 字段名后由 '<<' 和 '>>' 括起来的部分是对字段值的说明，会替换为实际值

开始！

```
{
input: {
question: "如果你出生于金湖县成立的那一年，那么你现在退休了吗？",
known: [
{
id: "金湖县成立的年份"
result: "1960",
},
{
id: "当前年份"
result: "2023",
},
{
id: "退休年龄"
result: "60",
},
],
},
}
```
"""
            prompt = """通过将给定问题分解为子任务，并逐步将子问题解决，从而最终解决原问题。根据给定的问题和已解决的子问题，推理要解决该问题还有哪些未解决的子问题。对于每个未解决的子问题，需要指定用于解决该问题的工具。以如下的JSON格式返回：
```
{
input: {
question: <<要解决的问题>>,
solved: [
{
id: <<用于唯一地标识该子问题的简短字符串>>,
desc: <<对该子问题的简洁描述>>,
result: <<该问题的答案>>,
},
...
],
},
output: {
reason: <<根据当前的状况进行推理的思考内容，尽量简洁>>,
final_result: <<该子段为可选项，仅当原问题n已解决时，在该字段中给出原问题的答案>>,
unsolved: [ 
{
id: <<用于唯一地标识该子问题的简短字符串>>,
desc: <<对该子问题的简洁描述>>,
tool_name: <<工具名称>>,
tool_input: <<工具输入>>,
},
...
],
},
}
```
其中，
1. 字段名后由 '<<' 和 '>>' 括起来的部分是对字段值的说明，输出时需要替换为实际值
2. 子问题的 id 可以是 "q1", "q2", ...，以此类推，且任何两个子问题的 id 不可重复
3. solved 字段是所有已解决子问题的列表
4. unsolved 字段是所有未解决子问题的列表
5. 对于每个未解决子问题，只能使用一个工具来解决它，工具名称必须从以下工具中选择：
  "Searcher": 可用于从知识库中搜索信息。工具输入必须是一个实体名
  "Calculator": 可用于解决算术计算问题。工具输入必须是一个合法的算术表达式，只能由数值、运算符组成，如果依赖于其他子问题的答案，可以使用对应的id占位

开始！

```
{
input: {
question: "如果你出生于金湖县成立的那一年，那么你现在退休了吗？",
solved: [
{
id: "q1",
desc: "查询金湖县的成立年份",
result: "1960",
},
{
id: "q2",
desc: "查询当前年份",
result: "2023",
},
],
},
output: {

}
}
```
"""
            print(f"\nInput:\n------\n{prompt}")
            r = self.__model.invoke(prompt)
            print(f"\nOuput:\n------\n{r}")
            m = self.ACTION_PATTERN.search(r)
            if not m:
                raise Exception(f"No action in answer: {r}")
            if m[1] == "finish":
                return f"最终答案是：{m[2]}"
            if m[1] not in self.__action_map:
                raise Exception(f"Unknown action in answer: {r}")

            o = self.__action_map[m[1]](m[2])
            prompt += r
            prompt += f"\n【观察】：{o}\n"
        return "我尽力了..."
'''
 如果你出生于闵桥镇成立的那一年，那么你现在退休了吗？
 1. 多少岁退休
 2. 如果你出生于闵桥镇成立的那一年，你现在多少岁？
    1. 你跟闵桥镇同岁
    2。 闵桥镇多少岁
'''
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
