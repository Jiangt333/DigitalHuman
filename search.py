from pymilvus import connections, Collection, utility
from text2vec import SentenceModel
import time, math, os
import openai

openai.api_key = os.environ.get('OPENAI_API_KEY')
print(openai.api_key)
embeddingModel_path = "../bert_chinese"
embeddingModel = SentenceModel(embeddingModel_path)
collection_name = "fengyang_answer"

search_text = "连州腊肉是怎么做的？"
k = 5

conn = connections.connect(
  host="localhost",
  port='19530'
)

# read the data file
data_list = []
with open('data.txt', 'r') as file:
    for line in file:
        if(line != '\n'):
            data = line.strip()     # use strip() to remove line breaks at the end of the text
            data_list.append(data)
            # print(line.strip())  

start_time = time.time()

# embedding the target text
emb_target = embeddingModel.encode(search_text)
print("embedding encode")

# normalization
sum = 0
for emb in emb_target:
    sum += emb**2
sum = math.sqrt(sum)
for i in range(len(emb_target)):
    emb_target[i] = emb_target[i] / sum
first_vec_len = 0
for x in emb_target:
	first_vec_len = first_vec_len + x**2
first_vec_len = math.sqrt(first_vec_len)
print(f"first_vec_len={first_vec_len}")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"对query进行embedding耗时: {elapsed_time}秒")

# load collection to memory
collection = Collection(collection_name)
collection.load()

# conduct similarity search
search_params = {
	"metric_type": "IP",
	"offset": 0,
	"ignore_growing": False, 
    "params": {"nprobe": 10}
}

results = collection.search(
    data=[emb_target], 
    anns_field="answer", 
    param=search_params,
    limit=k,
    expr=None,
    output_fields=['output'],
    consistency_level="Strong"
)

# print result
hits = results[0]
for idx, similarity in zip(hits.ids, hits.distances):
    # print(hits.ids[0], idx)
    print(data_list[idx])
    print("Cosine Similarity:", similarity)
    print()


print("call openai api")
system_prompt = f"You are a live streaming e-commerce host, and the audience will ask you some questions. I have searched the vector database based on the questions and obtained the most similar answer, but this answer is relatively simple. Now you need to output more complete and specific anchor answers based on the audience's questions {search_text} and the simpler answers I have found which in next message"
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user","content": data_list[hits.ids[0]]},
    ],
    temperature = 0
)
print(response['choices'][0]['message']['content'])

# result = ''
# for choice in response.choices:
#     result += '///' + choice.message.content
# print(result)














# def generate_response(prompt):
#     response = openai.Completion.create(
#         engine="gpt-3.5-turbo",
#         prompt=prompt,
#     )
#     result = ''
#     # 循环遍历GPT-3 API返回的response中的所有回答选项。
#     for choice in response.choices:
#         # 每个回答选项的文本内容加入到“result”字符串变量中。
#         result += '///' + choice.message.content
#     print(f"result: {result}")
#     print(f"response: {response.choices[0].text.strip()}")
#     # return response.choices[0].text.strip()
#     return result

# def process_messages(messages):
#     context = ""
#     for message in messages:
#         if message["role"] == "user":
#             # 用户消息
#             question = message["content"]
#             response = generate_response(f"Q: {question}\nA:")
#             context += f"Q: {question}\nA: {response}\n"
#         elif message["role"] == "system":
#             # 系统消息
#             context += f"System: {message['content']}\n"

#     return context

# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": data_list[hits.ids[0]]},
# ]

# response = process_messages(messages)
# print(response)

# # release collection, disconnect
# collection.release()
# connections.disconnect("default")