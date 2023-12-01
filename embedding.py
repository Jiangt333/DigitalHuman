from pymilvus import connections, db
from pymilvus import utility
from pymilvus import CollectionSchema, FieldSchema, DataType
from pymilvus import Collection
from text2vec import SentenceModel
import math
embeddingModel_path = "../bert_chinese"
embedding_dim = 768

conn = connections.connect(
  host="localhost",
  port='19530'
)

print("test")
print(utility.list_collections())
# create a collection
text_id = FieldSchema(
  name="text_id",
  dtype=DataType.INT64,
  is_primary=True,
)
# text_type = FieldSchema(
#   name="text_type",
#   dtype=DataType.INT64,
#   max_length=200,
#   default_value="-1"
# )
answer = FieldSchema(
  name="answer",
  dtype=DataType.FLOAT_VECTOR,
  dim=embedding_dim
)
schema = CollectionSchema(
#   fields=[text_id, text_type, answer],
  fields=[text_id, answer],
  description="answer search",
  enable_dynamic_field=True
)
collection_name = "fengyang_answer"
collection = Collection(
    name=collection_name,
    schema=schema,
    using='default',
    shards_num=2,
    consistency_level="Strong"
)
print(utility.list_collections())
print("build collection")
# Initialize the BERT model and specify the embedding vector dimension for the output
embeddingModel = SentenceModel(embeddingModel_path)

# read the data file
data_list = []
with open('data.txt', 'r') as file:
    for line in file:
        if(line != '\n'):
            data = line.strip()     # use strip() to remove line breaks at the end of the text
            data_list.append(data)
            # print(line.strip())  

# embedding
embeddings = embeddingModel.encode(data_list)
print(embeddings)
print("embedding the data")

# normalization
for emb in embeddings:
    sum = 0
    for j in emb:
      sum += j**2
    emb_len= math.sqrt(sum)
    for i in range(len(emb)):
      emb[i] = emb[i] / emb_len
    
#输出第一个embedding向量的长度，检查一下
first_vec_len = 0
for x in embeddings[0]:
	first_vec_len = first_vec_len + x**2
first_vec_len = math.sqrt(first_vec_len)
print(f"first_vec_len={first_vec_len}")

# insert embedded data to milvus
embedded_data = [[i for i in range(len(data_list))], embeddings]   # every item is a list  
collection.insert(embedded_data)
print("insert the data")
# build index
index_params = {
  "metric_type":"IP",
  "index_type":"IVF_FLAT",
  "params":{"nlist":128}
}
collection.create_index(
  field_name="answer", 
  index_params=index_params,
  index_name="index1"
)
utility.index_building_progress(collection_name)
print("build index")

connections.disconnect("default")
print("disconnect")
