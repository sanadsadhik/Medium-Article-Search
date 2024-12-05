import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY
from sentence_transformers import SentenceTransformer
import torch

df = pd.read_csv("data\medium_post_titles.csv", nrows=5000)

df = df.dropna()

df = df[~df["subtitle_truncated_flag"]]

df['title_extended'] = df['title'] + df['subtitle']

api_key = PINECONE_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)

pc.create_index(
    name="medium-data",
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(    
        cloud="aws",
        region="us-east-1"
    )
)

model = SentenceTransformer('all-MiniLM-L6-v2',device='cpu')
# print(model)

df['values'] = df['title_extended'].map(
    lambda x: (model.encode(x)).tolist())

df['id'] = df.reset_index(drop = 'index').index

df['metadata'] = df.apply(lambda x:{
    'title': x['title'],
    'subtitle': x['subtitle'],
    'category': x['category'],
}, axis=1)

df_upsert = df[['id','values','metadata']]
df_upsert['id'] = df_upsert['id'].map(lambda x: str(x))
# print(df_upsert)

idx = pc.Index('medium-data')

idx.upsert_from_dataframe(df_upsert)

# querying
xc = idx.query(vector=model.encode("where is my cat?").tolist(), top_k=3, include_metadata=True, include_values=True)

for result in xc['matches']:
    print(f"{round(result['score'],2)}: {result['metadata']['title']}")

