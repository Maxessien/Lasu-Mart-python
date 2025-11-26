from FlagEmbedding import FlagAutoModel

model = FlagAutoModel.from_finetuned('BAAI/bge-small-en-v1.5')



def encodeText(text):
    embedding = model.encode([text])
    return embedding[0]