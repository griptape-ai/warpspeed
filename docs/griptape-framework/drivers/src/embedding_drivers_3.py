from griptape.drivers.embedding.amazon_bedrock import AmazonBedrockTitanEmbeddingDriver

embeddings = AmazonBedrockTitanEmbeddingDriver().embed_string("Hello world!")

# display the first 3 embeddings
print(embeddings[:3])
