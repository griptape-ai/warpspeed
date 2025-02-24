---
search:
  boost: 2
---

## Overview

Griptape provides a way to build drivers for vector DBs where embeddings can be stored and queried. Every Vector Store Driver implements the following methods:

- `upsert_text_artifact()` for updating or inserting a new [TextArtifact](../../reference/griptape/artifacts/text_artifact.md) into vector DBs. The method will automatically generate embeddings for a given value.
- `upsert_text_artifacts()` for updating or inserting multiple [TextArtifact](../../reference/griptape/artifacts/text_artifact.md)s into vector DBs. The method will automatically generate embeddings for given values.
- `upsert_text()` for updating and inserting new arbitrary strings into vector DBs. The method will automatically generate embeddings for a given value.
- `upsert_vector()` for updating and inserting new vectors directly.
- `query()` for querying vector DBs.

Each Vector Store Driver takes a [BaseEmbeddingDriver](../../reference/griptape/drivers/embedding/base_embedding_driver.md) used to dynamically generate embeddings for strings.

!!! info

    When working with vector database indexes with Griptape Drivers, make sure the number of dimensions is equal to 1536. Nearly all embedding models create vectors with this number of dimensions. Check the documentation for your vector database on how to create/update vector indexes.

!!! info

    More Vector Store Drivers are coming soon.

## Vector Store Drivers

### Local

The [LocalVectorStoreDriver](../../reference/griptape/drivers/vector/local_vector_store_driver.md) can be used to load and query data from memory. Here is a complete example of how the Driver can be used to load a webpage into the Driver and query it later:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/vector_store_drivers_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/vector_store_drivers_1.txt"
    ```


### Griptape Cloud Knowledge Base

The [GriptapeCloudVectorStoreDriver](../../reference/griptape/drivers/vector/griptape_cloud_vector_store_driver.md) can be used to query data from a Griptape Cloud Knowledge Base. Loading into Knowledge Bases is not supported at this time, only querying. Here is a complete example of how the Driver can be used to query an existing Knowledge Base:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/vector_store_drivers_2.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/vector_store_drivers_2.txt"
    ```


### Pinecone

!!! info

    This Driver requires the `drivers-vector-pinecone` [extra](../index.md#extras).

The [PineconeVectorStoreDriver](../../reference/griptape/drivers/vector/pinecone_vector_store_driver.md) supports the [Pinecone vector database](https://www.pinecone.io/).

Here is an example of how the Driver can be used to load and query information in a Pinecone cluster:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/vector_store_drivers_3.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/vector_store_drivers_3.txt"
    ```


### Marqo

!!! info

    This Driver requires the `drivers-vector-marqo` [extra](../index.md#extras).

The [MarqoVectorStoreDriver](../../reference/griptape/drivers/vector/marqo_vector_store_driver.md) supports the Marqo vector database.

Here is an example of how the Driver can be used to load and query information in a Marqo cluster:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/vector_store_drivers_4.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/vector_store_drivers_4.txt"
    ```


### Mongodb Atlas

!!! info

    This Driver requires the `drivers-vector-mongodb` [extra](../index.md#extras).

The [MongodbAtlasVectorStoreDriver](../../reference/griptape/drivers/vector/mongodb_atlas_vector_store_driver.md) provides support for storing vector data in a MongoDB Atlas database.

Here is an example of how the Driver can be used to load and query information in a MongoDb Atlas Cluster:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/vector_store_drivers_5.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/vector_store_drivers_5.txt"
    ```


The format for creating a vector index should look similar to the following:

```json
{
  "fields": [
    {
      "numDimensions": 1536,
      "path": "<path_to_vector>",
      "similarity": "euclidean",
      "type": "vector"
    },
    {
      "path": "namespace",
      "type": "filter"
    }
  ]
}
```

Replace `path_to_vector` with the expected field name where the vector content will be.

### Azure MongoDB

!!! info

    This Driver requires the `drivers-vector-mongodb` [extra](../index.md#extras).

The [AzureMongoDbVectorStoreDriver](../../reference/griptape/drivers/vector/azure_mongodb_vector_store_driver.md) provides support for storing vector data in an Azure CosmosDb database account using the MongoDb vCore API

Here is an example of how the Driver can be used to load and query information in an Azure CosmosDb MongoDb vCore database. It is very similar to the Driver for [MongoDb Atlas](#mongodb-atlas):

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/vector_store_drivers_6.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/vector_store_drivers_6.txt"
    ```


### Redis

!!! info

    This Driver requires the `drivers-vector-redis` [extra](../index.md#extras).

The [RedisVectorStoreDriver](../../reference/griptape/drivers/vector/redis_vector_store_driver.md) integrates with the Redis vector storage system.

Here is an example of how the Driver can be used to load and query information in a Redis Cluster:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/vector_store_drivers_7.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/vector_store_drivers_7.txt"
    ```


The format for creating a vector index should be similar to the following:

```
FT.CREATE idx:griptape ON hash PREFIX 1 "griptape:" SCHEMA namespace TAG vector VECTOR FLAT 6 TYPE FLOAT32 DIM 1536 DISTANCE_METRIC COSINE
```

### OpenSearch

!!! info

    This Driver requires the `drivers-vector-opensearch` [extra](../index.md#extras).

The [OpenSearchVectorStoreDriver](../../reference/griptape/drivers/vector/opensearch_vector_store_driver.md) integrates with the OpenSearch platform, allowing for storage, retrieval, and querying of vector data.

Here is an example of how the Driver can be used to load and query information in an OpenSearch Cluster:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/vector_store_drivers_8.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/vector_store_drivers_8.txt"
    ```


The body mappings for creating a vector index should look similar to the following:

```json
{
    "mappings": {
        "properties": {
            "vector": {"type": "knn_vector", "dimension": 1536},
            "namespace": {"type": "keyword"},
            "metadata": {"type": "object", "enabled": true}
        }
    }
}
```

### PGVector

!!! info

    This Driver requires the `drivers-vector-pgvector` [extra](../index.md#extras).

The [PGVectorVectorStoreDriver](../../reference/griptape/drivers/vector/pgvector_vector_store_driver.md) integrates with PGVector, a vector storage and search extension for Postgres. While Griptape will handle enabling the extension, PGVector must be installed and ready for use in your Postgres instance before using this Vector Store Driver.

Here is an example of how the Driver can be used to load and query information in a Postgres database:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/vector_store_drivers_9.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/vector_store_drivers_9.txt"
    ```


### Qdrant

!!! info

    This Driver requires the `drivers-vector-qdrant` [extra](../index.md#extras).

The QdrantVectorStoreDriver supports the [Qdrant vector database](https://qdrant.tech/).

Here is an example of how the Driver can be used to query information in a Qdrant collection:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/vector_store_drivers_10.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/vector_store_drivers_10.txt"
    ```


### Astra DB

!!! info

    This Driver requires the `drivers-vector-astra-db` [extra](../index.md#extras).

The AstraDbVectorStoreDriver supports [DataStax Astra DB](https://www.datastax.com/products/datastax-astra).

The following example shows how to store vector entries and query the information using the driver:

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/vector_store_drivers_11.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/vector_store_drivers_11.txt"
    ```

