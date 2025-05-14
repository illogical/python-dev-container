from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://host.docker.internal:19530",
    token="root:Milvus"
)

client.create_database(
    db_name="my_database_1",
    properties={
        "database.replica.number": 3
    }
)