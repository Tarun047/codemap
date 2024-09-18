from dataclasses import dataclass

@dataclass
class GraphDatabaseConfiguration:
    uri: str
    username: str
    password: str
    database: str

