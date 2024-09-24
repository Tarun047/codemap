from dataclasses import dataclass

@dataclass
class DatabaseConfiguration:
    uri: str
    username: str
    password: str
    database: str
    port: int

