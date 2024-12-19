import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

class MySQLAgent:
    
    def __init__(self) -> None:
        load_dotenv()
        self.db_connector()

    def db_connector(self):
        user = os.getenv('DB_USER')
        pw = os.getenv('DB_PASSWORD')
        host = os.getenv('DB_HOST')
        port = os.getenv('DB_PORT')
        database = os.getenv('DB_NAME')

        if not all([user, pw, host, port, database]):
            raise ValueError("One or more required environment variables are missing.")

        self.connection_string = f"mysql+pymysql://{user}:{pw}@{host}:{port}/{database}?charset=utf8mb4"

        try:
            self.engine = create_engine(self.connection_string)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to the database: {e}")

    def read_table(self, query) -> pd.DataFrame:

        try:
            df = pd.read_sql(query, con=self.engine)
            df.columns = df.columns.str.lower()
            return df
        except Exception as e:
            raise ValueError(f"Failed to read table with query '{query}': {e}")

    def write_table(self, data, table_name, if_exists, index, data_type):

        try:
            data.to_sql(name=table_name, con=self.engine,
                        if_exists=if_exists, index=index, dtype=data_type)
            

        except Exception as e:
            raise ValueError(f"Failed to write data to table '{table_name}': {e}")
