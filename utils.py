import os
import psycopg
from psycopg.rows import dict_row
from psycopg.connection import Cursor

def connection_wrapper(callback: function) -> dict:
    conn = psycopg.connect(os.environ.get("POSTGRESQL_CONNECTION_URL"), row_factory=dict_row)
    cbValue = callback(conn.cursor())
    conn.close()
    return cbValue or None


def fetch_products():
    try:
        query = "SELECT product_name, description, category FROM products"
        def fetch_fn(cur: Cursor):
            cur.execute(query)
            return cur.fetchall()
        products = connection_wrapper(fetch_fn)
        return products
    except Exception as e:
        print("An error has occured", e)
        raise Exception(e)