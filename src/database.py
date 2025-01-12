import sqlite3
import json

def create_metadata_db(json_path, db_path):
    '''
    Create SQLite database with metadata from JSONL file.
    Parameters:
        json_path (str): path to JSONL file with metadata
        db_path (str): path to SQLite database
    Returns:
        None
    '''
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS metadata (
                    asin TEXT PRIMARY KEY,
                    main_category TEXT,
                    subcategories TEXT,
                    title TEXT,
                    subtitle TEXT,
                    author_name TEXT,
                    average_rating REAL,
                    rating_number INTEGER,
                    data TEXT
                )''')

    with open(json_path, 'r', encoding='utf-8') as f:
        i=0
        for line in f:
            i+=1
            try:
                item = json.loads(line.strip())
                asin = item.get('parent_asin')
                main_category = item.get('main_category')
                title = item.get('title')
                subtitle = item.get('subtitle')
                author_name = item.get('author', {}).get('name')
                subcategories = ', '.join(item.get('categories', []))
                average_rating = item.get('average_rating')
                rating_number = item.get('rating_number')
                data = json.dumps(item)
                c.execute('''INSERT OR REPLACE INTO metadata (asin, main_category, title, subtitle, author_name, subcategories, average_rating, rating_number, data)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                          (asin, main_category, title, subtitle, author_name, subcategories, average_rating, rating_number, data))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
            if i%10000==0:
                print(f"Processed {i} items")

    conn.commit()
    conn.close()

def get_metadata(product_id, db_path):
    '''
    Get metadata for a product from the database.
    Parameters:
        product_id (str): ASIN of product
        db_path (str): path to SQLite database
    Returns:
        metadata (json): metadata for quered product
    '''
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('SELECT data FROM metadata WHERE asin = ?', (product_id,))
        row = c.fetchone()
        conn.close()
        if row:
            return json.loads(row[0])
        else:
            return None
    except sqlite3.Error as e:
        print(f"Error while fetching {product_id} in {db_path}: {e}")
        return None

if __name__ == '__main__':
    create_metadata_db('../data/meta_Books.jsonl', '../data/metadata.db')