import json
import base64
import gzip
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def decompress_gzip_base64(data: str) -> dict:
    try:
        b1 = gzip.decompress(base64.b64decode(data))

    # Second layer
        b2 = gzip.decompress(base64.b64decode(b1))
        # compressed_bytes = base64.b64decode(data)
        # decompressed_bytes = gzip.decompress(compressed_bytes)
        return json.loads(b2.decode('utf-8'))
    except Exception as e:
        raise ValueError(f"Decompression failed: {e}")
 