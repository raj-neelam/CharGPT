import abc
import time
import typing
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import tiktoken

# --- Backend Logic: Tokenizer Wrappers ---







# --- Registry ---

# --- Registry ---
import inspect
import tokenizers

TOKENIZERS: typing.Dict[str, tokenizers.TokenizerBase] = {}

def register_tokenizer(tokenizer: tokenizers.TokenizerBase):
    TOKENIZERS[tokenizer.name] = tokenizer

# 1. Initialize Default Tiktoken Models (Standard)
# We still want these specific known models available by default
for name in ["cl100k_base", "p50k_base", "r50k_base", "gpt-4", "gpt-3.5-turbo"]:
    try:
        register_tokenizer(tokenizers.TiktokenWrapper(name))
    except ValueError:
        pass # Skip if not available

# 2. Automatically discover and register other Tokenizer classes from the tokenizers file
# This allows the user to just add a class to tokenizers.py and have it appear here.
print("Scanning tokenizers.py for custom classes...")
for name, obj in inspect.getmembers(tokenizers):
    if inspect.isclass(obj) and issubclass(obj, tokenizers.TokenizerBase) and obj is not tokenizers.TokenizerBase:
        # We skip TiktokenWrapper here because we manually instantiated specific instances above.
        # Use a check to see if it's strictly TiktokenWrapper class to avoid re-instantiating it without args (which would fail)
        if obj == tokenizers.TiktokenWrapper:
            continue
            
        # For other classes (custom ones), we try to instantiate them with no arguments
        try:
            instance = obj()
            register_tokenizer(instance)
            print(f"Registered custom tokenizer: {instance.name}")
        except TypeError as e:
            # This usually means it requires arguments we don't know about
            print(f"Skipping {name}: Could not instantiate (might require arguments). Error: {e}")
        except Exception as e:
            print(f"Skipping {name}: Error during instantiation. Error: {e}")


# --- FastAPI App ---

app = FastAPI(title="Tokenizer Tester")

class TokenizeRequest(BaseModel):
    text: str
    tokenizer_name: str

class TokenizeResponse(BaseModel):
    tokens: typing.List[int]
    decoded_parts: typing.List[str] # Helper to show exactly what text corresponds to what token
    vocab_size: int
    encode_time_ms: float
    decode_time_ms: float
    total_tokens: int

class TokenizerInfo(BaseModel):
    name: str
    vocab_size: int
    is_custom: bool

@app.get("/api/tokenizers", response_model=typing.List[TokenizerInfo])
def get_tokenizers():
    return [
        TokenizerInfo(name=t.name, vocab_size=t.vocabulary_size if t.name != "Simple Space (Custom)" else 0, is_custom=t.is_custom) 
        for t in TOKENIZERS.values()
    ]

@app.post("/api/tokenize", response_model=TokenizeResponse)
def tokenize(request: TokenizeRequest):
    if request.tokenizer_name not in TOKENIZERS:
        raise HTTPException(status_code=404, detail="Tokenizer not found")
    
    tokenizer = TOKENIZERS[request.tokenizer_name]
    
    # Timing Encode
    start_enc = time.perf_counter()
    tokens = tokenizer.encode(request.text)
    end_enc = time.perf_counter()
    
    # Timing Decode (per token to visualize boundaries correctly)
    # Note: Decoding individual tokens is sometimes lossy in BPE if not careful, 
    # but for visualization we want to see what each token represents.
    start_dec = time.perf_counter()
    decoded_parts = []
    
    # Optimization: Decode is usually fast, but decoding 1 by 1 can be slow for large texts.
    # However, for the UI we *need* the text mapping.
    # Tiktoken's decode_single_token_bytes equivalent is preferred if available to be exact.
    
    if isinstance(tokenizer, tokenizers.TiktokenWrapper):
        # Use more precise decoding if possible to handle byte boundaries, 
        # but standard decode([t]) is safer for general display
        for t in tokens:
            try:
                decoded_parts.append(tokenizer.decode([t]))
            except:
                 decoded_parts.append("<ERR>")
    else:
        # Fallback loop
        for t in tokens:
             decoded_parts.append(tokenizer.decode([t]))
             
    end_dec = time.perf_counter()

    return TokenizeResponse(
        tokens=tokens,
        decoded_parts=decoded_parts,
        vocab_size=tokenizer.vocabulary_size if not isinstance(tokenizer, tokenizers.SimpleSpaceTokenizer) else len(tokenizer.vocab),
        encode_time_ms=(end_enc - start_enc) * 1000,
        decode_time_ms=(end_dec - start_dec) * 1000,
        total_tokens=len(tokens)
    )

from fastapi.responses import FileResponse



@app.get("/", response_class=FileResponse)
def serve_ui():
    return "index.html"

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
