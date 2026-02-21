# Tokenizer Tester Manual

This manual explains how to add new tokenizers to the application and run the tester.

## Quick Start

1.  **Run the Application**:
    ```bash
    python tokenizer_tester.py
    ```
    The web interface will be available at `http://127.0.0.1:8000`.

2.  **Use the Interface**:
    - Select a tokenizer from the dropdown.
    - Enter text to see the tokenization results.

## Adding a Custom Tokenizer

To add a new tokenizer, simply add a new class to `tokenizers.py`. The application accepts any class that:
1.  Inherits from `TokenizerBase`.
2.  Implements the required methods (`vocabulary_size`, `encode`, `decode`, `is_custom`).
3.  Can be instantiated with **no arguments** (for automatic discovery).

### Example

Add this code to `e:\GPT\Tokenizer\tokenizers.py`:

```python
class MyCustomTokenizer(TokenizerBase):
    def __init__(self):
        super().__init__("My Custom Tokenizer")
        
    @property
    def vocabulary_size(self) -> int:
        return 100
        
    def encode(self, text: str) -> typing.List[int]:
        # Simple example: convert characters to ASCII values
        return [ord(c) for c in text]
        
    def decode(self, tokens: typing.List[int]) -> str:
        return "".join([chr(t) for t in tokens])
        
    @property
    def is_custom(self) -> bool:
        return True
```

### Restarting
After adding the code, **restart the application** (`Ctrl+C` then run `python tokenizer_tester.py` again). The new tokenizer "My Custom Tokenizer" will automatically appear in the list.

## Notes
- `TiktokenWrapper` is handled specially to provide standard OpenAI models.
- If your class requires arguments in `__init__`, it will be skipped by the auto-discovery mechanism unless you manually modify `tokenizer_tester.py` to instantiate it.
