import abc
import typing
import tiktoken

class TokenizerBase(abc.ABC):
    """Abstract base class for all tokenizers."""

    @abc.abstractmethod
    def __init__(self, name: str):
        self.name = name

    @property
    @abc.abstractmethod
    def vocabulary_size(self) -> int:
        pass

    @abc.abstractmethod
    def encode(self, text: str) -> typing.List[int]:
        pass

    @abc.abstractmethod
    def decode(self, tokens: typing.List[int]) -> str:
        pass
    
    @property
    @abc.abstractmethod
    def is_custom(self) -> bool:
        """Returns True if this is a user-defined custom tokenizer."""
        pass

class TiktokenWrapper(TokenizerBase):
    """Wrapper for OpenAI's tiktoken tokenizers."""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        try:
            self._encoding = tiktoken.get_encoding(model_name)
        except ValueError:
            # Fallback for model names like 'gpt-4' which map to encodings
            try:
                self._encoding = tiktoken.encoding_for_model(model_name)
            except KeyError:
                raise ValueError(f"Could not load tiktoken model: {model_name}")

    @property
    def vocabulary_size(self) -> int:
        return self._encoding.n_vocab

    def encode(self, text: str) -> typing.List[int]:
        return self._encoding.encode(text)

    def decode(self, tokens: typing.List[int]) -> str:
        return self._encoding.decode(tokens)

    @property
    def is_custom(self) -> bool:
        return False

class SimpleSpaceTokenizer(TokenizerBase):
    """A sample custom tokenizer that splits by spaces."""
    
    def __init__(self):
        super().__init__("Simple Space")
        self.vocab = {"<UNK>": 0}
        self.inverse_vocab = {0: "<UNK>"}
        # Pre-populate with some basics mimicking a dynamic build for demo
        # Realistically this would load a vocab file or dynamic build
        self._next_id = 1

    @property
    def vocabulary_size(self) -> int:
        return 1000000 # Conceptual infinite size for this simple dynamic one, or returns current len

    def _get_id(self, word: str) -> int:
        if word not in self.vocab:
            self.vocab[word] = self._next_id
            self.inverse_vocab[self._next_id] = word
            self._next_id += 1
        return self.vocab[word]

    def encode(self, text: str) -> typing.List[int]:
        words = text.split(" ")
        return [self._get_id(w) for w in words]

    def decode(self, tokens: typing.List[int]) -> str:
        return " ".join([self.inverse_vocab.get(t, "<UNK>") for t in tokens])

    @property
    def is_custom(self) -> bool:
        return True

class SpaceTokenizer(TokenizerBase):
    """A sample custom tokenizer that splits by spaces."""
    
    def __init__(self):
        super().__init__("SpaceWithNL")
        self.vocab = {"<UNK>": 0}
        self.inverse_vocab = {0: "<UNK>"}
        # Pre-populate with some basics mimicking a dynamic build for demo
        # Realistically this would load a vocab file or dynamic build
        self._next_id = 1

    @property
    def vocabulary_size(self) -> int:
        return 1000000 # Conceptual infinite size for this simple dynamic one, or returns current len

    def _get_id(self, word: str) -> int:
        if word not in self.vocab:
            self.vocab[word] = self._next_id
            self.inverse_vocab[self._next_id] = word
            self._next_id += 1
        return self.vocab[word]

    def encode(self, text: str) -> typing.List[int]:
        text = text.replace("\n", " \n ")
        # text = "".join(f" {c}" if not c.isalnum() else c for c in text)
        
        text = "".join(f" {c}" if not (c.isalnum() or c.isspace() or c in "().,!?[]{}<>=+-*/%^&|~`:;\"'") else c for c in text)
        for s in ["ing"] + list("().,!?[]{}<>=+-*/%^&|~`:;\"'\t\r\f\v0123456789"):
            text = text.replace(s, f" {s}")
        words = text.split(" ")
        return [self._get_id(w) for w in words]

    def decode(self, tokens: typing.List[int]) -> str:
        return " ".join([self.inverse_vocab.get(t, "<UNK>") for t in tokens])

    @property
    def is_custom(self) -> bool:
        return True
