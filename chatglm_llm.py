from langchain.llms.base import LLM
from typing import Optional, List
from langchain.llms.utils import enforce_stop_tokens
from transformers import AutoTokenizer, AutoModel

"""ChatGLM_G is a wrapper around the ChatGLM model to fit LangChain framework. May not be an optimal implementation"""


class ChatGLM(LLM):
    max_token: int = 10000
    temperature: float = 0.1
    top_p = 0.9
    history = []
    tokenizer: object = None
    model: object = None

    def _init_(self):
        super().__init__()

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "THUDM/chatglm-6b",
            trust_remote_code=True
        )
        self.model = (
            AutoModel.from_pretrained(
            "THUDM/chatglm-6b",
            trust_remote_code=True)
        .half()
        .cuda()
        )

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    def _call(self,
            prompt: str,
            stop: Optional[List[str]] = None) -> str:
        response, updated_history = self.model.chat(
            self.tokenizer,
            prompt,
            history=self.history,
            max_length=self.max_token,
            temperature=self.temperature,

        )
        #print("history: ", self.history)
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        self.history = updated_history
        return response
