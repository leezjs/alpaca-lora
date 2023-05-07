from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from peft import PeftModel

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

model = LlamaForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    load_in_8bit=True,
    device_map='auto',
)

# load PEFT checkpoint
base_model = PeftModel.from_pretrained(model, "./lora-alpaca")

pipe = pipeline(
    "text-generation",
    model=base_model, 
    tokenizer=tokenizer, 
    max_length=1024,
    temperature=0.6,
    top_p=0.95,
    repetition_penalty=1.2
)

local_llm = HuggingFacePipeline(pipeline=pipe)

from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

# We are going to set the memory to go back 4 turns
window_memory = ConversationBufferWindowMemory(k=4)

conversation = ConversationChain(
    llm=local_llm, 
    verbose=True, 
    memory=window_memory
)

# conversation.prompt.template = '''The following is a friendly conversation between a human and an AI called Alpaca. 
# The AI is talkative and provides lots of specific details from its context. 
# If the AI does not know the answer to a question, it truthfully says it does not know. 

# Current conversation:
# {history}
# Human: {input}
# AI:'''

conversation.prompt.template = '''The following is a friendly conversation between a human and an AI called Alpaca. 
If the AI does not know the answer to a question, it truthfully says it does not know. 
You should not fabricate any question the human does not ask.

Current conversation:
{history}
Human: {input}
AI:'''

print(conversation.predict(input="What is your name?"))
print(conversation.predict(input="Can you tell me what an Alpaca is?"))
print(conversation.predict(input="How is it different from a Llama?"))
print(conversation.predict(input="Can you give me some good names for a pet llama?"))
print(conversation.predict(input="Is your name Fred?"))
print(conversation.predict(input="What food should I feed my new llama?"))
