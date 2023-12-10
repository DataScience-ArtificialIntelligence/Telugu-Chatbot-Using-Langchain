from flask import Flask, render_template, request

from transformers import MarianMTModel, MarianTokenizer
from transformers import pipeline
from translate import Translator
from torch import cuda
import transformers

from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains.question_answering import load_qa_chain

from langchain.llms import HuggingFacePipeline

app = Flask(__name__)


model_name = 'facebook/blenderbot-400M-distill'

# Initialize the translator and the chatbot chain
translator_te_en = Translator(to_lang="en", from_lang="te")
translator_en_te = Translator(to_lang="te", from_lang="en")

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()
generate_text = transformers.pipeline(
    task="text2text-generation",
    tokenizer=tokenizer,
    model=model
)

template = """You are a generic friendly chatbot that answers well and respectfully to humans,
{question}
"""
prompt = PromptTemplate(template=template, input_variables=['question'])
llm = HuggingFacePipeline(pipeline=generate_text)
llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)


def translate_telugu_to_english(text):
    return translator_te_en.translate(text)


def translate_english_to_telugu(text):
    return translator_en_te.translate(text)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form.get("user_input")
    if user_input == "వెళ్ళొస్తాను" or user_input == "ఇక సెలవు" :
        response = "సంభాషణ ముగిసింది, ధన్యవాదాలు"
    else:
        english_translation = translate_telugu_to_english(user_input)
        response = llm_chain.run(english_translation)
        response = translate_english_to_telugu(response)
    return render_template("index.html", user_input=user_input, response=response)


if __name__ == "__main__":
    app.run(debug=True)
