import torch
import numpy as np
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = torch.load("C:\\Users\\Vincent\\Documents\\Formation\\NLP\\NLP.model")

tokenizer = torch.load("C:\\Users\\Vincent\\Documents\\Formation\\NLP\\tokenizer.model")

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch
import numpy as np
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Uncommunt si load model effectue

# model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
# tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

article_hi = "संयुक्त राष्ट्र के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है"

# translate Hindi to French
tokenizer.src_lang = "hi_IN"

def predict(review):
    encoded_hi = tokenizer(review, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded_hi,
        forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"]
        )
    traduction = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return traduction

iface = gr.Interface(fn=predict, inputs="text", outputs="text")
iface.launch()

