# 💻GUVI_LLM_PROJECT
This project involves training a GUVI Generative Pre-trained Transformer (GPT) model using data from GUVI. The model is designed to generate coherent and contextually relevant text based on the input provided.

Here  i have trained our model through getting the data's from GUVI webpage, Linkedin, Wikipedia, GitHub, FB, Instagram.

# 🗳️problem statement

The task is to deploy a fine-tuned GPT model, trained specifically on GUVI’s company data, using
Hugging Face services. Students are required to create a scalable and secure web application
using Streamlit or Gradio, making the model accessible to users over the internet. The deployment
should leverage Hugging Face spaces resources and any database to store the username and
login time.


# 🧮Objective
To deploy a pre-trained or Fine tuned GPT model using HUGGING FACE SPACES, making it
accessible through a web application built with Streamlit or Gradio.

# 📑use colab or high accessed RAM computer, because here we used codes are highly big running model

# 🧰Skills Takeaway

* Python
* Deep Learning
* Transformers
* Hugging face models
* LLM
* Streamlit

# 🏁Technologies used
* Python 3.8+
* PyTorch
* Transformers (Hugging Face)
* Streamlit
* Accelarate-u


# 🪕model here we used

The model architecture is based on OpenAI's GPT-2. GPT-2 is a transformer-based model that uses unsupervised learning to generate human-like text. The model has been fine-tuned using the collected dataset to improve its performance on specific tasks.

# ⚓ proper ordered procedure

* Data Collection: Text data is collected from various websites to create a comprehensive dataset.
* Data Preprocessing: The text data is cleaned and preprocessed to remove any irrelevant information and format it suitably for training.
* Tokenization: The text data is tokenized using the GPT-2 tokenizer.
* Fine-tuning: The GPT-2 model is fine-tuned using the preprocessed and tokenized text data


# 📋Finetuning
Fine-tune the model in Google Colab and export it:

Open the Colab notebook and run the training script.
Download the fine-tuned model

#  🎯Finally Upload to Hugging Face in the space,the entire code you having upload in it.

https://huggingface.co/spaces/rajann/llm_gpt_model/tree/main --->> this is our model's deployment link in hugging face
