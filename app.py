from flask import Flask, request, jsonify, render_template

import os

import requests

from dotenv import load_dotenv

from langchain.chains import TransformChain, LLMChain, SequentialChain
from langchain.chat_models import AzureChatOpenAI

from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

import openai

app = Flask(__name__)

load_dotenv('.env')

# Endpoint Settings
bing_search_url = "https://api.bing.microsoft.com/v7.0/search"
bing_subscription_key = os.environ['BING_SUBSCRIPTION_KEY']
openai_api_type = "azure"
openai_api_base = os.environ['OPENAI_API_BASE']
openai_api_key = os.environ['OPENAI_API_KEY']
gpt4_deployment_name = "gpt35"

# We are assuming that you have all model deployments on the same Azure OpenAI service resource above.  If not, you can change these settings below to point to different resources.
gpt4_endpoint = openai_api_base
gpt4_api_key = openai_api_key
dalle_endpoint = openai_api_base
dalle_api_key = openai_api_key
plugin_model_url = openai_api_base
plugin_model_api_key = openai_api_key

openai.api_type = openai_api_type
openai.api_base = openai_api_base
openai.api_version = "2023-06-01-preview"
openai.api_key =  openai_api_key

# Sample inputs for testing
sample_transcript = "Some sample transcript"
sample_guest = "Sample Guest Name"
sample_bio = "Sample guest bio"

# Make a call to the Bing Search Grounding API to retrieve a bio for the guest
def bing_grounding(input_dict: dict) -> dict:
    search_term = input_dict["guest"]
    headers = {"Ocp-Apim-Subscription-Key": bing_subscription_key}
    params = {"q": search_term, "textDecorations": True, "textFormat": "HTML"}
    response = requests.get(bing_search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    bio = search_results["webPages"]["value"][0]["snippet"]
    return {"bio": bio}

@app.route("/generate_social_media_post", methods=["POST"])
def generate_social_media_post():
    data = request.get_json()

    # Step 1 - Set transcript variables
    transcript = data.get("transcript", sample_transcript)

    # Step 2 - Make a call to a local Dolly 2.0 model optimized for Windows to extract the name of who I'm interviewing from the transcript
    guest = data.get("guest", sample_guest)

    # Step 3 - Make a call to the Bing Search Grounding API to retrieve a bio for the guest
    bing_chain = TransformChain(input_variables=["guest"], output_variables=["bio"], transform=bing_grounding)
    bio = bing_chain.run(guest)
    print(bio)
    system_template="You are a helpful large language model that can create a LinkedIn promo blurb for episodes of the podcast Behind the Tech, when given transcripts of the podcasts.  The Behind the Tech podcast is hosted by Kevin Scott.\n"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    user_prompt=PromptTemplate(
        template="Create a short summary of this podcast episode that would be appropriate to post on LinkedIn to promote the podcast episode.  The post should be from the first-person perspective of Kevin Scott, who hosts the podcast.\n" +
                "Here is the transcript of the podcast episode: {transcript} \n" +
                "Here is the bio of the guest: {bio} \n",
        input_variables=["transcript", "bio"],
    )
    human_message_prompt = HumanMessagePromptTemplate(prompt=user_prompt)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    # Get formatted messages for the chat completion
    blurb_messages = chat_prompt.format_prompt(transcript={transcript}, bio={bio}).to_messages()


    # Step 5 - Make a call to Azure OpenAI Service to get a social media blurb, 
    print("Calling GPT-4 model on Azure OpenAI Service to get a social media blurb...\n")
    gpt4 = AzureChatOpenAI(
        openai_api_base=gpt4_endpoint,
        openai_api_version="2023-03-15-preview",
        deployment_name=gpt4_deployment_name,
        openai_api_key=gpt4_api_key,
        openai_api_type = openai_api_type,
    )
    #print(gpt4)   #shows parameters

    output = gpt4(blurb_messages)
    social_media_copy = output.content

    gpt4_chain = LLMChain(llm=gpt4, prompt=chat_prompt, output_key="social_media_copy")

    print("Social Media Copy:\n")
    print(social_media_copy)
    print("\n")

    #imageURL = "test.jpg"
    # Step 5 - Generate a DALL-E prompt
    system_template="You are a helpful large language model that generates DALL-E prompts, that when given to the DALL-E model can generate beautiful high-quality images to use in social media posts about a podcast on technology.  Good DALL-E prompts will contain mention of related objects, and will not contain people or words.  Good DALL-E prompts should include a reference to podcasting along with items from the domain of the podcast guest.\n"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    user_prompt=PromptTemplate(
        template="Create a DALL-E prompt to create an image to post along with this social media text: {social_media_copy}",
        input_variables=["social_media_copy"],
    )
    human_message_prompt = HumanMessagePromptTemplate(prompt=user_prompt)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    # Get formatted messages for the chat completion
    dalle_messages = chat_prompt.format_prompt(social_media_copy={social_media_copy}).to_messages()

    # Call Azure OpenAI Service to get a DALL-E prompt 
    print("Calling GPT-4 model on Azure OpenAI Service to get a DALL-E prompt...\n")
    gpt4 = AzureChatOpenAI(
        openai_api_base=gpt4_endpoint,
        openai_api_version="2023-03-15-preview",
        deployment_name=gpt4_deployment_name,
        openai_api_key=gpt4_api_key,
        openai_api_type = openai_api_type,
    )

    output = gpt4(dalle_messages)
    dalle_prompt = output.content

    dalle_prompt_chain = LLMChain(llm=gpt4, prompt=chat_prompt, output_key="dalle_prompt")


    dalle_prompt += ", high-quality digital art"
    print("DALL-E Prompt:\n")
    print(dalle_prompt)
    print("\n")

    # Step 6 - Generate an image using DALL-E model
    response = openai.Image.create(
        prompt=dalle_prompt,
        size='1024x1024',
        n=1
    )
    imageURL = response["data"][0]["url"]
    print("Image URL: " + imageURL + "\n")

    # Append the podcast URL to the generated social media copy
    podcast_url = data.get("podcast_url", "https://www.microsoft.com/behind-the-tech")
    social_media_copy += " " + podcast_url

    # Return the social media copy and image URL as a JSON response
    response_data = {"social_media_copy": social_media_copy, "image_url": imageURL}
    return jsonify(response_data)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)