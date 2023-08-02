def related_links(url):
    #Import Modules
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity
    import torch
    from transformers import BertModel, BertTokenizer
    import json
    import pickle
    import numpy as np
    
    save_directory = "/Users/david/Desktop/QA Media/Models"

    # Load the model and tokenizer from the directory
    model = BertModel.from_pretrained(save_directory)
    tokenizer = BertTokenizer.from_pretrained(save_directory)
    
    # Load the dictionary from a file
    with open('dict_file.json', 'r') as file:
        articles = json.load(file)
        
    
    if url not in list(articles.keys()):   
        article_content = [x for x in articles.values()]

        # Function to get embeddings
        def get_embeddings(text):
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            outputs = model(**inputs)
            return outputs.last_hidden_state[:, 0, :].detach().numpy()

        # Assume `documents` is a list of your news articles 
        with open('arrays.pkl', 'rb') as f:
            embeddings = pickle.load(f)

        new_url = url

        new_response = requests.get(new_url)
        new_soup = BeautifulSoup(new_response.text, 'html.parser')
        new_text = new_soup.find('meta', attrs={'name': 'description'}).get('content')
        if new_url not in list(articles.keys()):
            articles[new_url] = new_text

        new_embedding = get_embeddings(articles[new_url])

        # Compute similarity
        similarity_scores = [cosine_similarity(new_embedding, emb) for emb in embeddings]

        # Get top 5 similar documents
        top_5 = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:5]

        embeddings.append(new_embedding)

        # Save the list of arrays.
        with open('arrays.pkl', 'wb') as f:
            pickle.dump(embeddings, f)

        # Save the dictionary to a file
        with open('dict_file.json', 'w') as file:
            json.dump(articles, file)

        top_5_urls = []

        for i in top_5:
            top_5_urls.append([x for x in articles.keys()][i])
            top_5_urls = [x for x in top_5_urls if x != new_url]
            
    else:
        article_content = [x for x in articles.values()]

        # Function to get embeddings
        def get_embeddings(text):
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            outputs = model(**inputs)
            return outputs.last_hidden_state[:, 0, :].detach().numpy()

        # Assume `documents` is a list of your news articles 
        with open('arrays.pkl', 'rb') as f:
            embeddings = pickle.load(f)

        new_url = url

        new_response = requests.get(new_url)
        new_soup = BeautifulSoup(new_response.text, 'html.parser')
        new_text = new_soup.find('meta', attrs={'name': 'description'}).get('content')
        new_embedding = get_embeddings(articles[new_url])

        # Compute similarity
        similarity_scores = [cosine_similarity(new_embedding, emb) for emb in embeddings]

        # Get top 5 similar documents
        top_10 = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:10]

        top_5_urls = []

        for i in top_10:
            top_5_urls.append([x for x in articles.keys()][i])
            top_5_urls = [x for x in top_5_urls if x != new_url][:5]
    

    return top_5_urls