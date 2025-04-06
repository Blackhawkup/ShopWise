import numpy as np
import pandas as pd
import os
from apyori import apriori
import requests
import json

class RecommendationEngine:
    def __init__(self, file_path='.\Market_Basket_Optimisation.csv'):
        print(f"📂 Loading data from: {file_path}")
        self.dataset = pd.read_csv(file_path, header=None)
        self.transactions = []
        for i in range(len(self.dataset)):
            self.transactions.append([str(self.dataset.values[i,j]) for j in range(20) if str(self.dataset.values[i,j]) != 'nan'])
        print(f"🛒 Total transactions loaded: {len(self.transactions)}")
        
        # Generate association rules
        print("⚙️ Generating association rules...")
        self.rules = list(apriori(
            transactions=self.transactions, 
            min_support=0.003, 
            min_confidence=0.2, 
            min_lift=3, 
            min_length=2, 
            max_length=2
        ))
        print(f"📊 Total rules generated: {len(self.rules)}")
        
        # Extract rules into a more accessible format
        self.rule_dict = {}
        for item in self.rules:
            pair = item[0]
            items = [x for x in pair]
            if len(items) >= 2:  # Ensure we have pairs
                for i in range(len(items)):
                    for j in range(len(items)):
                        if i != j and items[i] != 'nan' and items[j] != 'nan':
                            if items[i] not in self.rule_dict:
                                self.rule_dict[items[i]] = []
                            self.rule_dict[items[i]].append(items[j])
        
    def get_recommendations(self, item):
        """Get recommendations for a given item"""
        if item in self.rule_dict:
            return self.rule_dict[item]
        return []
        
    def get_items_list(self):
        """Get list of all unique items"""
        return list(self.rule_dict.keys())


class OllamaChatbot:
    def __init__(self, model="gemma3:latest", recommendation_engine=None):
        self.model = model
        self.base_url = "http://localhost:11434/api"
        self.recommendation_engine = recommendation_engine
        self.conversation_history = []
        
        # Test connection
        try:
            requests.get(f"{self.base_url}/version")
            print(f"✅ Connected to Ollama API using {model} model")
        except:
            print("⚠️ Warning: Could not connect to Ollama API. Make sure Ollama is running.")
        
    def generate_response(self, user_input):
        """Generate a response using the Ollama API"""
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Prepare context with conversation history
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.conversation_history])
        
        # Create the API request
        url = f"{self.base_url}/generate"
        payload = {
            "model": self.model,
            "prompt": context,
            "stream": False
        }
        
        try:
            print("🤔 Thinking...")
            response = requests.post(url, json=payload)
            response_json = response.json()
            
            if "response" in response_json:
                bot_response = response_json["response"]
                self.conversation_history.append({"role": "assistant", "content": bot_response})
                return bot_response
            else:
                return "Error: Unable to generate response."
        except Exception as e:
            return f"Error connecting to Ollama API: {str(e)}"
    
    def process_input(self, user_input):
        """Process user input and enhance with recommendations"""
        # Check if user is asking about recommendations
        if "recommend" in user_input.lower() or "suggestion" in user_input.lower():
            # Extract potential items from user input
            if self.recommendation_engine:
                items_list = self.recommendation_engine.get_items_list()
                mentioned_items = [item for item in items_list if item.lower() in user_input.lower()]
                
                recommendations = []
                for item in mentioned_items:
                    item_recommendations = self.recommendation_engine.get_recommendations(item)
                    if item_recommendations:
                        recommendations.extend(item_recommendations)
                
                if recommendations:
                    # Generate response with Ollama
                    base_response = self.generate_response(user_input)
                    
                    # Enhance with recommendations
                    recommendation_text = f"\n\nBased on your interest in {', '.join(mentioned_items)}, "
                    recommendation_text += f"you might also like: {', '.join(set(recommendations))}."
                    
                    return base_response + recommendation_text
        
        # Default to regular response
        return self.generate_response(user_input)


def main():
    print("\n" + "="*50)
    print("🤖 Terminal-based Recommendation Chatbot")
    print("="*50)
    
    # Initialize recommendation engine
    file_path = './Market_Basket_Optimisation.csv'
    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        alt_path = input("Enter the path to your CSV file: ")
        if alt_path:
            file_path = alt_path
    
    try:
        rec_engine = RecommendationEngine(file_path)
        print("✅ Recommendation engine initialized successfully")
    except Exception as e:
        print(f"⚠️ Error initializing recommendation engine: {str(e)}")
        print("⚠️ Continuing without recommendation functionality")
        rec_engine = None
    
    # Choose Ollama model
    default_model = "gemma:2b"
    model_choice = input(f"Enter Ollama model name (default: {default_model}): ")
    model = model_choice if model_choice else default_model
    
    # Initialize chatbot
    chatbot = OllamaChatbot(model=model, recommendation_engine=rec_engine)
    
    print("\n" + "-"*50)
    print("💬 Chatbot is ready! Type 'exit' to quit.")
    print("💡 Try asking about product recommendations.")
    print("-"*50)
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break
            
        response = chatbot.process_input(user_input)
        print(f"\nBot: {response}")


if __name__ == "__main__":
    main()