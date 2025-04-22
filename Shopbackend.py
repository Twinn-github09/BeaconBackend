from flask import Flask, request, jsonify, render_template ,send_file
from groq import Groq
import os
from dotenv import load_dotenv
import requests
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
import asyncio
from langchain.chains.retrieval_qa.base import RetrievalQA
import google.generativeai as genai
from flask_cors import cross_origin
from werkzeug.utils import secure_filename
import base64
from google.generativeai import GenerativeModel
from pymongo import MongoClient
from flask_cors import CORS
from langchain_scrapegraph.tools import SmartScraperTool
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
# Load environment variables
load_dotenv()

# API keys
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
client =Groq(api_key="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")

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
SERPER_API_KEY = os.getenv('SERPER_API_KEY')
SGAI_API_KEY = os.getenv('SGAI_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)
model = GenerativeModel('gemini-1.5-pro')

app = Flask(__name__)
CORS(app) 
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Fetch a sample document from the collection
def fetch_user_medical_record(user_id):
    uri = "mongodb+srv://twinnroshan:Roseshopping@cluster0.zf5b3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(uri)
    db = client['ShoppingSys']  
    medical_collection = db['CustomerForms'] 
    medical_record = medical_collection.find_one({"email": user_id})
    if medical_record:
        return {key: value for key, value in medical_record.items() if key != "_id"}
    return {"health_conditions": []}

# RAG Setup Functions
def load_vector_store(persist_directory="./chroma_db_final_db"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma(persist_directory=persist_directory,
                          embedding_function=embeddings)
    return vector_store

def create_qa_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",
        temperature=0.1,
        model_kwargs={
            "max_output_tokens": 8192,
            "top_k": 10,
            "top_p": 0.95
        }
    )
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

# Web Scraping Functions
@app.route('/chat', methods=['POST'])
@cross_origin()
def chat():
    try:
        data = request.json
        print(data)
        if not data or 'user_prompt' not in data:
            return jsonify({"error": "User prompt is required."}), 400

        user_prompt = data['user_prompt']
        emailid = data['email']
        medical_info = fetch_user_medical_record(emailid)
        print(medical_info)
    
        greetings = {"hi", "hii", "hello", "hey", "hola", "namaste",'hi there'}
        if user_prompt.lower() in greetings:
            return jsonify({"main_response": "Welcome to BeaconSmart?", "related_cases": []}), 200

        if not isinstance(user_prompt, str) or not user_prompt.strip():
            return jsonify({"error": "Invalid user prompt"}), 400

        try:
            vector_store = load_vector_store()
            qa_chain = create_qa_chain(vector_store)
            rag_response = qa_chain.invoke({"query": user_prompt})
            print(rag_response)
            print(rag_response["result"])
        except Exception as e:
            print(f"RAG system error: {str(e)}")
            return jsonify({"error": "Failed to process with RAG system"}), 500

        # 3. Get Groq Response 
        try:
            messages = [
                {"role": "system", "content":  f'''You are a professional nutritionist. Given the text containing food-related information, extract and summarize the key nutritional values.  
                    Your response should include:
                    {rag_response['result']}
                    Given a user with the following Medical details:  
                    - Name: {medical_info['name']}    
                    - Age: {medical_info['age']}  
                    - Gender: {medical_info['gender']}  
                    - Favorite Foods: {', '.join(medical_info['favorite_foods'])}  
                    - Allergic Foods: {', '.join(medical_info['allergic_foods'])}  
                    - Medical Conditions: {', '.join(medical_info['medical_conditions'])}  
                    - Married: {medical_info['married']}  
                    - Number of Children: {medical_info['children']}  
                    
                    provide a concise dietary recommendation.  

                    Your response should include:  
                    - The macronutrient breakdown (carbohydrates, proteins, and fats).  
                    - Key vitamins and minerals present in the food.  
                    - Any dietary considerations (e.g., suitable for diabetics, high protein for muscle gain).  
                    - A concise, user-friendly summary for better understanding. 

                    Ensure the response is clear, factual, and practical for individuals managing "medical_condition".  
                    '''  
                },
                {"role": "user", "content": f"""
                    Question: {user_prompt}
                    Relevant Context: {rag_response['result']}
                """}
            ]

            groq_response =client.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                temperature=0.7,
                
            )
        except Exception as e:
            print(f"Groq API error: {str(e)}")
            return jsonify({"error": "Failed to get AI response"}), 500

       
        response_data = {
            "main_response": groq_response.choices[0].message.content,
            }

        return jsonify(response_data), 200

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500
    
@app.route('/recipe', methods=['POST'])
@cross_origin()
def recipe():
    try:
        data = request.json
        
        print(data)
        if not data or 'user_prompt' not in data:
            return jsonify({"error": "User prompt is required."}), 400
        user_prompt = data['user_prompt']
        emailid = data['email']
        medical_info = fetch_user_medical_record(emailid)
        print(medical_info)
        greetings = {"hi", "hii", "hello", "hey", "hola", "namaste"}
        if user_prompt.lower() in greetings:
            return jsonify({"main_response": "Welcome to BeaconSmart?", "related_cases": []}), 200

        if not isinstance(user_prompt, str) or not user_prompt.strip():
            return jsonify({"error": "Invalid user prompt"}), 400

        try:
            query = f"site:indiantamilrecipe.com {user_prompt}"
            response = requests.post(
                'https://google.serper.dev/search',
                headers={'X-API-KEY': SERPER_API_KEY},
                json={'q': query}
            )
            result = response.json()
            recipe_url = result['organic'][0]['link']

            print("Recipe URL:", recipe_url)
            os.environ["SGAI_API_KEY"] = "sgai-530ee0ce-11bc-4c99-8c8a-3918ce5cc9b4"
            tool = SmartScraperTool()
            scraped_result = tool.invoke({
                "website_url": recipe_url,
                "user_prompt": f"Get the recipe for {user_prompt}",
            })
            print("Scraped Result:", scraped_result)
            # Pass the scraped result to Groq for further processing
            messages = [
                {"role": "system", "content":  f'''You are a smart AI assistant for a digital supermarket.
                Given a user with the following personal details:  
                - Name: {medical_info['name']}    
                - Age: {medical_info['age']}  
                - Gender: {medical_info['gender']}  
                - Favorite Foods: {', '.join(medical_info['favorite_foods'])}  
                - Allergic Foods: {', '.join(medical_info['allergic_foods'])}  
                - Medical Conditions: {', '.join(medical_info['medical_conditions'])}  
                - Married: {medical_info['married']}  
                - Number of Children: {medical_info['children']}  
                generate a structured shopping list of recommended foods suitable for this condition.  

                The response should include:  
                - A list of suitable ingredients with practical quantities (e.g., "1 kg of whole grains", "500 ml of low-fat milk").  
                - Logical grouping of items (fruits, vegetables, proteins, grains, etc.).  
                - Consideration of dietary restrictions and common alternatives when applicable.  

                Additionally, provide the full recipe details from the given recipe content.  
                '''  },
                {"role": "user", "content": f"""
                    Question: {user_prompt}
                    Recipe: {scraped_result}
                """}
            ]

            groq_response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages, 
                temperature=0.7,
             
            )
        except Exception as e:
            print(f"Groq API error: {str(e)}")
            return jsonify({"error": "Failed to get AI response"}), 500

       
        response_data = {
            "main_response": groq_response.choices[0].message.content,
            }

        return jsonify(response_data), 200

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500
    
@app.route('/get_name', methods=["POST"])
@cross_origin()
def get_name():
    current_user_id = request.args.get('email')
    uri = "mongodb+srv://twinnroshan:Roseshopping@cluster0.zf5b3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(uri)
    db = client["ShoppingSys"]
    collection = db["CustomerForms"]
    name = collection.find_one({"email": current_user_id}, {"name": 1, "_id": 0})
    customer_data = {
            "name": name
        }
    return jsonify(customer_data)
    
@app.route('/submit', methods=["POST"])
@cross_origin()
def submit():
    uri = "mongodb+srv://twinnroshan:Roseshopping@cluster0.zf5b3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(uri)
    db = client["ShoppingSys"]
    collection = db["CustomerForms"]
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        required_fields = ["email", "name", "age", "gender"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        favorite_foods = data.get("favorite_foods", [])
        allergic_foods = data.get("allergic_foods", [])
        medical_conditions = data.get("medical_conditions", [])

        if isinstance(favorite_foods, str):
            favorite_foods = [food.strip() for food in favorite_foods.split(",")]
        if isinstance(allergic_foods, str):
            allergic_foods = [food.strip() for food in allergic_foods.split(",")]
        if isinstance(medical_conditions, str):
            medical_conditions = [condition.strip() for condition in medical_conditions.split(",")]

        customer_data = {
            "email": data["email"],
            "name": data["name"],
            "age": int(data["age"]),
            "gender": data["gender"],
            "favorite_foods": favorite_foods,
            "married": data.get("married", False),
            "children": int(data.get("children", 0)),
            "allergic_foods": allergic_foods,
            "medical_conditions": medical_conditions
        }

        collection.insert_one(customer_data)
        return jsonify({"message": "Form submitted successfully!"}), 200

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

client1 = MongoClient("mongodb+srv://Shopping:Shopping123456@grocery-items.5qfdu.mongodb.net/")
db1 = client1['IoT_project_data']
collection = db1['Iot_ML_project']

@app.route('/get_product_details', methods=['POST'])
@cross_origin()
def get_product_details():
    try:
        data = request.get_json()
        product_name = data.get('productName', '').lower().strip()

        if not product_name:
            return jsonify({"error": "Product name is required"}), 400

        # Search for products with similar names (case insensitive)
        query = {
            "ProductName": {"$regex": f".*{product_name}.*", "$options": "i"}
        }

        products = list(collection.find(query, {
            "_id": 0,  # Exclude MongoDB _id field
            "ProductName": 1,
            "Brand": 1,
            "Price": 1,
            "Image_Url": 1,
            "Quantity": 1
        }).limit(10))  # Limit to 10 results

        if not products:
            return jsonify({"message": "No products found", "products": []}), 404

        return jsonify(products)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


uri2 = "mongodb+srv://twinnroshan:Roseshopping@cluster0.zf5b3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client2 = MongoClient(uri2)
db2 = client2["ShoppingSys"]
collection2 = db2["FruitvegDetail"]

SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_ADDRESS = 'twinnroshan@gmail.com'
EMAIL_PASSWORD = 'mzqx fokt jzjm dptu'

@app.route('/get_live_prices', methods=['GET'])
def get_live_prices():
    try:
        # Get all products from MongoDB
        products = list(collection2.find({}))
        
        # Convert ObjectId to string for JSON serialization
        for product in products:
            product['_id'] = str(product['_id'])
        
        return jsonify(products)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/send_price_drop_email', methods=['POST'])
def send_price_drop_email():
    try:
        data = request.json
        email = data.get('email')
        price_changes = data.get('priceChanges', {})
        
        if not email or not price_changes:
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Create email content
        subject = "Price Drop Alert from BeaconSmart"
        
        # Create HTML content for the email
        html = f"""
        <html>
            <body>
                <h2>Price Drop Alert!</h2>
                <p>Hello,</p>
                <p>We've detected price drops on products you might be interested in:</p>
                <table border="1" cellpadding="5" cellspacing="0">
                    <tr>
                        <th>Product</th>
                        <th>Old Price</th>
                        <th>New Price</th>
                        <th>Savings</th>
                    </tr>
        """
        
        for product_name, change in price_changes.items():
            html += f"""
                    <tr>
                        <td>{product_name}</td>
                        <td>₹{change['oldPrice']}</td>
                        <td>₹{change['newPrice']}</td>
                        <td>{change['percentage']}% (₹{change['oldPrice'] - change['newPrice']})</td>
                    </tr>
            """
        
        html += """
                </table>
                <p>Happy shopping!</p>
                <p>Best regards,<br>The BeaconSmart Team</p>
            </body>
        </html>
        """
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = email
        msg['Subject'] = subject
        msg.attach(MIMEText(html, 'html'))
        
        # Send email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        
        return jsonify({'message': 'Email sent successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
def home():
    return render_template('bot.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=5000)