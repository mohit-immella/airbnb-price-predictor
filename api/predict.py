from http.server import BaseHTTPRequestHandler
import json
import joblib
import pandas as pd
import numpy as np
import os

model_path = os.path.join(os.path.dirname(__file__), '..', 'airbnb_final_pipeline.pkl')
model = joblib.load(model_path)

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode('utf-8'))
            
            input_df = pd.DataFrame({
                'city': [data.get('city', 'Goa')],
                'room_type': [data.get('room_type', 'Entire home/apt')],
                'accommodates': [int(data.get('accommodates', 2))],
                'bedrooms': [int(data.get('bedrooms', 1))],
                'amenities': [data.get('amenities', 'Wi-Fi, Air Conditioning')]
            })
            
            log_pred = model.predict(input_df)
            predicted_price = float(np.expm1(log_pred)[0])
            
            response = {
                "success": True,
                "predicted_price": round(predicted_price, 2)
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
