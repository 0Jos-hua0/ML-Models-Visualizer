from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import os
from stellarbasin import load_model

app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload_model():
    try:
        if 'model' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        model_file = request.files['model']
        if model_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the file temporarily
        os.makedirs('uploads', exist_ok=True)
        file_path = os.path.join('uploads', model_file.filename)
        model_file.save(file_path)

        # Call the model loader
        error_message, result = load_model(file_path)
        
        # Clean up the uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

        if error_message:
            return jsonify({'error': error_message}), 400

        # Prepare response
        response = {
            'message': 'Model loaded and visualized successfully!',
            'model_type': result.get('model_type', 'Unknown'),
            'explanation': result.get('explanation', {}),
        }

        # Add visualization based on type
        visualization = result.get('visualization', {})
        if visualization.get('type') == 'plotly':
            response['plotly'] = visualization.get('data')
        elif visualization.get('type') == 'image':
            response['image'] = visualization.get('data')

        return jsonify(response)

    except Exception as e:
        print("❌ Error during model upload:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
