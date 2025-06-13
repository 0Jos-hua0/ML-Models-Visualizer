from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
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

        # Call your centralized function
        error_message, result = load_model(model_file)

        if error_message:
            return jsonify({'error': error_message}), 400

        response = {
            "message": "Model uploaded and visualized successfully.",
            "report": result.get("report"),
            "report_summary": result.get("report_summary")  # 🆕 Add this line
        }

        if "plotly" in result:
            response["plotly"] = result["plotly"]
        elif "image" in result:
            response["image"] = result["image"]

        #print("🧾 FINAL RESPONSE:", response)

        return jsonify(response)

    except Exception as e:
        print(" Error during model upload:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
