from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
os.system("pip install torch==2.1.0")
import torch
from model.ae_transformer import AETransformer
from model.utils import preprocess_data
from flask_cors import CORS
import traceback
import logging
from classification_pipeline import run_classification_pipeline

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up folder for file uploads
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'mat'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 60 * 1024 * 1024  # 60MB max file size

# Ensure uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load pre-trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'ae_transformer_model.pth')
model = None
try:
    input_dim = 40
    latent_dim = 32
    model = AETransformer(input_dim=input_dim, latent_dim=latent_dim)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    logger.info(f"AETransformer model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading AETransformer model from {MODEL_PATH}: {e}")
    logger.error(traceback.format_exc())

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    try:
        logger.info(f"Serving file: {filename}")
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        logger.error(f"Error serving file {filename}: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Error serving file: {str(e)}'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        logger.info("Received upload request")
        
        # Check for both files in the request
        if 'hsi_file' not in request.files or 'gt_file' not in request.files:
            logger.error("Missing files in request")
            return jsonify({'error': 'Both hsi_file and gt_file must be provided'}), 400

        hsi_file = request.files['hsi_file']
        gt_file = request.files['gt_file']

        if hsi_file.filename == '' or gt_file.filename == '':
            logger.error("Empty filenames")
            return jsonify({'error': 'No selected file(s)'}), 400

        dataset_name = request.form.get('dataset_name')
        if not dataset_name:
            logger.error("No dataset name provided")
            return jsonify({'error': 'No dataset_name provided'}), 400

        logger.info(f"Processing files: {hsi_file.filename}, {gt_file.filename} for dataset: {dataset_name}")

        if not allowed_file(hsi_file.filename) or not allowed_file(gt_file.filename):
            logger.error(f"Invalid file type: {hsi_file.filename}, {gt_file.filename}")
            return jsonify({'error': 'File type not allowed'}), 400

        hsi_filename = secure_filename(hsi_file.filename)
        gt_filename = secure_filename(gt_file.filename)

        hsi_path = os.path.join(app.config['UPLOAD_FOLDER'], hsi_filename)
        gt_path = os.path.join(app.config['UPLOAD_FOLDER'], gt_filename)

        try:
            hsi_file.save(hsi_path)
            gt_file.save(gt_path)
            logger.info(f"Files saved: {hsi_path}, {gt_path}")

            # Import the pipeline utility function
            from utils import run_pipeline_with_files

            # Run the pipeline with uploaded files using new pipeline
            logger.info("Starting pipeline processing")

            # Get optional patch_size from form data, default to 16
            patch_size_str = request.form.get('patch_size', '16')
            try:
                patch_size = int(patch_size_str)
                if patch_size <= 0 or patch_size > 32:
                    raise ValueError("patch_size must be between 1 and 32")
            except ValueError:
                logger.error(f"Invalid patch_size value: {patch_size_str}")
                return jsonify({'error': 'Invalid patch_size parameter, must be an integer between 1 and 32'}), 400

            results = run_pipeline_with_files(hsi_path, gt_path, dataset_name, patch_size=patch_size)
            logger.info("Pipeline processing completed")

            # Add URLs for output images to results dynamically based on filenames in results
            base_url = request.host_url.rstrip('/')
            images = []
            for img in results.get('images', []):
                filename = img.get('url', '').split('/')[-1]
                images.append({
                    'url': f"{base_url}/uploads/{filename}",
                    'name': img.get('name', ''),
                    'description': img.get('description', '')
                })
            results['images'] = images

            # Add uploaded file paths and dataset_name to results for frontend use
            results['hsi_path'] = hsi_filename
            results['gt_path'] = gt_filename
            results['dataset_name'] = dataset_name

            return jsonify({'message': 'Files uploaded and processed successfully', 'results': results})

        except Exception as e:
            logger.error(f"Error during file processing: {e}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f'File processing failed: {str(e)}'}), 500

    except Exception as e:
        logger.error(f"Unexpected error in upload route: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json()
    if not data or 'input_data' not in data:
        return jsonify({'error': 'Invalid input data'}), 400

    try:
        # Convert input data to torch tensor
        input_data = data['input_data']
        input_tensor = torch.tensor(input_data, dtype=torch.float32)

        # Reshape input_tensor to 2D (num_samples, input_dim)
        if input_tensor.ndim == 3:
            h, w, c = input_tensor.shape
            input_tensor = input_tensor.reshape(-1, c)
        elif input_tensor.ndim == 2:
            # Already 2D, no change
            pass
        else:
            return jsonify({'error': 'Input data has invalid shape'}), 400

        print(f"Input tensor shape for prediction: {input_tensor.shape}")

        x_recon, z_trans = model.predict(input_tensor)  # Get reconstructed output and latent
        # Compute reconstruction error (MSE) per sample
        recon_error = torch.mean((input_tensor - x_recon) ** 2, dim=1)
        prediction_list = recon_error.tolist()

        print(f"Reconstruction error shape: {recon_error.shape}")
    except Exception as e:
        print(f"Prediction error: {str(e)}")  # Debugging output
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    return jsonify({'prediction': prediction_list})


from model.utils import load_test_data, evaluate_ae_model

@app.route('/metrics', methods=['GET'])
def metrics():
    dataset_name = request.args.get('dataset_name', 'pavia')  # default to 'pavia'
    test_file_path = request.args.get('test_file_path', 'backend/data/pavia.mat')  # default path

    try:
        test_data, test_labels = load_test_data(test_file_path, dataset_name)
        accuracy, confusion_mat = evaluate_ae_model(model, test_data, test_labels)
        metrics = {
            'accuracy': accuracy,
            'confusion_matrix': confusion_mat
        }
    except Exception as e:
        return jsonify({'error': f'Failed to compute metrics: {str(e)}'}), 500

    return jsonify(metrics)

@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.get_json()
        logger.info(f"Received classify request data: {data}")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Request remote addr: {request.remote_addr}")
        hsi_filename = data.get('hsi_path')
        gt_filename = data.get('gt_path')
        dataset_name = data.get('dataset_name')
        if not hsi_filename or not gt_filename or not dataset_name:
            logger.error(f"Missing required parameters in classify request: hsi_path={hsi_filename}, gt_path={gt_filename}, dataset_name={dataset_name}")
            return jsonify({'error': 'Missing required parameters'}), 400

        classification_results = run_classification_pipeline(hsi_filename, gt_filename, dataset_name, app.config['UPLOAD_FOLDER'])

        base_url = request.host_url.rstrip('/')
        classification_image_url = f"{base_url}/uploads/{os.path.basename(classification_results['anomaly_overlay_path'])}"
        tsne_image_url = f"{base_url}/uploads/{os.path.basename(classification_results.get('tsne_visualization_path', ''))}" if classification_results.get('tsne_visualization_path') else None
        anomaly_score_map_url = f"{base_url}/uploads/{os.path.basename(classification_results.get('anomaly_score_map_path', ''))}" if classification_results.get('anomaly_score_map_path') else None
        anomaly_report_csv = f"{base_url}/uploads/anomaly_report_{dataset_name}.csv"

        return jsonify({
            'message': 'Classification completed',
            'classification_image_url': classification_image_url,
            'tsne_image_url': tsne_image_url,
            'anomaly_score_map_url': anomaly_score_map_url,
            'anomaly_report_csv_url': anomaly_report_csv
        })

    except Exception as e:
        logger.error(f"Error in classification endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Classification failed: {str(e)}'}), 500

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'Backend is connected'})

@app.route('/', methods=['GET'])
def root():
    return jsonify({'message': 'Backend is running'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
