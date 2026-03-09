"""Flask API for Text-to-Speech NLP Application"""

import os
import sys
from typing import Any, Dict, Optional, Tuple
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from datetime import datetime

# Add project path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'text-to-speech-recognition'))

# Import with type hints
try:
    from src.data.preprocess import preprocess_text  # type: ignore
    from src.tts.synthesize import synthesize_speech  # type: ignore
except ImportError as e:
    print(f"Warning: Could not import TTS modules - {e}")
    # Fallback implementations
    def preprocess_text(text: str):
        """Fallback preprocessing"""
        import string
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.split()
    
    def synthesize_speech(text: str, rate: int = 150, volume: float = 1.0):
        """Fallback synthesis"""
        print(f"TTS: {text} (rate={rate}, volume={volume})")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request size

# Request logging
@app.before_request
def log_request() -> None:
    """Log incoming requests"""
    app.logger.info(f"{request.method} {request.path} - {request.remote_addr}")

# ==================== WEB ROUTES ====================

@app.route('/')
def index() -> str:
    """Serve the main web interface"""
    return render_template('index.html')

# ==================== API ROUTES ====================

@app.route('/api/health', methods=['GET'])
def health() -> Tuple[Response, int]:
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Text-to-Speech NLP API',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }), 200

@app.route('/api/preprocess', methods=['POST'])
def preprocess() -> Tuple[Response, int]:
    """Preprocess text endpoint"""
    try:
        data: Optional[Dict[str, Any]] = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing required field: text'}), 400
        
        text: str = data['text'].strip()
        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        # Preprocess
        tokens: list = preprocess_text(text)
        cleaned_text: str = ' '.join(tokens)
        
        return jsonify({
            'original_text': text,
            'cleaned_text': cleaned_text,
            'tokens': tokens,
            'token_count': len(tokens),
            'original_length': len(text),
            'cleaned_length': len(cleaned_text),
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        app.logger.error(f"Preprocessing error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/synthesize', methods=['POST'])
def synthesize() -> Tuple[Response, int]:
    """Text-to-speech synthesis endpoint"""
    try:
        data: Optional[Dict[str, Any]] = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing required field: text'}), 400
        
        text: str = data['text'].strip()
        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        # Optional parameters
        rate: int = data.get('rate', 150)
        volume: float = data.get('volume', 1.0)
        
        # Validate parameters
        if not isinstance(rate, (int, float)) or rate < 50 or rate > 300:
            return jsonify({'error': 'Rate must be between 50 and 300'}), 400
        if not isinstance(volume, (int, float)) or volume < 0 or volume > 1:
            return jsonify({'error': 'Volume must be between 0 and 1'}), 400
        
        # Preprocess first
        tokens = preprocess_text(text)
        processed_text = ' '.join(tokens)
        
        # Synthesize
        synthesize_speech(processed_text, rate=rate, volume=volume)
        
        return jsonify({
            'original_text': text,
            'processed_text': processed_text,
            'rate': rate,
            'volume': volume,
            'status': 'success',
            'message': 'Speech synthesis completed',
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        app.logger.error(f"Synthesis error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/pipeline', methods=['POST'])
def pipeline() -> Tuple[Response, int]:
    """Complete pipeline: preprocess and synthesize"""
    try:
        data: Optional[Dict[str, Any]] = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing required field: text'}), 400
        
        text: str = data['text'].strip()
        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        # Get optional parameters
        rate: int = data.get('rate', 150)
        volume: float = data.get('volume', 1.0)
        auto_speak: bool = data.get('auto_speak', True)
        
        # Step 1: Preprocess
        tokens = preprocess_text(text)
        cleaned_text = ' '.join(tokens)
        
        # Step 2: Synthesize (if requested)
        if auto_speak:
            synthesize_speech(cleaned_text, rate=rate, volume=volume)
        
        return jsonify({
            'status': 'success',
            'pipeline_steps': ['preprocessing', 'synthesis' if auto_speak else 'skipped'],
            'input_text': text,
            'cleaned_text': cleaned_text,
            'tokens': tokens,
            'token_count': len(tokens),
            'rate': rate if auto_speak else None,
            'volume': volume if auto_speak else None,
            'message': 'Pipeline execution completed',
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        app.logger.error(f"Pipeline error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/info', methods=['GET'])
def info() -> Tuple[Response, int]:
    """Get API information and available endpoints"""
    return jsonify({
        'service_name': 'Text-to-Speech NLP API',
        'version': '1.0.0',
        'description': 'A complete text-to-speech system with NLP preprocessing',
        'endpoints': {
            'GET /': 'Web interface',
            'GET /api/health': 'Health check',
            'POST /api/preprocess': 'Text preprocessing only',
            'POST /api/synthesize': 'Full synthesis pipeline',
            'POST /api/pipeline': 'Complete pipeline with options',
            'GET /api/info': 'API information'
        },
        'request_examples': {
            'preprocess': {
                'endpoint': '/api/preprocess',
                'method': 'POST',
                'body': {'text': 'Hello, how are you?'}
            },
            'synthesize': {
                'endpoint': '/api/synthesize',
                'method': 'POST',
                'body': {
                    'text': 'Hello, how are you?',
                    'rate': 150,
                    'volume': 1.0
                }
            },
            'pipeline': {
                'endpoint': '/api/pipeline',
                'method': 'POST',
                'body': {
                    'text': 'Hello, how are you?',
                    'rate': 150,
                    'volume': 1.0,
                    'auto_speak': True
                }
            }
        }
    }), 200

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error: Exception) -> Tuple[Response, int]:
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error: Exception) -> Tuple[Response, int]:
    """Handle 405 errors"""
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error: Exception) -> Tuple[Response, int]:
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Development server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=True
    )
