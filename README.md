# 📝 Handwritten Math Grader

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red### 🌐 Deployment

### Streamlit Cloud
1. Fork this repository
2. Connect your GitHub account to [Streamlit Cloud](https://share.streamlit.io/)
3. Create a new app and select your forked repository
4. Add your Gemini API key in the Streamlit Cloud secrets manager:
   ```toml
   [gemini]
   api_key = "your_gemini_api_key_here"
   ```
5. The app will automatically install system dependencies from `packages.txt`
6. Deploy and share your app

**Note**: The application automatically handles cloud deployment by gracefully falling back to PIL-only image processing when OpenCV system libraries are not available.

For detailed deployment instructions, see `docs/deployment-streamlit-cloud.md`ps://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent Streamlit application that converts handwritten mathematical solutions to editable text and automatically grades them using advanced symbolic validation and AI-powered evaluation.
<img width="1906" height="858" alt="image" src="https://github.com/user-attachments/assets/255b4354-569d-44ab-b81e-93d4bd52e519" />
<img width="1915" height="847" alt="image" src="https://github.com/user-attachments/assets/46f1dfce-cea5-4d2b-bbc8-5de3d4ddb54c" />
<img width="1902" height="867" alt="image" src="https://github.com/user-attachments/assets/5da1e4d9-2e18-4f1e-a04e-5bd14a4f1dbe" />



## 🚀 Features

### 📸 Advanced OCR Processing
- **Google Gemini Vision API**: Primary OCR with mathematical notation recognition
- **Enhanced LaTeX Support**: Automatic conversion from LaTeX to readable mathematical notation
- **Tesseract Fallback**: Reliable fallback when Gemini is unavailable
- **Layout Preservation**: Maintains original document structure and formatting

### 🧮 Intelligent Mathematical Analysis
- **SymPy Integration**: Symbolic mathematical expression validation
- **Enhanced Expression Parsing**: Robust handling of complex mathematical notation
- **Equation Extraction**: Automatic identification and parsing of mathematical equations
- **Geometry Problem Solving**: Specialized angle calculation and geometric problem resolution

### 🎯 AI-Powered Grading
- **Multi-Criteria Evaluation**: Assesses correctness, methodology, clarity, and completeness
- **Detailed Feedback**: Comprehensive analysis with improvement suggestions
- **Symbolic Validation**: Mathematical expression equivalence checking
- **Error Handling**: Graceful handling of parsing errors with informative feedback

### 📤 Export & Documentation
- **Word Document Export**: Generate professional reports with grading results
- **Structured Output**: Well-formatted mathematical content
- **Session Management**: Persistent state across user interactions

## 🛠️ Technical Requirements

- **Python**: 3.10 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 500MB free space

### Required Dependencies
```
streamlit>=1.28.0
google-generativeai>=0.3.0
sympy>=1.12
opencv-python>=4.8.0
pytesseract>=0.3.10
pillow>=10.0.0
numpy>=1.24.0
python-docx>=0.8.11
python-dotenv>=1.0.0
requests>=2.31.0
```

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd handwritten-grader
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Tesseract (Required)
**Windows:**
1. Download Tesseract from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install and add to PATH
3. Verify installation: `tesseract --version`

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

### 5. API Configuration

#### Option A: Environment Variables (Recommended for Local Development)
```bash
# Copy the example file
cp .env.example .env

# Edit .env file and add your Gemini API key:
GEMINI_API_KEY=your_gemini_api_key_here
```

#### Option B: Streamlit Secrets (For Cloud Deployment)
```bash
# Copy the example secrets file
cp .streamlit/secrets.toml.example .streamlit/secrets.toml

# Edit secrets.toml and add:
[gemini]
api_key = "your_gemini_api_key_here"
```

### 6. Get Gemini API Key (Optional but Recommended)
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file or `secrets.toml`

*Note: The application works without Gemini using Tesseract fallback, but Gemini provides superior mathematical notation recognition.*

## 🚀 Running the Application

```bash
streamlit run frontend/streamlit_app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📖 User Guide

### Step 1: Upload Images
- Click "Browse files" to upload your question and solution images
- Supported formats: PNG, JPEG, JPG
- Images should be clear and well-lit for best OCR results

### Step 2: Extract Text
- Click "🚀 Extract Text" to perform OCR
- The system will try Gemini first, then fallback to Tesseract
- Review the extracted regions and confidence scores

### Step 3: Edit Text (Optional)
- Use the "✏️ Edit Extracted Text" area to make corrections
- Fix any OCR errors or unclear mathematical notation
- The editor preserves line breaks and formatting

### Step 4: Grade Solution
- Click "🧮 Grade Solution" to analyze the mathematical content
- Review symbolic validation results
- Read AI-generated feedback and suggestions

### Step 5: Export Results
- Click "📄 Export to Word" to generate a report
- Download the `.docx` file with complete grading analysis

## 🏗️ Project Architecture

```
handwritten-grader/
├── frontend/                 # Streamlit UI components
│   ├── streamlit_app.py     # Main application entry point
│   └── components/          # Reusable UI components
│       ├── uploader.py      # File upload interface
│       ├── image_viewer.py  # Image display component
│       ├── ocr_editor.py    # OCR results editor
│       └── grader_ui.py     # Grading results display
├── app/                     # Core application logic
│   └── services/            # Business logic services
│       ├── gemini_client.py # Google Gemini API integration
│       ├── ocr_service.py   # OCR orchestration
│       ├── math_parser.py   # Mathematical expression parsing
│       ├── geometry_parser.py # Geometry problem solving
│       ├── grader.py        # Grading orchestration
│       ├── preprocess.py    # Image preprocessing
│       ├── layout_segmentation.py # Layout analysis
│       └── exporter.py      # Document export
├── tests/                   # Unit tests
├── docs/                    # Documentation
├── .streamlit/              # Streamlit configuration
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
└── README.md               # This file
```

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=app

# Run specific test file
pytest tests/test_math_parser.py -v
```

## 🌐 Deployment

### Streamlit Cloud
1. Fork this repository
2. Connect your GitHub account to [Streamlit Cloud](https://share.streamlit.io/)
3. Create a new app and select your forked repository
4. Add your Gemini API key in the Streamlit Cloud secrets manager
5. Deploy and share your app

For detailed deployment instructions, see `docs/deployment-streamlit-cloud.md`

### Local Network Deployment
```bash
streamlit run frontend/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

## 🔧 Configuration

### Environment Variables
```bash
GEMINI_API_KEY=your_gemini_api_key        # Optional: For enhanced OCR
STREAMLIT_PORT=8501                       # Optional: Custom port
DEBUG=false                               # Optional: Debug mode
```

### Streamlit Configuration
Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"

[server]
maxUploadSize = 10
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Run code formatting
black .

# Run linting
flake8 .
```

## 📝 Changelog

### v1.2.0 (Latest)
- ✨ Enhanced LaTeX parsing and conversion
- 🔧 Improved mathematical expression recognition
- 🐛 Fixed Streamlit deprecation warnings
- 📖 Better error handling and user feedback
- 🎨 Enhanced UI with better formatting

### v1.1.0
- 🚀 Added Gemini Vision API integration
- 📊 Improved grading accuracy
- 🔄 Added fallback OCR mechanisms

### v1.0.0
- 🎉 Initial release
- 📸 Basic OCR functionality
- 🧮 Symbolic math validation
- 📤 Word document export

## 🆘 Troubleshooting

### Common Issues

**Q: OCR extraction fails**
- Ensure Tesseract is properly installed and in PATH
- Check image quality and lighting
- Verify file format is supported (PNG, JPEG, JPG)

**Q: Gemini API errors**
- Verify your API key is correct
- Check your Google Cloud billing account
- Ensure you have sufficient API quota

**Q: Mathematical expressions not parsing**
- Review the extracted text for clarity
- Use the text editor to fix OCR errors
- Ensure mathematical notation follows standard conventions

**Q: Application won't start**
- Check Python version (3.10+ required)
- Verify all dependencies are installed
- Check for port conflicts

**Q: OpenCV/cv2 import errors on Streamlit Cloud**
- This is normal and handled automatically
- The app falls back to PIL-only image processing
- Ensure `packages.txt` includes system dependencies
- No action needed - the app should work fine

**Q: "libGL.so.1: cannot open shared object file" error**
- This occurs when deploying to cloud environments
- The app automatically handles this by disabling OpenCV features
- Image processing continues with PIL as fallback
- Performance may be slightly reduced but functionality remains

### Getting Help
- 📧 Create an issue on GitHub
- 📖 Check the documentation in `docs/`
- 🔍 Search existing issues for solutions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Gemini**: Advanced vision AI capabilities
- **SymPy**: Symbolic mathematics library
- **Streamlit**: Rapid web application framework
- **Tesseract**: Open-source OCR engine
- **OpenCV**: Computer vision library

---

**Built with ❤️ for mathematics education and automated grading.**
