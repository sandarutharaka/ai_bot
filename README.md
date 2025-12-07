# Ares - AI Bot

An advanced self-thinking AI assistant with autonomous learning capabilities, voice support, and persistent memory.

## Features

- **Self-Learning AI**: Autonomous system that monitors, learns, and evolves from interactions
- **Voice Support**: Speech recognition and text-to-speech capabilities
- **Persistent Memory**: SQLite knowledge store with conversation history
- **Multi-Model Support**: Integration with Google Gemini and OpenAI APIs
- **Rich CLI Interface**: Beautiful command-line experience with rich formatting
- **Modular Architecture**: Easy to extend and customize
- **Context-Aware**: Maintains conversation context for better responses

## Project Structure

```
AI bot/
├── main.py                 # Main autonomous learning system
├── ares_improved.py        # Improved Ares implementation with modular architecture
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- Windows, macOS, or Linux

### Optional System Requirements (for Voice)

- **Windows**: Windows 10 or higher
- **macOS**: macOS 10.12 or higher
- **Linux**: PulseAudio or ALSA for audio

## Installation

### 1. Clone or Download the Project

```bash
cd "c:\Users\sanda\Downloads\AI bot"
```

### 2. Create a Virtual Environment (Recommended)

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

**Dependencies included:**
- `google-generativeai` - Google Gemini API client
- `python-dotenv` - Environment variable management
- `rich` - Beautiful terminal output
- `openai` - OpenAI API client
- `pyttsx3` - Text-to-speech engine
- `SpeechRecognition` - Speech recognition library
- `pyaudio` - Audio I/O (optional, for voice)

### 4. Set Up Environment Variables

Create a `.env` file in the project root directory:

```bash
# .env file
GEMINI_API_KEY=your_google_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

**How to get API Keys:**

- **Google Gemini**: Visit [Google AI Studio](https://aistudio.google.com/apikey) and create an API key
- **OpenAI**: Visit [OpenAI Platform](https://platform.openai.com/api-keys) and create an API key

## Usage

### Running the Main Application

```bash
python main.py
```

### Running the Improved Ares Version

```bash
python ares_improved.py
```

## Features Explained

### Autonomous Learning System

The AI bot learns from every interaction:
- Stores conversation history and patterns
- Identifies recurring topics and preferences
- Evolves responses based on learned patterns
- Maintains persistent memory across sessions

### Voice Features

Enable voice interaction:
- **Speech Recognition**: Convert spoken words to text
- **Text-to-Speech**: Convert AI responses to spoken audio
- Automatic fallback if audio devices unavailable

### Rich CLI Interface

Enjoy an enhanced command-line experience:
- Colored text and panels
- Tables and formatted output
- Progress indicators
- Interactive prompts

## Configuration

### Memory Storage

- Conversation history: `ares_memory.json`
- Knowledge base: Automatically managed by the system
- SQLite database: Created automatically for ares_improved.py

### Customization

You can customize:
- AI model parameters
- Response styles
- Learning algorithms
- Voice settings

Edit the configuration sections in `main.py` or `ares_improved.py` as needed.

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'google.generativeai'`

**Solution**: 
```bash
pip install --upgrade google-generativeai
```

### Audio/Voice Not Working

**Problem**: Voice features not responding

**Solution**:
```bash
# Install audio dependencies
pip install pyttsx3 SpeechRecognition pyaudio
```

### API Key Issues

**Problem**: `APIError` or authentication failures

**Solution**:
1. Verify `.env` file exists in project root
2. Check API key is correct
3. Ensure API is enabled in your Google/OpenAI console
4. Test API key independently

### Environment Variable Not Loading

**Problem**: `.env` file not being read

**Solution**:
1. Ensure `.env` is in the project root directory
2. Verify filename is exactly `.env` (no extension)
3. Restart the application after creating `.env`

## Dependencies Details

| Package | Version | Purpose |
|---------|---------|---------|
| google-generativeai | Latest | Google Gemini API integration |
| openai | 1.54.0 | OpenAI API integration |
| python-dotenv | Latest | Environment variable management |
| rich | Latest | Terminal UI and formatting |
| pyttsx3 | Latest | Text-to-speech (offline) |
| SpeechRecognition | Latest | Speech-to-text |
| pyaudio | Latest | Audio I/O (optional) |

## Architecture

### Main Components

1. **AutonomousLearningSystem** - Core learning engine
2. **ExternalAI** - API wrapper for LLM providers
3. **VoiceManager** - Speech recognition and TTS
4. **KnowledgeBase** - Persistent memory storage
5. **CLI** - Command-line interface

### Data Flow

```
User Input → Speech Recognition → AI Processing → Learning System
                                        ↓
                                  External LLM
                                        ↓
                                Response Generation
                                        ↓
                                   Text-to-Speech
                                        ↓
                                  User Output
```

## Performance Tips

1. **Use Virtual Environment**: Keeps dependencies isolated
2. **Enable Caching**: Speeds up repeated queries
3. **Batch Requests**: Group API calls when possible
4. **Monitor Memory**: Check `ares_memory.json` size periodically

## Security Best Practices

1. **Never commit `.env`**: Add to `.gitignore`
2. **Use environment variables**: Don't hardcode API keys
3. **Rotate API keys regularly**: For production use
4. **Validate inputs**: Especially for custom extensions

## Contributing

To extend Ares:

1. Create new modules in the project
2. Follow the existing code structure
3. Add documentation for new features
4. Test thoroughly before deployment

## License

This project is provided as-is for personal and educational use.

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review the code comments and docstrings
3. Check API documentation for Google Gemini and OpenAI
4. Verify all dependencies are installed correctly

## Version History

- **v1.0**: Initial Autonomous Learning System (main.py)
- **v1.1**: Improved modular architecture (ares_improved.py)

---

**Last Updated**: December 2024

**System**: Windows PowerShell
