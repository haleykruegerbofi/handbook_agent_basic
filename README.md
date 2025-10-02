# Axos Employee Handbook Assistant

A LangChain/LangGraph-powered chatbot that helps employees search the Axos employee handbook and submit help desk tickets for questions not found in the documentation.

## Features

- **PDF Handbook Search**: Loads and searches through the employee handbook PDF using natural language queries
- **Smart Chunking**: Automatically chunks large PDFs (41k+ tokens) into manageable segments for processing
- **Help Desk Integration**: Drafts and submits help desk tickets for work-related questions not found in the handbook
- **Modern Chat Interface**: Clean, responsive web UI with real-time chat functionality
- **State Management**: Uses LangGraph for intelligent conversation flow and decision-making

## Architecture

This application uses:
- **Flask** for the web server
- **LangChain/LangGraph** for the AI agent orchestration
- **OpenAI GPT-4** for natural language understanding
- **PyPDF** for PDF text extraction
- **TikToken** for token counting and chunking

The agent workflow:
1. Receives user question
2. Searches handbook chunks for relevant information
3. If found: Returns answer from handbook
4. If not found and work-related: Offers to create help desk ticket
5. If not work-related: Politely declines to help

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure OpenAI API Key

Create a `.env` file in the project root (copy from `env_template.txt`):

```
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Add Employee Handbook PDF

By default, the app loads the handbook from a hardcoded path in `app.py`:

```python
HANDBOOK_PATH = r"employee_handbook.pdf"
```

Update `HANDBOOK_PATH` to your PDF's full path, or place `employee_handbook.pdf` next to `app.py`.

### 4. Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

## Usage

1. **Ask Questions**:
   - Type questions about company policies, benefits, procedures, etc.
   - The assistant will search through the handbook and provide answers

2. **Submit Help Desk Tickets**:
   - If your work-related question isn't in the handbook, the assistant will offer to create a ticket
   - Review the drafted ticket details (subject, description, priority)
   - Click "Submit Ticket" to send it to the help desk

## Project Structure

```
handbook_agent_basic/
├── app.py                  # Main Flask application and LangGraph agent
├── templates/
│   └── index.html         # Frontend chat interface
├── employee_handbook.pdf  # Your employee handbook (add this)
├── requirements.txt       # Python dependencies
├── env_template.txt       # Environment variable template
├── .env                   # Your environment variables (create this)
└── README.md             # This file
```

## Key Components

### LangGraph State
- `question`: User's input question
- `answer`: Generated response
- `found_in_handbook`: Boolean flag for handbook search result
- `is_work_related`: Boolean flag for work-related questions
- `ticket_drafted`: Boolean flag for ticket creation
- `ticket_content`: Dictionary containing ticket details

### Tools
1. **search_handbook_tool**: Searches through handbook chunks for answers
2. **draft_ticket_tool**: Creates help desk tickets for unanswered work questions

### API Endpoints
- `GET /`: Main chat interface
- `POST /chat`: Process user messages and agent responses
- `GET /status`: Check handbook loading status

## Token Management

The application handles large PDFs by:
- Splitting text into ~8000 token chunks
- Processing chunks sequentially until an answer is found
- Using tiktoken for accurate GPT-4 token counting

## Customization

### Adjust Chunk Size
In `app.py`, modify the `chunk_text` function:
```python
HANDBOOK_CHUNKS = chunk_text(content, max_tokens=8000)  # Adjust max_tokens as needed
```

### Change Model
Update the LLM initialization:
```python
llm = ChatOpenAI(
    temperature=0,
    model="gpt-4o-mini",  # Can use "gpt-4", "gpt-3.5-turbo", etc.
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
```

### Modify Ticket System
The ticket submission in `/chat` endpoint can be modified to integrate with your actual ticketing system.

## Troubleshooting

1. **"Handbook not loaded" error**: Ensure `employee_handbook.pdf` exists in the project root
2. **OpenAI API errors**: Verify your API key is set correctly in `.env`
3. **Memory issues with large PDFs**: Reduce chunk size or implement pagination
4. **Slow responses**: Consider using a faster model like `gpt-3.5-turbo`

## Security Notes

- Never commit `.env` file to version control
- Implement proper authentication before deploying to production
- Consider rate limiting for API calls
- Sanitize user inputs before processing

## License

This project is for internal Axos use only.
