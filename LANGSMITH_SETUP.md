# LangSmith Setup Guide

LangSmith provides powerful debugging, testing, and monitoring capabilities for your LangChain application. This guide will help you set up and use LangSmith with your Employee Handbook Assistant.

## 🚀 Quick Start

### 1. Get a LangSmith API Key

1. Go to [smith.langchain.com](https://smith.langchain.com)
2. Sign up or log in with your account
3. Navigate to Settings → API Keys
4. Create a new API key
5. Copy the key (starts with `ls__`)

### 2. Configure Your Environment

Add these variables to your `.env` file:

```env
# LangSmith Configuration
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=ls__your_actual_api_key_here
LANGCHAIN_PROJECT=employee-handbook-assistant
```

### 3. Install Dependencies

The required package is already in `requirements.txt`:

```bash
pip install langsmith
```

Or if updating an existing installation:

```bash
pip install --upgrade langsmith
```

### 4. Run Your Application

Start the Flask app normally:

```bash
python app.py
```

You should see in the console:
```
[INIT] LangSmith tracing enabled for project: employee-handbook-assistant
```

## 📊 Viewing Traces in LangSmith

### Access Your Project Dashboard

1. Go to [smith.langchain.com](https://smith.langchain.com)
2. Click on "Projects" in the sidebar
3. Find your project: `employee-handbook-assistant`
4. Click to open the project dashboard

### What You'll See

For each user interaction, LangSmith captures:

- **Full conversation traces** - See the entire flow through your LangGraph
- **LLM calls** - Individual calls to GPT-4 with tokens, latency, and costs
- **Tool executions** - When `search_handbook_tool` and `draft_ticket_tool` run
- **State transitions** - How your graph state changes at each step
- **Chunk searches** - Tagged with `chunk_0`, `chunk_1`, etc. for debugging

### Key Metrics to Monitor

- **Latency** - How long each step takes
- **Token usage** - Input/output tokens per request
- **Success rate** - How often the handbook search finds answers
- **Cost tracking** - Estimated costs per query

## 🔍 Debugging with LangSmith

### Trace Details

Click on any trace to see:
- Input/output at each step
- Exact prompts sent to the LLM
- Which handbook chunks were searched
- Why a ticket was or wasn't offered

### Useful Filters

In the LangSmith UI, you can filter by:
- **Run name** - Each trace shows the user's question (first 50 chars)
- **Status** - Success, error, or pending
- **Metadata** - Environment (development/production)
- **Tags** - `search_handbook`, `chunk_0`, etc.

### Common Issues to Look For

1. **Long latencies** - Check which chunks are being searched
2. **NOT_FOUND responses** - Review the chunk content and prompts
3. **Incorrect work-related detection** - Check the work_check_prompt responses
4. **High token usage** - Consider reducing chunk sizes

## ⚙️ Advanced Configuration

### Custom Project Names

Change the project name in your `.env`:

```env
LANGCHAIN_PROJECT=my-custom-project-name
```

### Disable Tracing in Production

To disable tracing (e.g., to save costs):

```env
LANGCHAIN_TRACING_V2=false
# Or just remove/comment out LANGCHAIN_API_KEY
```

### Environment-Specific Projects

Use different projects for different environments:

```python
# In app.py
project_name = f"handbook-{os.getenv('FLASK_ENV', 'prod')}"
os.environ["LANGCHAIN_PROJECT"] = project_name
```

## 📈 Evaluation and Testing

### Create Test Datasets

In LangSmith, you can:
1. Select successful traces
2. Add them to a dataset
3. Use for regression testing

### Monitor Performance

Set up alerts for:
- Latency exceeding thresholds
- Error rates
- Token usage spikes

## 🛠️ Troubleshooting

### Tracing Not Working?

Check:
1. API key is valid (starts with `ls__`)
2. `LANGCHAIN_TRACING_V2=true` (not `True` or `TRUE`)
3. Network access to `api.smith.langchain.com`
4. Console shows "LangSmith tracing enabled"

### Missing Traces?

- Traces may take 5-10 seconds to appear
- Check your project name matches
- Verify API key permissions

### High Costs?

- LangSmith has a free tier (usually sufficient for development)
- Disable tracing in production if needed
- Use sampling for high-traffic applications

## 📚 Resources

- [LangSmith Documentation](https://docs.smith.langchain.com)
- [LangSmith Python SDK](https://github.com/langchain-ai/langsmith-sdk)
- [LangChain Discord](https://discord.gg/langchain) - Get help from the community

## 💡 Pro Tips

1. **Use Run Names**: The app automatically sets run names with the user's question
2. **Tag Important Flows**: Add custom tags for specific scenarios
3. **Compare Traces**: Use LangSmith's comparison view to debug differences
4. **Export Data**: Download traces as JSON for detailed analysis
5. **Share Traces**: Get shareable links for specific traces when debugging with teammates

---

With LangSmith configured, you now have full visibility into your Employee Handbook Assistant's behavior, making it easier to debug issues, optimize performance, and ensure quality responses!
