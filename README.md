# 🚗 Multimodal Annotation Agent for Autonomous Driving

A multimodal intelligent annotation assistant based on **VLM** and **LangChain**. This tool can automatically convert annotators' natural language descriptions and scene images into standardized JSON annotation data that complies with Pydantic's strong constraints.



## ✨ Core Features

*   **Multimodal Understanding**：Integrates the Qwen-series large models to process both text descriptions and 
    driving scene images simultaneously.
*   **Strong Structured Output Validation**：Defines strict annotation schemas based on Pydantic (including weather, lanes, traffic signs, ego-vehicle behavior, etc.).
*   **Automated Workflow & Self-Correction**：Automatically performs scene analysis, retrieval of on-vehicle 
    detection/cloud pre-annotation results, VLM inference with thinking and self-correction, JSON output format 
    validation, and persistent storage of results.
*   **Few-shot Prompt Engineering**：Well-designed few-shot prompting that guides the model to precisely align with complex autonomous driving annotation standards through high-quality examples.
*   **Interactive Frontend**：Real-time chat interface built with Streamlit, supporting streaming output and result downloading.

## 📂 Project Structure
```text
├── agent/      # Core logics
│   ├── tools/  # Tools for Agent
│   │   ├── annotation_tools.py  # tools definitions
│   │   └── middleware.py     # middleware for tool usage and debug
│   └── annotation_agent.py
├── config/     
├── models/
├── prompts/    # System Prompt and few-shot examples
├── schemas/    # Pydantic data schemas
├── utils/
├── run_app.py  # run streamlit
├── requirements.txt/
└── README.md
```

