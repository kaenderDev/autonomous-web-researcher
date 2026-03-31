
# ResearchAgent Pro 🤖🔍

**ResearchAgent Pro** is an autonomous AI agent designed to transform a single search query into a comprehensive, structured research report. Built using **Claude Code**, this project showcases advanced LLM orchestration, ethical web scraping, and a highly decoupled software architecture.

## 🌟 Key Features

- **Autonomous Multi-Step Research:** The agent breaks down a topic, searches for multiple sources, and aggregates findings.
- **Clean Architecture:** Strict separation between the core research logic and external providers (Search APIs, LLMs).
- **Intelligent Synthesis:** Uses advanced NLP to remove noise, extract relevant facts, and generate Markdown reports.
- **Pluggable Providers:** Easily swap between Google Search, Serper, or Bing via the Strategy Pattern.

---

## 🏗️ Architectural Blueprint

The project follows a **Modular Layered Architecture** to demonstrate high-level software engineering maturity:

- **`Core/`**: Domain logic for research orchestration and report synthesis.
- **`Providers/`**: Adapters for external APIs (Search Engines, LLM Clients).
- **`Scrapers/`**: Resilient web crawling logic with error handling and content cleaning.
- **`Output/`**: Formatter for Markdown, PDF, or JSON exports.

### Design Patterns Applied:
- **Strategy Pattern:** For interchangeable search and LLM engines.
- **Dependency Injection:** To ensure testability and loose coupling.
- **Chain of Responsibility:** For the data pipeline (Search -> Extract -> Refine -> Report).

---

## 🛠️ Built with Claude Code

This project was architected and developed using **Claude Code**, demonstrating the power of AI-agentic workflows in:
1. **Scaffold Engineering:** Automated generation of complex directory structures.
2. **Interface-Driven Development:** Defining rigid contracts before implementation.
3. **Robust Error Handling:** AI-generated retry logic for unstable web sources.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- API Keys for: Anthropic (Claude), Serper/Google Search.

### Installation
1. **Clone and Setup:**
   ```bash
   git clone [https://github.com/your-username/research-agent-pro.git](https://github.com/your-username/research-agent-pro.git)
   cd research-agent-pro
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Configure Environment:**
Create a `.env` file:
 ```Snippet de código 
 ANTHROPIC_API_KEY=your_key SEARCH_API_KEY=your_key
 ```
 3. **Run the Agent:**
 ```bash 
 python main.py --topic "The future of Rust in systems programming"
 ```
 
## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

----------

_Developed as a showcase of Senior Software Architecture & AI Collaboration._
