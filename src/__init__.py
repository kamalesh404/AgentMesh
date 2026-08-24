"""AgentMesh: a multi-agent AI framework.

Subpackages:
    agents        Agent base classes and specialized agent implementations.
    tools         Schema-validated tools (search, code exec, files, shell...).
    memory        Short-term, long-term, episodic, and semantic memory.
    orchestration Graphs, supervision, debate, and pipeline coordination.
    protocols     Message format, capability handshakes, and consensus.
    providers     LLM backends for OpenAI, Anthropic, Ollama, and offline use.
    utils         Logging, retry, configuration, and serialization helpers.
"""

__title__ = "agentmesh"
__version__ = "0.1.0"
__author__ = "AgentMesh Contributors"
__license__ = "MIT"

__all__ = [
    "__title__",
    "__version__",
    "__author__",
    "__license__",
]
