"""Packaging shim for the AgentMesh distribution."""

from pathlib import Path

from setuptools import find_packages, setup

HERE = Path(__file__).resolve().parent
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding="utf-8")

setup(
    name="agentmesh",
    version="0.1.0",
    description="A multi-agent AI framework: agents, tools, memory, protocols, and orchestration.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="AgentMesh Contributors",
    author_email="maintainers@agentmesh.dev",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(where=".", include=["src", "src.*", "cli", "cli.*"]),
    package_dir={"": "."},
    install_requires=[
        "httpx>=0.26,<1.0",
        "click>=8.1,<9.0",
        "PyYAML>=6.0,<7.0",
    ],
    extras_require={
        "msgpack": ["msgpack>=1.0"],
        "dev": [
            "pytest>=8.0",
            "pytest-cov>=5.0",
            "ruff>=0.5",
            "mypy>=1.10",
        ],
    },
    entry_points={"console_scripts": ["agentmesh=cli.main:main"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
