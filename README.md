\section{Repository Overview}

This repository reproduces experiments that test whether large language models (LLMs) display gradient-descent-like \textit{in-context learning} on simple linear regression and classification tasks. Each prompt presents $N = 2d + 1$ training examples followed by a test query. The model’s prediction is compared against an analytical single-step gradient descent baseline (GD-1).

\subsection*{Folder Structure}
\begin{verbatim}
.
├── Classification_Model/    # Notebooks and scripts for linear classification
├── Regression_Model/        # Notebooks and scripts for linear regression
└── .DS_Store                # macOS metadata (can be ignored)
\end{verbatim}

Each subfolder contains a Jupyter notebook that can be executed end-to-end.

\section{Requirements}

\subsection*{Python Environment}
\begin{itemize}
\item Python 3.10–3.11
\item CUDA-capable GPU (optional; tested on NVIDIA GTX 1080 Ti)
\end{itemize}

\subsection*{Dependencies}
\begin{verbatim}
typing-extensions>=4.12.2
transformers>=4.46.0
tokenizers>=0.20.1
accelerate>=0.34.2
safetensors>=0.4.3
sentencepiece>=0.2.0
matplotlib>=3.8.0
scikit-learn>=1.3.0
openai>=1.40.0
packaging>=23.2
\end{verbatim}

Install with:
\begin{verbatim}
pip install -r requirements.txt
\end{verbatim}
or directly:
\begin{verbatim}
pip install "typing-extensions>=4.12.2" "transformers>=4.46.0" \
            "tokenizers>=0.20.1" "accelerate>=0.34.2" "safetensors>=0.4.3" \
            "sentencepiece>=0.2.0" "matplotlib>=3.8.0" \
            "scikit-learn>=1.3.0" "openai>=1.40.0" "packaging>=23.2"
\end{verbatim}

\subsection*{GPU Notes}
The code defaults to half-precision (fp16) inference for smaller Hugging Face models. Flash attention and bfloat16 are not required for a GTX 1080 Ti.

\section{Model Backends}

\subsection*{Option A: Hugging Face}
Set:
\begin{verbatim}
export LLM_BACKEND=hf
export HF_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
\end{verbatim}
Models between 1--3B parameters fit comfortably on a single 1080 Ti. Attention is forced to the \texttt{eager} mode for compatibility.

\subsection*{Option B: OpenAI (GPT-5)}
Set:
\begin{verbatim}
export LLM_BACKEND=openai
export OPENAI_API_KEY=sk-...
export GPT5_MODEL=gpt-5-thinking
\end{verbatim}
\noindent Notes:
\begin{itemize}
\item Use \texttt{max\_completion\_tokens} instead of \texttt{max\_tokens}.
\item Do not override temperature; GPT-5 uses fixed sampling parameters.
\item If responses are truncated, raise the token limit to 16--32.
\end{itemize}

\section{Execution Instructions}

\subsection*{Classification}
\begin{enumerate}
\item Open the notebook in \texttt{Classification\_Model/}.
\item Set the backend environment variables.
\item Execute all cells. The notebook will:
  \begin{itemize}
  \item Sample $N = 2d + 1$ labeled examples.
  \item Build textual prompts and query the LLM.
  \item Compute accuracy and $F_1$ scores for the LLM and GD-1.
  \item Plot performance as a function of input range $\alpha$.
  \end{itemize}
\end{enumerate}

\subsection*{Regression}
\begin{enumerate}
\item Open the notebook in \texttt{Regression\_Model/}.
\item Set the same backend environment variables.
\item Run all cells to generate regression tasks and compute mean squared error (MSE) against GD-1 baselines.
\end{enumerate}

\section{Key Parameters}

\begin{itemize}
\item $d$: feature dimension (default 10)
\item $N$: number of context examples, $N = 2d + 1$
\item $\alpha$: input range scale (default $\{0.5, 1.0, 1.5, 2.0\}$)
\item $\eta$: learning rate for GD-1 (default 1.0)
\item $n_{tasks}$: number of sampled problems per $\alpha$ (default 10, recommended 1000 for publication-scale experiments)
\end{itemize}

Example configuration:
\begin{verbatim}
D = 10
N = 2*D + 1
ALPHAS = [0.5, 1.0, 1.5, 2.0]
n_tasks = 1000
ETA = 1.0
\end{verbatim}

\section{What the Code Does}

\begin{itemize}
\item Generates synthetic linear tasks:
  \begin{align*}
  \text{Regression: } & y = W^\top x \\
  \text{Classification: } & y = \mathrm{sign}(W^\top x)
  \end{align*}
\item Prompts the LLM with $N$ labeled examples and one test query.
\item Computes LLM prediction $\hat{y}$ and compares it to GD-1:
  \begin{align*}
  W_1 &= \frac{\eta}{N} X^\top y, \\
  \hat{y}_{\text{GD-1}} &= 
  \begin{cases}
    W_1^\top x_{\text{test}}, & \text{regression}, \\
    \mathrm{sign}(W_1^\top x_{\text{test}}), & \text{classification}.
  \end{cases}
  \end{align*}
\item Reports accuracy, $F_1$, and MSE metrics across $\alpha$ sweeps.
\end{itemize}

\section{Replicating the Published Results}

To match the reference paper setup:
\begin{enumerate}
\item Set the backend (Hugging Face or OpenAI).
\item Configure:
\begin{verbatim}
n_tasks = 1000
D = 10
N = 2*D + 1
ALPHAS = [0.5, 1.0, 1.5, 2.0]
ETA = 1.0
\end{verbatim}
\item Execute all cells in both notebooks and export plots.
\end{enumerate}

Optional analyses include:
\begin{itemize}
\item Order sensitivity (context permutation)
\item Leave-one-out influence
\item 2D decision boundary visualization for $d=2$
\end{itemize}

\section{Troubleshooting}

\begin{itemize}
\item \textbf{OpenAI Parameter Errors:} Remove \texttt{temperature} and replace \texttt{max\_tokens} with \texttt{max\_completion\_tokens}.
\item \textbf{Short Responses:} Increase token limit to 16--32.
\item \textbf{Import Errors:} Update the following:
\begin{verbatim}
pip install -U "typing-extensions>=4.12.2" "transformers>=4.46.0" \
             "tokenizers>=0.20.1"
\end{verbatim}
\item \textbf{CUDA Memory:} Use fp16 inference and smaller models.
\end{itemize}

\section{Citation}

If this codebase contributes to your work, please cite both this repository and the original ICL paper that inspired it.

\section{License}
Add license details here (e.g., MIT License).

\section{Maintainer}
\textbf{Shou-Yue}\\
Notable commits:
\begin{itemize}
\item Updated training size (Classification Model)
\item First complete regression attempt
\item Initial classification setup
\end{itemize}
