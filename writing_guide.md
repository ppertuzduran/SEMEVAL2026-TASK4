# SemEval System Paper — LLM Writing Guide

You are helping write an academic system paper for a SemEval shared task workshop, following the ACL paper format. You already know the code, experiments, and results from the notebook. Use this guide to write each section. Write in formal, precise academic English. Do not be verbose — SemEval system papers are short (4–8 pages total).

---

## PAPER HEADER

**Write:**
- **Title** following the exact convention: `[TeamName] at SemEval-[Year] Task [N]: [Brief description of your approach]`
- **Authors** block: full names, institution, city, country, email addresses
- This goes at the very top of the [.tex](file:///c:/Users/pertu/OneDrive/Documentos/MASTER%20DEGREE/SEMESTER-4-202601/SEMEVAL2026/paper/v1/latex/acl_latex.tex) file in `\title{}` and `\author{}`

---

## SECTION 1: Abstract (~150–200 words)

**Purpose:** A standalone summary. The reader must understand the full paper from this alone.

**Write in this order:**
1. One sentence identifying the task and the shared task name.
2. One sentence on what the task requires (input/output, number of subtasks).
3. Two to three sentences describing your system(s): what model(s) you used, how they work at a high level, and what distinguishes them.
4. One sentence with your best official result: metric, subtask, and rank.

**Constraints:** No citations. No equations. No figures. Must be self-contained.

---

## SECTION 2: Introduction (~300–400 words)

**Purpose:** Motivates the problem, introduces the task, and states your contribution.

**Write in this order:**
1. **Motivation paragraph** (~100 words): Why is this problem real and hard? Ground it in a concrete observation or statistic. Cite 1–2 foundational or relevant papers.
2. **Task paragraph** (~100 words): Introduce the SemEval task. What is the input? What are the outputs? How many subtasks? What dataset is used?
3. **Contribution paragraph** (~100 words): What did you build? How many systems? What was the key result? One sentence with your final rank(s).
4. **Code line** (1 sentence): State that code is publicly available and provide the URL.

---

## SECTION 3: Background (~350–450 words)

**Purpose:** Describes the dataset and situates your work in prior research.

**Write in this order:**
1. **Dataset paragraph** (~150 words): Describe the task dataset — who created it, how many instances (train/test), what the annotation process was, what inter-annotator agreement was, and what the label schema is (how many classes, what levels of hierarchy).
2. **Class distribution** (reference a table): Include a LaTeX table showing label counts per split (train vs. test). Mention notable class imbalance if present.
3. **Related work paragraph** (~150–200 words): Review 3–5 relevant prior works. For each, state what they did and what result they got. End with one sentence explaining what gap your approach addresses.

---

## SECTION 4: System(s) Overview (~400–600 words)

**Purpose:** The main technical section. Explains your architecture(s). One `\subsection` per system.

**Write for each system:**
1. **Backbone** — What pretrained model did you use? Why?
2. **Input encoding** — How is the input formatted? (e.g., `[CLS] text_a [SEP] text_b [SEP]`, or independent passes)
3. **Model head** — What is placed on top of the encoder? (linear layer, MLP, etc.)
4. **Special design choices** — Any interaction features, custom pooling, multi-task setup, etc.
5. **Reference a figure** — Each system should have a pipeline diagram figure (`\begin{figure*}`)

**Also include (shared across systems):**
- How you handle class imbalance (e.g., class-weighted loss, formula for weights)
- Whether you use label smoothing, early stopping, etc.
- A comparison table if you have 2+ systems (rows = architectural components, columns = system names)

---

## SECTION 5: Experimental Setup (~200–300 words)

**Purpose:** Makes your experiments reproducible. Describes hyperparameters and hardware.

**Write:**
1. **Optimizer details**: name, learning rate, LR schedule type, warmup ratio, weight decay, gradient clipping
2. **Training details**: effective batch size, gradient accumulation if used, max epochs, label smoothing value
3. **Model selection**: metric used for early stopping, patience value
4. **Hardware**: GPU model, VRAM, CPU, RAM, precision (fp16/bf16/fp32)
5. **A hyperparameter table**: two-column table (Hyperparameter | Value) listing all of the above

---

## SECTION 6: Results (~400–600 words)

**Purpose:** Reports and interprets all results. Most data-dense section of the paper.

**Write in this order:**
1. **Main results paragraph** (~100 words): Compare all systems on all subtasks (Macro F1, Weighted F1). State which system won and why you think so.
2. **Main results table**: rows = systems, columns = subtask × metric. One table covering all systems.
3. **Per-class analysis paragraph (best system)** (~150 words): Describe which classes the best system handled well and which it struggled with. Hypothesize why based on class size or linguistic ambiguity.
4. **Per-class table (best system)**: Precision, Recall, F1 per class.
5. **Official leaderboard paragraph** (~100 words): State official test set results and rank on each subtask. Compare to other participating systems if known.
6. **Official results table**: Task | Macro F1 | Rank
7. **Error analysis paragraph** (~150 words): Analyze the confusion matrix. What gets confused with what? What does this reveal about the difficulty of the task?
8. **Confusion matrix figures** (one per subtask, if available)

---

## SECTION 7: Conclusion (~200–300 words)

**Purpose:** Wraps up the paper. Summarizes findings and proposes future work.

**Write in this order:**
1. **Summary** (~100 words): Restate the task, your approach, and which system performed best. State the key reason why it outperformed the other (one-sentence technical insight).
2. **Error insight** (~75 words): Summarize the main failure mode revealed by error analysis (e.g., which classes were confused, what this implies about the task difficulty).
3. **Future work** (~75 words): 2–3 concrete, specific future directions (e.g., ensembling, data augmentation targeting minority classes, different architectures).

---

## SECTION 8: Acknowledgments (1–2 sentences)

**Purpose:** Credits funding or institutional support. Required if the work was supported by a grant or scholarship.

**Write:** A single short paragraph crediting the relevant scholarship program, grant, or institution.

---

## BIBLIOGRAPHY

- Add all cited works to [references.bib](file:///c:/Users/pertu/OneDrive/Documentos/MASTER%20DEGREE/SEMESTER-4-202601/SEMEVAL2026/paper/v1/latex/references.bib) in BibTeX format.
- Cite using `\cite{key}` in the text.
- The bibliography does NOT count toward the page limit.
- Do not include works you do not actually cite in the paper.

---

## FIGURES AND TABLES — Formatting Notes

| Element | LaTeX environment | Notes |
|---|---|---|
| Narrow figure (1 column) | `\begin{figure}` | For confusion matrices |
| Wide figure (full width) | `\begin{figure*}` | For pipeline diagrams |
| Narrow table (1 column) | `\begin{table}` | For hyperparameters, per-class results |
| Wide table (full width) | `\begin{table*}` | For architectural comparison |
| All figures/tables | Must have `\caption{}` and `\label{}` | Referenced in text with `\ref{}` |

---

## WRITING STYLE RULES

- Write in **third person** ("We propose...", "The model achieves...")
- Use **past tense** for experiments ("We trained...", "The system achieved...")
- Use **present tense** for describing the method as it currently exists ("The classifier takes as input...")
- Every claim about performance must be supported by a table or citation
- Every figure and table must be explicitly referenced in the text before it appears
- Avoid filler phrases like "In this paper, we..." at the start of every paragraph
