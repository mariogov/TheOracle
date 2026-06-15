<div align="center">

<img src="assets/hero.jpg" alt="The Oracle — a luminous constellation-eye reading streams of Python code and splitting them into pass and fail" width="100%" />

# 🔮 The Oracle

### A machine learning to tell whether code is *true*.

**An open research program building toward domain super-intelligence for Python software engineering** — a deterministic predictor that judges any AI-generated code change **Pass** or **Fail** against reality, *before* you ever trust it.

![Mission](https://img.shields.io/badge/mission-Domain_Super--Intelligence-8b5cf6?style=for-the-badge)
![Focus](https://img.shields.io/badge/focus-Python_Software_Engineering-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Engine](https://img.shields.io/badge/engine-Rust-CE412B?style=for-the-badge&logo=rust&logoColor=white)
![Status](https://img.shields.io/badge/status-active_research-00b894?style=for-the-badge)
![License](https://img.shields.io/badge/license-PolyForm_Noncommercial-4c6ef5?style=for-the-badge)

**[The Mission](#-the-mission) · [Why It Matters](#-why-the-oracle-exists) · [What It Does](#-what-the-oracle-does) · [How It Works](#️-how-it-works) · [The Journey](#-the-journey-so-far) · [Roadmap](#️-the-road-to-super-intelligence)**

</div>

---

> **This repository is the public window into one mission: teaching a machine to predict, with certainty, whether AI-generated software actually works.**
> It is where you can watch — commit by commit, number by number — an AI agent climb toward **domain super-intelligence for Python software engineering**. Crack Python, and the same blueprint unlocks every other language. Unlock every language, and you unlock super-intelligence for the entire discipline of engineering. **That is the goal. This is the climb.**

---

## 🚀 The Mission

<img src="assets/mission-ascent.jpg" alt="A luminous staircase of tiers ascending through cosmic clouds toward a brilliant apex — the climb to super-intelligence" width="100%" />

Today, AI can *write* code. It cannot *know* whether that code is correct — and neither can you, until a human reads it or a test suite runs. That trust gap is the single biggest thing standing between "AI that suggests code" and "AI that ships engineering."

**The Oracle closes that gap.** It is a *binary reality predictor*: give it any change an AI agent proposes to a Python codebase, and it tells you whether that change will **pass or fail** against the ground truth of real, executed tests — and it tells you *why* when it predicts failure.

The mission is deliberately staged:

```
   Python software engineering   ─►   every programming language   ─►   engineering itself
   ── prove it once, rigorously ──     ── replicate the blueprint ──    ── domain super-intelligence ──
```

When the predictor is right often enough, reliably enough, on hard enough problems, a line is crossed: **reviewing AI-generated Python patches by hand becomes statistically unjustified.** That crossing is the first instance of domain super-intelligence — and the template for all the rest.

---

## 💡 Why The Oracle Exists

AI writes code faster than any human can possibly review it. But speed without trust is a liability:

- 🕳️ **AI hallucinates correctness.** It will confidently claim a fix works when it doesn't.
- 🧍 **Human review doesn't scale.** Reading every AI patch re-introduces the bottleneck AI was supposed to remove.
- 🎲 **"Looks right" isn't "is right."** Static inspection — by humans *or* models — repeatedly fails to catch behavioral bugs that only surface when the code actually runs.

The Oracle's answer is to stop guessing from how code *looks* and start predicting from what code *does*. Every verdict is anchored to a single, incorruptible source of truth: **the real test suite, executed in a real environment.** No vibes. No opinions. Just a falsifiable prediction against reality.

> **The promise:** *Ship AI-written Python with the confidence that a calibrated machine — not a tired reviewer at 4 PM — has already told you whether it works.*

---

## 🎯 What The Oracle Does

For any AI-generated code change, the Oracle answers four grounded questions — each one checked against reality, not asserted:

| | Question | What you get | Grounded against |
|---|----------|--------------|------------------|
| **Q1** | Does what the AI *claimed* it did actually **exist**? | Pass / Fail | The bytes & AST on disk |
| **Q2** | Does the change **work**? | **Pass / Fail** + confidence | The real Docker test oracle |
| **Q3** | **Why** would it fail? | A named failure mode + closest real examples | Attached to the Q2 verdict |
| **Q5** | How does it **impact reality**? | Predicted vs. observed change events | The live runtime shift log |

**Feature → benefit, in one line each:**

| The Oracle gives you… | …so that you can |
|---|---|
| A binary **Pass/Fail** verdict on AI code | trust a patch without reading every line |
| A **calibrated confidence** + abstention | know exactly when *not* to trust the prediction |
| A **named reason** on predicted failure | fix the real defect instead of hunting for it |
| Verdicts anchored to **executed tests** | stop shipping code that only *looks* correct |
| A learning loop that **converts mistakes into training** | watch the predictor get sharper over time |

---

## ⚙️ How It Works

<img src="assets/how-it-works.jpg" alt="Code fragments flow into a constellation-shaped intelligence core, which emits a single verdict splitting into green pass and red fail, anchored to a crystalline cube of reality" width="100%" />

The Oracle is built on **ME-JEPA-Code** — a Joint-Embedding Predictive Architecture for code. The pipeline is deliberately simple to state and hard to fool:

1. **Perceive.** An AI agent's edit — its `Edit`, `Write`, `Bash`, and test runs — *is* the data-generating process. The Oracle perceives the change as it happens.
2. **Encode.** The change is projected through a **panel** of distinct, frozen embedders — each a different "sense" for what code means (see *[The Panel](#-the-panel)* below).
3. **Predict.** A trained head emits a binary verdict with a conformal confidence interval and a teleological-constellation guard that rejects out-of-distribution inputs instead of bluffing.
4. **Ground.** The verdict is measured against the **Docker oracle** — `swebench.harness.run_evaluation` — the same real test execution that defines whether code truly works.
5. **Learn.** Every miss becomes weighted training signal through a protected mistake-loop, so the predictor's calibration improves with experience.

The discipline that makes it trustworthy: **fail closed on the unknown, never fake a pass, and read the source of truth back for every claim.**

---

## 🌌 The Panel

<img src="assets/constellation.jpg" alt="A teleological constellation — many distinct glowing star-clusters connected into one coherent shape" width="100%" />

The Oracle never collapses code into a single opaque vector. Instead it builds a **teleological constellation** — an array of per-embedder vectors, each living in its own space, each a different perspective on the same change. Meaning emerges from the *whole constellation*, the way a single star tells you nothing but a constellation tells you where you are.

A core research finding drives the whole program: **static embedders see what code *looks like*, not what it *does*.** Behavioral bugs hide in execution, not in syntax. So the frontier of this work is teaching the panel to perceive **execution** — coverage, value divergence, and real test behavior — which is exactly where the Oracle's accuracy has climbed the most.

---

## 🧭 The North Star

Everything in this repository traces to one falsifiable predicate. Nothing ships unless it moves a number on this sheet:

```text
THE SYSTEM WORKS  ⇔
    prediction_oracle_correlation ≥ 0.95
    STABLE across 4 consecutive rolling windows
    on the 300 × 8 SWE-bench Lite Python corpus
    using real Docker-oracle Pass/Fail as ground truth
    AND confirmed on a second independent panel within 0.05
```

The moment this evaluates **TRUE**, human review of AI-generated Python patches becomes statistically unjustified. **That is the finish line — and the birth of the first engineering super-intelligence.**

---

## 📈 The Journey So Far

<img src="assets/progress.jpg" alt="A glowing trajectory line climbing toward a golden goal-line near the top of a cosmic grid" width="100%" />

This is honest, in-progress research — and the climb is real. Predicting correctness from how code *looks* hits a hard ceiling. Teaching the Oracle to perceive what code *does* breaks through it:

| Stage of the climb | Oracle correlation |
|---|:---:|
| 📉 Static baseline (judging code by appearance) | ~0.48 |
| ➕ Execution-aware inputs (value capture) | ~0.46 → 0.54 |
| ➕ Coverage gating (ignore tests that never run the change) | ~0.54 |
| ➕ Real test-scope tracing (run what the oracle runs) | ~0.66 |
| 🚀 Combined execution-aware panel (best measured head) | **~0.74** |
| 🎯 **The target** | **0.95** |

*Measured on the 300 × 8 SWE-bench Lite Python corpus. These numbers move as the work advances — that's the point of tracking them in the open.*

Every one of those jumps is a logged experiment in this repo's issue history. **The gap from 0.74 to 0.95 is the work that remains** — and you can watch it close in real time.

---

## 🗺️ The Road to Super-Intelligence

```
  ✅  Build the binary reality predictor for Python        ← the engine exists
  🔬  Drive oracle correlation 0.48 → 0.74 → 0.95          ← we are here
  🐍  Make human review of Python patches unjustified      ← the first super-intelligence
  🌐  Replicate the blueprint across every language        ← generalize
  🏛️  Domain super-intelligence for engineering itself     ← the mission
```

---

## 🏗️ Inside the Repository

The Oracle is a Rust workspace. The trunk:

| Crate | Role |
|---|---|
| `context-graph-mejepa` | The predictor, compiler, heal scheduler, and evaluation — the trunk model |
| `context-graph-mejepa-embedders` | The panel: distinct embedder "senses" for code |
| `context-graph-mejepa-instruments` | Frozen teleological constellation (zero-trainable-parameter targets) |
| `context-graph-mejepa-corpus` | Mutation operators + the SWE-bench Docker-oracle bridge |
| `context-graph-mejepa-train` | Trainer, replay buffer, and the mistake-loop learner |
| `context-graph-mejepa-tct` | Teleological constellations + the out-of-distribution guard |
| `context-graph-mcp` | The product surface — an MCP server AI agents call over JSON-RPC |

> **Architecture invariant:** *slot identity is sacred.* The panel is an array of per-embedder vectors — never flattened into a single blob for comparison. Meaning lives in the constellation, not in any one star.

---

## 🔧 Build It

```bash
# Clone
git clone https://github.com/ChrisRoyse/TheOracle.git
cd TheOracle

# Build the trunk model and the MCP server
cargo build --release -p context-graph-mejepa
cargo build --release -p context-graph-mcp

# Run the test suite
cargo test
```

Requires a recent stable Rust toolchain (see `rust-toolchain.toml`).

---

## 🛰️ Follow The Progress

This repository **is** the progress tracker. The mission advances in the open:

- 📌 **Issues** are the live research log — every experiment, finding, and number is recorded there.
- 📊 **Commits** move specific numbers on the North Star sheet, or they don't ship.
- ⭐ **Star the repo** to follow the climb from 0.74 toward 0.95 — and the moment the first engineering super-intelligence is proven.

> *Built by [Chris Royse](https://github.com/ChrisRoyse). The Oracle is the work of an AI agent, supervised in the open, climbing toward something that has never existed before.*

---

## 📜 License

Released under the **[PolyForm Noncommercial License 1.0.0](LICENSE)**. Free for noncommercial use, research, and study.

---

<div align="center">

<img src="assets/footer.jpg" alt="A vast serene starfield with a distant oracle eye rising on the horizon" width="100%" />

### 🔮 *Teaching machines to know what is true — one verdict at a time.*

</div>
