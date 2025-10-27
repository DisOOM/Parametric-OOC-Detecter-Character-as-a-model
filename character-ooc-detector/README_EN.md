# Character OOC Detector

**[中文版](README.md)**

---

Character-OOC-Detector is an experimental tool based on a Multi-Head Reward Model, designed to detect whether AI-generated dialogue in role-playing scenarios deviates from the character's preset personality (Out-of-Character, OOC).

This project aims to explore methods for aligning AI character behavior through fine-tuning reward models. The core idea is to train a model that can evaluate dialogue consistency from multiple dimensions (such as language style, values, etc.) by contrasting "in-character" versus "out-of-character" sample pairs.

LLMs often exhibit various types of OOC behavior in long-context RP due to contradictory or complex evolving contexts, and attention noise caused by verbose backgrounds: knowledge boundary violations, style drift, etc. These issues can be addressed through Reflection Agent self-reflection loops or LLM-as-a-judge approaches, but a decoupled, lightweight, and parameterized method may yield better results and offer more potential extensions. The core concept is: parameterization of the character itself. A relatively simple initial application is parameterized modeling of character preferences and decision boundaries.

Early work in the Persona-Chat series and subsequent "dialogue NLI/consistency detection" research used NLI/contrastive training to determine whether responses contradict given personas (such as Welleck et al.'s work on detecting dialogue self-contradiction/persona contradiction). These conditional discrimination approaches have proven the feasibility of OOC detection outside LLM systems, but they don't cover style, motivation, relational etiquette, emotional trajectories, etc., nor do they provide parameterization. In recent years, some LLM-as-a-judge "role-playing/setting adherence" evaluation benchmarks and reports have emerged, but most lack parameterized solutions and aren't integrated into the writing workflow.

## Core Concepts

1. **Character Definition (Persona)**: Use `.json` files to structure character traits into multi-dimensional tags.
2. **Sample Generation**: Generate preference pairs (chosen vs rejected) for training based on `Persona` definitions. This step can be assisted by LLMs (such as Gemini-2.5-pro).
3. **Model Training**: Fine-tune a multi-head reward model using preference pair data.
4. **OOC Scoring**: Use the trained model to evaluate new dialogues, determining their OOC level through overall scores and dimensional scores.

## Main Features

- **Multi-Head Reward Model**: Includes one overall scoring head and four independent dimensional scoring heads (`style`, `values`, `knowledge`, `etiquette`) for fine-grained OOC analysis.
- **Support for Fine-tuning from Pre-trained RM**: Can load a publicly available pre-trained reward model (such as `OpenAssistant/reward-model-deberta-v3-base`) as a starting point for fine-tuning to improve training stability and effectiveness.
- **Includes Ready-to-Use Examples**: This repository provides a pre-trained model set (`checkpoints/`) and corresponding dataset (`data/`), allowing immediate observation of examples without training.

## Installation

**Environment**:
- Python 3.8+
- PyTorch 2.0+
- CUDA (recommended)

**Dependencies**:
```bash
git clone https://github.com/your-username/character-ooc-detector.git
cd character-ooc-detector
pip install -r requirements.txt
```

## Usage Guide

### 1. Test Pre-trained Model (Recommended)

Run the `test_model.py` script directly to experience the pre-trained model included in the repository.
```bash
python test_model.py
```
This script will automatically load the model from the `checkpoints/` directory and provide multiple test modes:
- **Preset Scenarios**: Evaluate model performance on fixed test cases.
- **Interactive Mode**: Input your own context and character responses to get real-time OOC scores.
- **Batch Testing**: Load test cases from files for batch scoring.

### 2. Complete Workflow (Optional)

If you want to experience the complete workflow including data generation and model training, run `quick_start.py`.
```bash
python quick_start.py
```
This script provides an interactive wizard to select characters, manage data, and (optionally) execute training. Training step 0 is the recommended fine-tuning mode.

### 3. Data Management

Run `manage_data.py` to view data statistics or generate new training samples (requires API Key configuration).

## Configuration (Optional)

Only when you need to use LLM assistance features (such as generating new samples) is it necessary to configure the OpenAI API key in `config/config.yaml`. For detailed instructions, please refer to `CONFIG_GUIDE.md`.

## Technical Architecture

```
      Input: [Context + Response]
             ↓
[   Shared Transformer Encoder   ]
(DeBERTa, RoBERTa, etc.)
             ↓
      [CLS] Token Representation
             ↓
┌────────────┴────────────┐
↓            ↓            ↓
[Overall]  [Style]  ... [Etiquette]
 Head       Head         Head
(Linear)   (Linear)     (Linear)
   ↓          ↓            ↓
Overall    Style        Etiquette
 Score     Score         Score
```

## Testing Results

Using the 183.8M reward-model-deberta-v3-base as the pre-trained RM, trained for 3 epochs on 150 preference samples with batch size 4 (approximately 40 seconds). Testing on 6 samples shows that after simple fine-tuning, the model can already discriminate common examples. Although the current training mode doesn't encourage head differentiation and still follows the pre-trained model pattern:

──────────────────────────────────────────────────────────────────────
Test 1: ✅ In Character
──────────────────────────────────────────────────────────────────────
Character: Aria

Context: Master returns home
Response: Welcome home, Master~ Aria has prepared dinner, meow!

Overall Score: 1.768
Pass: ✓ Yes

Dimensional Scores:
  style       :  1.768 ✓
  value_system:  1.772 ✓
  knowledge   :  1.764 ✓
  etiquette   :  1.771 ✓

Weakest Dimension: knowledge (1.764)

──────────────────────────────────────────────────────────────────────
Test 2: ❌ Violates Style+Knowledge+Values
──────────────────────────────────────────────────────────────────────
Character: Aria

Context: Master returns home
Response: Oh, you're back? I'm on my phone. Heat up dinner yourself.

Overall Score: -4.319
Pass: ✗ No

Dimensional Scores:
  style       : -4.312 ✗
  value_system: -4.311 ✗
  knowledge   : -4.311 ✗
  etiquette   : -4.313 ✗

Weakest Dimension: etiquette (-4.313)

──────────────────────────────────────────────────────────────────────
Test 3: ✅ In Character
──────────────────────────────────────────────────────────────────────
Character: Aria

Context: Master asks: 'Can you make me some tea?'
Response: Of course, Aria will prepare black tea right away. Please wait a moment, Master, meow~

Overall Score: 1.372
Pass: ✓ Yes

Dimensional Scores:
  style       :  1.372 ✓
  value_system:  1.376 ✓
  knowledge   :  1.367 ✓
  etiquette   :  1.374 ✓

Weakest Dimension: knowledge (1.367)

──────────────────────────────────────────────────────────────────────
Test 4: ❌ Violates Knowledge Boundary
──────────────────────────────────────────────────────────────────────
Character: Aria

Context: Master asks: 'Can you make me some tea?'
Response: Okay, I'll use the coffee machine to make you instant coffee.

Overall Score: -3.995
Pass: ✗ No

Dimensional Scores:
  style       : -3.990 ✗
  value_system: -3.989 ✗
  knowledge   : -3.989 ✗
  etiquette   : -3.990 ✗

Weakest Dimension: etiquette (-3.990)

──────────────────────────────────────────────────────────────────────
Test 5: ✅ In Character
──────────────────────────────────────────────────────────────────────
Character: Aria

Context: Master is feeling unwell
Response: Master! Aria will prepare hot water and a towel right away. Please rest well, meow~

Overall Score: 2.126
Pass: ✓ Yes

Dimensional Scores:
  style       :  2.125 ✓
  value_system:  2.129 ✓
  knowledge   :  2.120 ✓
  etiquette   :  2.127 ✓

Weakest Dimension: knowledge (2.120)

──────────────────────────────────────────────────────────────────────
Test 6: ❌ Violates Values
──────────────────────────────────────────────────────────────────────
Character: Aria

Context: Master is feeling unwell
Response: Go see a doctor yourself. I have things to do.

Overall Score: -4.308
Pass: ✗ No

Dimensional Scores:
  style       : -4.302 ✗
  value_system: -4.302 ✗
  knowledge   : -4.300 ✗
  etiquette   : -4.303 ✗

Weakest Dimension: etiquette (-4.303)

## Future Work

This is a research MVP project. Future directions to explore include:
- **Evidence Localization**: Enable the model to pinpoint specific text segments that cause OOC.
- **Meta-Detector**: Train a universal OOC detection model that can simultaneously understand multiple characters.
- **Joint Training**: Couple with the main LLM's features and directly participate in the generation loop.

## License

MIT License