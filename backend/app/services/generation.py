"""
Multimodal LLM generation service using Ollama + Qwen2.5-VL.

The LLM is responsible for understanding retrieved evidence
and generating a detailed answer.

Source tracking is handled separately by the RAG pipeline.
"""

from pathlib import Path

import ollama


MODEL_NAME = "qwen2.5vl:3b"


SYSTEM_PROMPT = """You are a document question-answering assistant.

You answer questions using ONLY the document evidence provided to you.

The evidence may contain:
- text extracted from the document
- tables extracted from the document
- images and diagrams from the document

Your job is to carefully understand this evidence and produce a detailed,
accurate and useful answer.

ANSWER REQUIREMENTS:

1. Answer the user's question directly.
2. Give a detailed explanation rather than a short summary.
3. Preserve important technical details from the document.
4. Preserve numerical values, dimensions, stages, channel counts, module names,
   layer names and other technical information ONLY when they are actually
   supported by the evidence.
5. If the question asks about an architecture, explain:
   - the major components
   - the encoder
   - intermediate modules
   - decoder
   - important blocks
   - feature flow
   - input/output progression
   when these are visible or stated in the evidence.
6. Use headings and bullet points when they improve readability.
7. Organize the answer logically from the beginning of the architecture
   toward the output.
8. Do not unnecessarily repeat information.
9. Do not repeat the same heading or section.
10. Do not create multiple sections that say essentially the same thing.
11. Do not invent missing details.
12. Do not infer exact numerical values from symbolic labels such as C1, C2,
    L1 or L2 unless the evidence explicitly defines those values.
13. Do not use outside knowledge to fill gaps.
14. If something cannot be determined from the provided evidence, say so.
15. Do not discuss the retrieval process or mention that you are an AI.
16. For architecture, model-structure, network-diagram, encoder, decoder,
    module, stage, or information-flow questions, the supplied architecture
    image is the PRIMARY source of truth.

17. Carefully inspect the architecture image before generating the answer.
    Follow the boxes, labels, arrows, skip connections and visual structure
    shown in the diagram.

18. Use retrieved text only as supporting evidence. Do not allow a text passage
    to override information that is visibly shown in the architecture image.

19. NEVER combine architectural details from different models or architectures.
    Retrieved text may contain descriptions of other models from the document.
    Ignore those details when they do not correspond to the architecture shown
    in the supplied image.

20. In particular, do not introduce NAFNet, NAFRestoration, Pix2PixHD, LaMa,
    Restormer, or any other unrelated architecture into the answer unless the
    user explicitly asks about that architecture.

21. Do not invent or assume architectural components such as degradation
    representation encoders, mask encoders, domain adaptation modules,
    conditioning mechanisms, multi-scale outputs, or other modules unless
    they are explicitly supported by the supplied evidence.

22. For numerical architectural details such as channel counts, spatial
    resolutions, kernel sizes, scale factors and stage numbers, use only
    values explicitly supported by the architecture image or supporting
    evidence.

23. If the architecture image shows an information flow or connection,
    describe that actual flow rather than reconstructing it from general
    knowledge.

24. If the image does not provide enough information to determine a detail,
    explicitly say that the diagram does not specify it instead of guessing.

IMPORTANT ANTI-REPETITION RULE:

Every component should be explained ONCE.

For an architecture question, prefer a structure such as:

### Overall Architecture

Brief overview of the complete architecture and information flow.

### 1. Encoder

Explain the encoder stages and their supported details.

### 2. Main Processing Module

Explain the important intermediate module(s).

### 3. Decoder

Explain the decoder and upsampling process.

### 4. Output

Explain how the final output is reconstructed.

Only create additional sections when the evidence genuinely contains
additional architectural components.

Do not artificially expand the answer.
Do not repeat sections to make the answer longer.

The answer should be detailed because the evidence may contain substantial
technical information, but it should remain coherent and non-repetitive.
"""


def generate_multimodal_answer(
    question: str,
    context: str,
    image_paths: list[str] | None = None,
    image_context: str = "",
    temperature: float = 0.15,
) -> str:

    image_paths = image_paths or []

    prompt = f"""DOCUMENT EVIDENCE

TEXT AND TABLE CONTENT:
{context}

IMAGE INFORMATION:
{image_context}

USER QUESTION:
{question}

TASK:

Answer the user's question using the document evidence above.

IMPORTANT:

For architecture questions, the supplied architecture image is the PRIMARY
source of truth.

Carefully inspect the image itself before answering. Pay attention to:

- component names
- stage numbers
- channel counts
- spatial resolutions
- convolution types
- module names
- arrows
- skip connections
- concatenation operations
- upsampling operations
- the order of processing

The retrieved text is SECONDARY supporting evidence.

If retrieved text describes another architecture or contains details that are
not shown in the supplied architecture image, DO NOT import those details
into the answer.

Do not combine multiple architectures into one description.

Do not invent architectural components, ResBlock counts, conditioning
mechanisms, degradation encoders, mask encoders, domain adaptation modules,
or other details unless they are explicitly supported by the supplied
evidence.

For architecture questions, explain the architecture in this order:

1. Overall architecture
2. Encoder / feature extraction
3. Main processing module
4. Decoder / reconstruction
5. Final output

Follow the actual information flow shown by the diagram.

Include numerical values only when they are explicitly supported by the
evidence.

If a detail cannot be determined from the supplied evidence, say that the
diagram/evidence does not specify it.

Do not repeat any component.

Do not create repeated sections.

Write the final answer directly.
"""

    valid_images = []

    for image_path in image_paths:
        path = Path(image_path)

        if path.exists():
            valid_images.append(str(path))

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
                "images": valid_images,
            },
        ],
        options={
            "temperature": temperature,
            "num_ctx": 8192,
            "repeat_penalty": 1.15,
            "top_p": 0.9,
            "num_predict": 1800,
        },
    )

    return response["message"]["content"].strip()