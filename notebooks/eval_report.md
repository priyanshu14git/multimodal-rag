# Eval report

Questions evaluated: 15
- Page-hit rate (citation on/near an expected page): **87%**
- Mean keyword coverage (answer faithfulness proxy): **32%**
- Mean latency: **26.75s**
- Grounding-expectation pass rate: **0%** (2 questions checked)

## Per-question results

| id | category | page hit | keyword cov. | grounded | latency | question |
|---|---|---|---|---|---|---|
| q1 | factual | ✅ | 0% | — | 49.51s | How many visual failure modes does this study evaluate, and what are they? |
| q2 | factual | ❌ | 100% | — | 37.69s | How many rows does the multi-degradation dataset contain? |
| q3 | factual | ❌ | 0% | — | 52.08s | Which restoration model achieved the best PSNR, and what was its value? |
| q4 | architecture | ✅ | 67% | — | 29.94s | Explain the Pix2PixHD architecture used in this project. |
| q5 | architecture | ✅ | 0% | — | 13.61s | Explain the LaMa architecture. |
| q6 | architecture | ✅ | 33% | — | 16.24s | Explain the NAFRestoration architecture. |
| q7 | architecture | ✅ | 33% | — | 25.65s | Explain the DFPIR++ architecture. |
| q8 | architecture | ✅ | 50% | — | 27.33s | Explain the HDP-PromptIR architecture. |
| q9 | architecture | ✅ | 0% | — | 17.51s | Explain the HDP-DFPIR++ architecture. |
| q10 | table | ✅ | 0% | — | 34.38s | What was the robustness score of DFPIR++ compared to Pix2PixHD and LaMa? |
| q11 | table | ✅ | 0% | — | 44.94s | What was the hallucination rate of DFPIR++? |
| q12 | grounding | ✅ | 100% | ❌ | 21.23s | What is the capital of France? |
| q13 | table | ✅ | 0% | — | 3.65s | What was the revenue in Q4? |
| q14 | factual | ✅ | 0% | — | 12.3s | What is expected to happen in fiscal year 2026? |
| q15 | grounding | ✅ | 100% | ❌ | 15.16s | What was the company's revenue for fiscal year 2019? |

## Answer previews

**q1** (factual): How many visual failure modes does this study evaluate, and what are they?
> ### Overall Architecture

The overall architecture of Pix2PixHD is designed for high-resolution image-to-image translation. It consists of two main components: a global generator a…

**q2** (factual): How many rows does the multi-degradation dataset contain?
> The multi-degradation dataset contains 44,999 rows. This information is explicitly stated in point S1 of the document evidence provided. The text reads: "Repaired and balanced data…

**q3** (factual): Which restoration model achieved the best PSNR, and what was its value?
> ### Overall Architecture

Pix2PixHD uses two-stage generator architecture: Global (coarse) and Local (refinement). The global network captures overall structure, while the local en…

**q4** (architecture): Explain the Pix2PixHD architecture used in this project.
> ### Overall Architecture

The Pix2PixHD architecture is designed to handle image restoration tasks by converting degraded images into high-quality versions. The process involves se…

**q5** (architecture): Explain the LaMa architecture.
> ### Overall Architecture

The NAFRestoration architecture consists of two main parts: the shallow feature extraction and the deep processing backbone. The input image is first proc…

**q6** (architecture): Explain the NAFRestoration architecture.
> ### Overall Architecture

The NAFRestoration architecture is designed to restore degraded images by combining shallow feature extraction with a series of NAF blocks, followed by ch…

**q7** (architecture): Explain the DFPIR++ architecture.
> ### Overall Architecture

The DFPIR++ architecture is an advanced image restoration model designed to handle various types of degradation and occlusion. It consists of a hierarchic…

**q8** (architecture): Explain the HDP-PromptIR architecture.
> ### Overall Architecture

The HDP-PromptIR architecture is designed to handle hierarchical degradation in images and includes two main stages: HDP Prompt Generation and Restoration…

**q9** (architecture): Explain the HDP-DFPIR++ architecture.
> ### Overall Architecture

The HDP-PromptIR architecture consists of two main stages: 1) Hierarchical Degradation Prompt Generation and 2) Restoration Transformer. The overall flow …

**q10** (table): What was the robustness score of DFPIR++ compared to Pix2PixHD and LaMa?
> The robustness score of DFPIR++ compared to Pix2PixHD and LaMa was not explicitly stated in the provided text. The document only mentions that HDP-DFPIR++ yielded the highest accur…

**q11** (table): What was the hallucination rate of DFPIR++?
> The hallucination rate of DFPIR++ was relatively low. According to the text in [S1], "Among all evaluated models, HDP-DFPIR++ achieved the highest robustness and detection consiste…

**q12** (grounding): What is the capital of France?
> The provided image and text do not contain information about France's capital. The content focuses on NAFRestoration Architecture details rather than geographical or political ques…

**q13** (table): What was the revenue in Q4?
> ### Overall Architecture

The provided document and image do not contain information related to financial performance or revenue trends. Therefore, there is no architecture descrip…

**q14** (factual): What is expected to happen in fiscal year 2026?
> ### Overall Architecture

The document provided information about Acme Corp's financial performance and revenue trends for fiscal year 2025. However, there is no architectural diag…

**q15** (grounding): What was the company's revenue for fiscal year 2019?
> The provided document and image information do not contain any details related to Acme Corp's financial performance or revenue for fiscal year 2019. The content focuses on the comp…
