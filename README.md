# VLM Prior Bias Probe 🔍

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gemini 1.5](https://img.shields.io/badge/Gemini-1.5%20Pro-green.svg)](https://ai.google.dev/gemini-api)

> A reproducible probe for measuring when Vision-Language Models trust world priors over visual evidence

## 🎯 Overview

Vision-Language Models (VLMs) can rely on world knowledge instead of carefully examining pixels, leading to systematic errors when images contradict familiar patterns. For example, a VLM might insist an Audi logo has 4 rings even when shown an image with 5 rings.

This repository provides a **controlled, automated framework** to measure this phenomenon across multiple domains with paired experimental conditions.

## ✨ Key Features

- **🎮 Single-factor control**: Only one visual attribute changes per image (ring count, leg count, etc.)
- **🔄 Paired variants**: Direct A/B testing with `vision-only` vs `vision+title` conditions  
- **📊 Automated scoring**: Self-contained metadata enables fully automated evaluation
- **⚡ Lightweight**: All images generated procedurally with PIL—no external assets required
- **🔬 Reproducible**: Standardized evaluation harness with JSON output for analysis

## 🏗️ Repository Structure

```
simple-vlm-bias/
├── dataset_generator.py           # Generate test images and metadata
├── test_gemini.py                 # Evaluation harness for Gemini models
├── api_key/
│   └── gemini_key.txt            # Your Gemini API key
├── vlms-are-biased-notitle/      # Vision-only test images
│   └── <topic>/images/*.png
├── vlms-are-biased-in_image_title/ # Vision+title test images  
│   └── <topic>/images/*.png
└── results/
    └── gemini_results_*.json     # Evaluation results
```

## 🧪 Test Domains

| Domain | Visual Element | Control Variable | Examples |
|--------|---------------|------------------|----------|
| **Car Logos** | Audi rings | Ring count | 4 vs 5 rings |
| **Animals** | Zebra legs | Leg count | 4 vs 5 legs |
| **Brand Elements** | Adidas stripes | Stripe count | 3 vs 4 stripes |
| **Optical Illusions** | Müller-Lyer lines | Line length | Equal vs unequal |

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.9+ required
pip install pillow google-generativeai
```

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/simple-vlm-bias.git
   cd simple-vlm-bias
   ```

2. **Add your Gemini API key**
   ```bash
   mkdir api_key
   echo "your_gemini_api_key_here" > api_key/gemini_key.txt
   ```

3. **Generate test datasets**
   ```bash
   # Generate all domains
   python dataset_generator.py --generate all --num-images 5
   
   # Or generate specific domain
   python dataset_generator.py --generate animals --num-images 3
   ```

4. **Run evaluation**
   ```bash
   python test_gemini.py vlms-are-biased-notitle/animals/animals_notitle_metadata.json \
     --model gemini-1.5-pro --samples 10 --output-dir results
   ```

5. **Analyze results**
   ```bash
   python test_gemini.py dummy --analyze results/gemini_results_XXXXXXXX.json
   ```

## 📈 What Gets Measured

For each test sample, we compute three key metrics:

- **`matches_ground_truth`**: Does the model's response contain the correct visual answer?
- **`shows_expected_bias`**: Does the response reflect the expected world prior (e.g., "4 rings")?  
- **`bias_confirmed`**: Does the model show bias AND get the answer wrong?

Results are automatically aggregated by topic with percentages for easy analysis.

## 🔧 API Reference

### Dataset Generation (`dataset_generator.py`)

| Function | Purpose |
|----------|---------|
| `generate_audi_logo_image()` | Creates car logo with variable ring count |
| `generate_zebra_image()` | Renders zebra with variable leg count |
| `generate_adidas_shoe_image()` | Draws shoe with variable stripe count |
| `generate_line_illusion_image()` | Creates Müller-Lyer optical illusion |
| `add_title_to_image()` | Adds text caption to create titled variant |

### Evaluation (`test_gemini.py`)

| Function | Purpose |
|----------|---------|
| `test_with_gemini()` | Core evaluation loop with bias metrics |
| `analyze_results()` | Offline analysis of saved results |
| `load_gemini_client()` | API client setup with error handling |

## 📊 Example Results

```json
{
  "topic": "animals",
  "samples": 10,
  "correct_responses": 3,
  "biased_responses": 6,
  "accuracy": 30.0,
  "bias_rate": 60.0
}
```

## 🛠️ Customization

**Add new test domains:**
1. Implement generation function in `dataset_generator.py`
2. Add topic to the generation pipeline
3. Define appropriate prompts and expected responses

**Test other VLMs:**
- Modify `test_gemini.py` to use different model APIs
- Maintain the same evaluation metrics for consistency

## ❓ Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: google.generativeai` | Run `pip install google-generativeai` |
| API key errors | Ensure `api_key/gemini_key.txt` contains only your key (no quotes/spaces) |
| Image not found | Don't move metadata files from their image directories |
| Rate limits | Increase `time.sleep()` value in `test_gemini.py` |
| Font issues | Missing `arial.ttf` is OK; code falls back to PIL default |

## 🤝 Contributing

We welcome contributions! Areas of interest:

- 🎨 **New test domains** (faces, text, scientific diagrams)
- 🧮 **Enhanced scoring metrics** (confidence scores, explanation analysis)
- 🛡️ **Bias mitigation techniques** (prompting strategies, fine-tuning approaches)
- 🔧 **Additional VLM support** (Claude, GPT-4V, LLaVA)

## 📞 Contact & Collaboration

- **Author**: Aman Kumar
- **Email**: aman24012@iiitd.ac.in
- **Research Interest Form**: [Fill out here](https://docs.google.com/forms/d/1nyHmBjqQIHcEL5vWBqckcGm8NLuv0ZTy67JkuDh8TNQ)

*Researchers, practitioners, and students interested in VLM evaluation, bias detection, or AI safety are encouraged to reach out!*

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google Gemini team for the evaluation API
- PIL/Pillow community for image generation tools
- Vision-Language research community for inspiration

---

<p align="center">
  <strong>🔬 Built for reproducible AI research</strong><br>
  <em>Help us understand when VLMs see what they expect vs. what's actually there</em>
</p>
