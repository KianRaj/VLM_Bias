# test_ollama.py - VLM Bias Testing with LLaVA via Ollama

import os
import json
import argparse
import time
from PIL import Image
import re
import warnings
import ollama
import base64
from io import BytesIO

warnings.filterwarnings("ignore")

def check_and_pull_model(model_name):
    """Checks if the Ollama model is available locally and pulls it if not."""
    print(f"Checking for Ollama model: {model_name}...")
    try:
        # List models and check if the desired one is present
        local_models = [m['name'] for m in ollama.list()['models']]
        if model_name not in local_models:
            print(f"Model not found locally. Pulling '{model_name}'...")
            ollama.pull(model_name)
            print("Model pulled successfully.")
        else:
            print("Model already available locally.")
        return True
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        print("Please ensure the Ollama application is running.")
        return False

def image_to_base64(image):
    """Converts a PIL Image to a base64 encoded string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def ask_ollama_llava(image, question, model_name="llava:latest"):
    try:
        # Convert the PIL image to a base64 string
        base64_image = image_to_base64(image)
        
        # Make the API call to the local Ollama service
        response = ollama.generate(
            model=model_name,
            prompt=question,
            images=[base64_image],
            stream=False
        )
        return response['response'].strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

def analyze_response(response, ground_truth, expected_bias):
    response_lower = response.lower()
    word_to_digit = {
        'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'
    }
    analysis_text = response_lower
    for word, digit in word_to_digit.items():
        analysis_text = analysis_text.replace(word, f" {digit} ")

    numbers = re.findall(r'\b\d+\b', analysis_text)
    has_ground_truth = str(ground_truth) in numbers
    has_expected_bias = str(expected_bias) in numbers
    bias_confirmed = has_expected_bias and not has_ground_truth

    return {
        'numbers_found': numbers,
        'has_ground_truth': has_ground_truth,
        'has_expected_bias': has_expected_bias,
        'bias_confirmed': bias_confirmed
    }

def test_single_image(image_path, item):
    try:
        image = Image.open(image_path).convert('RGB')
        response = ask_ollama_llava(image, item['prompt'])
        analysis = analyze_response(response, item['ground_truth'], item['expected_bias'])
        return {
            'response': response,
            'analysis': analysis
        }
    except Exception as e:
        return {
            'response': f"ERROR: {str(e)}",
            'analysis': {'error': True}
        }

def analyze_results(results_file):
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: Results file not found at {results_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in results file: {results_file}")
        return

    if not results:
        print("No results found in file.")
        return

    bias_count = sum(1 for r in results if r.get('bias_confirmed', False))
    correct_count = sum(1 for r in results if r.get('has_ground_truth', False))
    total_samples = len(results)

    print(f"\n" + "=" * 50)
    print(f"ANALYSIS OF SAVED RESULTS: {os.path.basename(results_file)}")
    print(f"=" * 50)
    print(f"Total samples: {total_samples}")
    if total_samples > 0:
        print(f"Correct answers: {correct_count}/{total_samples} ({correct_count/total_samples*100:.1f}%)")
        print(f"Bias detected: {bias_count}/{total_samples} ({bias_count/total_samples*100:.1f}%)")
    print(f"=" * 50)

def main():
    parser = argparse.ArgumentParser(description="Run bias tests with a VLM via Ollama or analyze results.")
    parser.add_argument("file_path", help="Path to metadata JSON for testing, or results JSON for analysis.")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to test (testing mode only).")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save result files.")
    parser.add_argument("--analyze", action="store_true", help="Enable analysis mode. Expects a results JSON file.")
    args = parser.parse_args()

    if args.analyze:
        print("Analyzing existing results...")
        analyze_results(args.file_path)
    else:
        print("OLLAMA VLM BIAS TESTER")
        print("=" * 40)
        
        # Check for and pull the LLaVA model
        model_name = "llava:latest"
        if not check_and_pull_model(model_name):
            return

        with open(args.file_path, 'r') as f:
            metadata = json.load(f)

        if args.max_samples:
            metadata = metadata[:args.max_samples]

        print(f"Testing {len(metadata)} samples...")
        print("-" * 40)

        results = []
        for i, item in enumerate(metadata, 1):
            print(f"\n[{i}/{len(metadata)}] {item['ID']}")
            base_dir = os.path.dirname(args.file_path)
            image_path = os.path.join(base_dir, item['image_path'])

            if not os.path.exists(image_path):
                print(f"  Image not found: {image_path}")
                continue

            result = test_single_image(image_path, item)

            print(f"  Response: {result['response']}")
            print(f"  Numbers found: {result['analysis'].get('numbers_found', [])}")
            print(f"  Ground truth ({item['ground_truth']}): {'YES' if result['analysis'].get('has_ground_truth') else 'NO'}")
            print(f"  Expected bias ({item['expected_bias']}): {'YES' if result['analysis'].get('has_expected_bias') else 'NO'}")
            print(f"  Bias detected: {'YES' if result['analysis'].get('bias_confirmed') else 'NO'}")

            results.append({**item, 'model_response': result['response'], **result['analysis']})

        print(f"\n" + "=" * 50)
        print(f"RESULTS SUMMARY")
        print(f"Total samples: {len(results)}")
        # ... (summary printing) ...
        
        os.makedirs(args.output_dir, exist_ok=True)
        output_filename = f"ollama_llava_results_{int(time.time())}.json"
        output_file_path = os.path.join(args.output_dir, output_filename)
        
        with open(output_file_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file_path}")

if __name__ == "__main__":
    main()