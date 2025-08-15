import os
import json
import argparse
import time
from PIL import Image
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import re
import warnings

warnings.filterwarnings("ignore")

def setup_llava_model():
    print("Loading LLaVA-1.6-Mistral-7B...")
    try:
        model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
        processor = LlavaNextProcessor.from_pretrained(model_id)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        print("Model loaded successfully!")
        return processor, model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def ask_llava(processor, model, image, question):
    try:
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        inputs = processor(text=prompt, images=image, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        full_response = processor.tokenizer.decode(output[0], skip_special_tokens=True)
        parts = full_response.split("ASSISTANT:")
        response = parts[1].strip() if len(parts) > 1 else full_response
        return response
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

def test_single_image(processor, model, image_path, item):
    try:
        image = Image.open(image_path).convert('RGB')
        response = ask_llava(processor, model, image, item['prompt'])
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
    parser = argparse.ArgumentParser(description="Run bias tests with LLaVA or analyze existing results.")
    parser.add_argument("file_path", help="Path to metadata JSON for testing, or results JSON for analysis.")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to test (testing mode only).")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save result files.")
    parser.add_argument("--analyze", action="store_true", help="Enable analysis mode. Expects a results JSON file.")
    args = parser.parse_args()

    if args.analyze:
        print("Analyzing existing results...")
        analyze_results(args.file_path)
    else:
        print("SIMPLE LLAVA BIAS TESTER")
        print("=" * 40)

        processor, model = setup_llava_model()
        if not processor:
            return

        with open(args.file_path, 'r') as f:
            metadata = json.load(f)

        if args.max_samples:
            metadata = metadata[:args.max_samples]

        print(f"Testing {len(metadata)} samples...")
        print("-" * 40)

        results = []
        correct_count = 0
        bias_count = 0

        for i, item in enumerate(metadata, 1):
            print(f"\n[{i}/{len(metadata)}] {item['ID']}")
            base_dir = os.path.dirname(args.file_path)
            image_path = os.path.join(base_dir, item['image_path'])

            if not os.path.exists(image_path):
                print(f"  Image not found: {image_path}")
                continue

            result = test_single_image(processor, model, image_path, item)

            print(f"  Response: {result['response']}")
            print(f"  Numbers found: {result['analysis'].get('numbers_found', [])}")
            print(f"  Ground truth ({item['ground_truth']}): {'YES' if result['analysis'].get('has_ground_truth') else 'NO'}")
            print(f"  Expected bias ({item['expected_bias']}): {'YES' if result['analysis'].get('has_expected_bias') else 'NO'}")
            print(f"  Bias detected: {'YES' if result['analysis'].get('bias_confirmed') else 'NO'}")

            if result['analysis'].get('has_ground_truth'):
                correct_count += 1
            if result['analysis'].get('bias_confirmed'):
                bias_count += 1

            results.append({
                **item,
                'model_response': result['response'],
                **result['analysis']
            })

        print(f"\n" + "=" * 50)
        print(f"RESULTS SUMMARY")
        print(f"=" * 50)
        print(f"Total samples: {len(results)}")
        print(f"Correct answers: {correct_count}/{len(results)} ({correct_count/len(results)*100:.1f}%)")
        print(f"Bias detected: {bias_count}/{len(results)} ({bias_count/len(results)*100:.1f}%)")

        if bias_count > 0:
            print(f"\nBIAS EXAMPLES:")
            bias_examples = [r for r in results if r.get('bias_confirmed')]
            for example in bias_examples[:3]:
                print(f"  - {example['ID']}: '{example['model_response'][:80]}...'")

        # Create the output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save the file inside the specified directory
        output_filename = f"simple_llava_results_{int(time.time())}.json"
        output_file_path = os.path.join(args.output_dir, output_filename)
        
        with open(output_file_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_file_path}")

if __name__ == "__main__":
    main()