# test_grok.py - Grok API Testing for VLM Bias Dataset
"""
Test generated bias datasets with Grok API (xAI)
Adapted from Gemini testing script for Grok model compatibility
"""

import os
import json
import argparse
import time
import base64
from PIL import Image
import io

# Grok API imports
try:
    import requests
    HAS_REQUESTS = True
    print("✅ Requests module imported successfully.")
except ImportError as e:
    HAS_REQUESTS = False
    print("❌ Requests module NOT found.")
    print(f"ImportError: {e}")

# --- Grok API Functions ---
def load_grok_client(api_key_path="api_key/grok_key.txt"):
    """Load Grok API key and return it"""
    try:
        with open(api_key_path, 'r') as f:
            api_key = f.read().strip()
        print(f"✅ Grok API key loaded successfully")
        return api_key
    except FileNotFoundError:
        raise FileNotFoundError(f"API key not found at {api_key_path}")
    except Exception as e:
        print(f"❌ Error loading Grok API key: {e}")
        return None

def encode_image_to_base64(image_path):
    """Convert image to base64 string for API"""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if too large (optional - adjust max size as needed)
            max_size = 1024
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            img_bytes = buffer.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            return img_base64
    except Exception as e:
        print(f"❌ Error encoding image: {e}")
        return None

def call_grok_api(api_key, prompt, image_base64, model_name="grok-vision-beta"):
    """Make API call to Grok vision model"""
    url = "https://api.x.ai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Construct the message with image
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]
        }
    ]
    
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.1,  # Lower temperature for consistent results
        "max_tokens": 100,
        "stream": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            return "No response generated"
            
    except requests.exceptions.RequestException as e:
        print(f"❌ API Request Error: {e}")
        return f"API_ERROR: {str(e)}"
    except json.JSONDecodeError as e:
        print(f"❌ JSON Decode Error: {e}")
        return f"JSON_ERROR: {str(e)}"
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return f"ERROR: {str(e)}"

def test_with_grok(metadata_file, model_name="grok-vision-beta", 
                   max_samples=10, output_dir="results"):
    """Test generated dataset with Grok API"""
    print(f"DEBUG: HAS_REQUESTS = {HAS_REQUESTS}")

    if not HAS_REQUESTS:
        print("❌ Requests library not available. Please install with: pip install requests")
        return
    
    # Load API key
    api_key = load_grok_client()
    if not api_key:
        return
    
    print(f"🧪 Testing with Grok model: {model_name}")
    
    # Load metadata
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print(f"❌ Metadata file not found: {metadata_file}")
        return
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in metadata file: {metadata_file}")
        return
    
    # Limit samples for testing
    if max_samples:
        metadata = metadata[:max_samples]
    
    results = []
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📊 Processing {len(metadata)} samples...")
    
    for i, item in enumerate(metadata, 1):
        try:
            print(f"[{i}/{len(metadata)}] Processing {item['ID']}")
            
            # Load image
            image_path = os.path.join(
                os.path.dirname(metadata_file), item['image_path']
            )
            
            if not os.path.exists(image_path):
                print(f"  ❌ Image not found: {image_path}")
                continue
            
            # Encode image to base64
            image_base64 = encode_image_to_base64(image_path)
            if not image_base64:
                print(f"  ❌ Failed to encode image: {image_path}")
                continue
            
            # Make API call
            response_text = call_grok_api(api_key, item['prompt'], image_base64, model_name)
            
            # Store result
            result = item.copy()
            result['model_response'] = response_text
            result['model_name'] = model_name
            result['timestamp'] = int(time.time())
            
            # Simple bias analysis
            response_lower = response_text.lower()
            ground_truth_lower = str(item['ground_truth']).lower()
            expected_bias_lower = str(item['expected_bias']).lower()
            
            result['matches_ground_truth'] = ground_truth_lower in response_lower
            result['shows_expected_bias'] = expected_bias_lower in response_lower
            result['bias_confirmed'] = result['shows_expected_bias'] and not result['matches_ground_truth']
            
            results.append(result)
            
            print(f"  📝 Response: {response_text[:100]}...")
            print(f"  🎯 Ground Truth: {item['ground_truth']}")
            print(f"  🧠 Expected Bias: {item['expected_bias']}")
            print(f"  ✅ Correct: {result['matches_ground_truth']}")
            print(f"  ⚠️  Bias detected: {result['bias_confirmed']}")
            
            # Delay to avoid rate limiting
            time.sleep(1.0)  # Slightly longer delay for API stability
            
        except Exception as e:
            print(f"  ❌ Error processing {item['ID']}: {e}")
            continue
    
    # Save results
    timestamp = int(time.time())
    output_file = os.path.join(output_dir, f"grok_results_{timestamp}.json")
    
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return results
    
    # Print analysis
    if results:
        bias_detected = sum(1 for r in results if r.get('bias_confirmed', False))
        correct_answers = sum(1 for r in results if r.get('matches_ground_truth', False))
        total_samples = len(results)
        
        print(f"\n" + "="*50)
        print(f"🔍 GROK BIAS ANALYSIS RESULTS")
        print(f"="*50)
        print(f"📊 Total samples processed: {total_samples}")
        print(f"✅ Correct answers: {correct_answers}/{total_samples} ({correct_answers/total_samples*100:.1f}%)")
        print(f"⚠️  Expected bias detected: {bias_detected}/{total_samples} ({bias_detected/total_samples*100:.1f}%)")
        
        if bias_detected > 0:
            print(f"🚨 Grok model shows bias in {bias_detected/total_samples*100:.1f}% of cases")
        else:
            print(f"✨ No significant bias detected in Grok model")
            
        print(f"📁 Detailed results: {output_file}")
        print(f"="*50)
    
    return results

def analyze_results(results_file):
    """Analyze saved results file"""
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"❌ Results file not found: {results_file}")
        return
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in results file: {results_file}")
        return
    
    if not results:
        print("❌ No results found in file")
        return
    
    # Group by topic
    topics = {}
    for result in results:
        topic = result.get('topic', 'unknown')
        if topic not in topics:
            topics[topic] = []
        topics[topic].append(result)
    
    print(f"\n" + "="*60)
    print(f"📈 DETAILED GROK BIAS ANALYSIS")
    print(f"="*60)
    
    total_bias = 0
    total_correct = 0
    total_samples = 0
    
    for topic, topic_results in topics.items():
        bias_detected = sum(1 for r in topic_results if r.get('bias_confirmed', False))
        correct_answers = sum(1 for r in topic_results if r.get('matches_ground_truth', False))
        total = len(topic_results)
        
        total_bias += bias_detected
        total_correct += correct_answers
        total_samples += total
        
        print(f"\n🎯 {topic.upper()}:")
        print(f"   📊 Samples: {total}")
        print(f"   ✅ Correct: {correct_answers}/{total} ({correct_answers/total*100:.1f}%)")
        print(f"   ⚠️  Bias: {bias_detected}/{total} ({bias_detected/total*100:.1f}%)")
        
        # Show some example responses
        bias_examples = [r for r in topic_results if r.get('bias_confirmed', False)]
        correct_examples = [r for r in topic_results if r.get('matches_ground_truth', False)]
        
        if bias_examples:
            print(f"   📝 Bias example: '{bias_examples[0]['model_response'][:60]}...'")
        if correct_examples:
            print(f"   ✨ Correct example: '{correct_examples[0]['model_response'][:60]}...'")
    
    print(f"\n🎯 OVERALL SUMMARY:")
    print(f"   📊 Total samples: {total_samples}")
    print(f"   ✅ Overall accuracy: {total_correct}/{total_samples} ({total_correct/total_samples*100:.1f}%)")
    print(f"   ⚠️  Overall bias rate: {total_bias}/{total_samples} ({total_bias/total_samples*100:.1f}%)")
    
    print(f"\n" + "="*60)

def compare_with_other_models(grok_results_file, other_results_file):
    """Compare Grok results with other model results"""
    try:
        with open(grok_results_file, 'r') as f:
            grok_results = json.load(f)
        with open(other_results_file, 'r') as f:
            other_results = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ Results file not found: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in results file: {e}")
        return
    
    if not grok_results or not other_results:
        print("❌ No results found in one or both files")
        return
    
    # Calculate metrics
    grok_bias = sum(1 for r in grok_results if r.get('bias_confirmed', False))
    grok_correct = sum(1 for r in grok_results if r.get('matches_ground_truth', False))
    grok_total = len(grok_results)
    
    other_bias = sum(1 for r in other_results if r.get('bias_confirmed', False))
    other_correct = sum(1 for r in other_results if r.get('matches_ground_truth', False))
    other_total = len(other_results)
    
    other_model_name = other_results[0].get('model_name', 'Other Model') if other_results else 'Other Model'
    
    print(f"\n" + "="*60)
    print(f"🔄 MODEL COMPARISON")
    print(f"="*60)
    print(f"🤖 Grok Model:")
    print(f"   📊 Samples: {grok_total}")
    print(f"   ✅ Accuracy: {grok_correct/grok_total*100:.1f}%")
    print(f"   ⚠️  Bias Rate: {grok_bias/grok_total*100:.1f}%")
    
    print(f"\n🤖 {other_model_name}:")
    print(f"   📊 Samples: {other_total}")
    print(f"   ✅ Accuracy: {other_correct/other_total*100:.1f}%")
    print(f"   ⚠️  Bias Rate: {other_bias/other_total*100:.1f}%")
    
    print(f"\n📈 COMPARISON:")
    acc_diff = (grok_correct/grok_total) - (other_correct/other_total)
    bias_diff = (grok_bias/grok_total) - (other_bias/other_total)
    
    print(f"   📊 Accuracy difference: {acc_diff*100:+.1f}% (Grok vs {other_model_name})")
    print(f"   ⚠️  Bias difference: {bias_diff*100:+.1f}% (Grok vs {other_model_name})")
    
    if acc_diff > 0:
        print(f"   ✨ Grok is more accurate")
    elif acc_diff < 0:
        print(f"   ✨ {other_model_name} is more accurate")
    else:
        print(f"   ⚖️  Similar accuracy")
    
    if bias_diff > 0:
        print(f"   🚨 Grok shows more bias")
    elif bias_diff < 0:
        print(f"   🚨 {other_model_name} shows more bias")
    else:
        print(f"   ⚖️  Similar bias levels")
    
    print(f"="*60)

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Test VLM Bias Dataset with Grok API")
    parser.add_argument("metadata_file", nargs='?', help="Path to metadata JSON file")
    parser.add_argument("--model", default="grok-vision-beta", 
                       help="Grok model to use (default: grok-vision-beta)")
    parser.add_argument("--samples", type=int, default=10, 
                       help="Max samples to test (default: 10)")
    parser.add_argument("--output-dir", default="results",
                       help="Output directory for results (default: results)")
    parser.add_argument("--analyze", type=str, help="Analyze existing Grok results file")
    parser.add_argument("--compare", nargs=2, metavar=('GROK_RESULTS', 'OTHER_RESULTS'),
                       help="Compare Grok results with other model results")
    
    args = parser.parse_args()
    
    if args.compare:
        print("🔄 Comparing models...")
        compare_with_other_models(args.compare[0], args.compare[1])
    elif args.analyze:
        print("🔍 Analyzing existing Grok results...")
        analyze_results(args.analyze)
    else:
        if not args.metadata_file:
            print("❌ Please provide a metadata file or use --analyze/--compare options")
            parser.print_help()
            return
            
        print("🧪 Testing VLM bias with Grok API...")
        if not os.path.exists(args.metadata_file):
            print(f"❌ Metadata file not found: {args.metadata_file}")
            print("\nGenerate dataset first with: python dataset_generator.py --generate all")
            return
        
        results = test_with_grok(
            args.metadata_file, 
            args.model, 
            args.samples,
            args.output_dir
        )

if __name__ == "__main__":
    main()