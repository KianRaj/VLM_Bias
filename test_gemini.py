# test_gemini.py - Gemini API Testing for VLM Bias Dataset
"""
Test generated bias datasets with Google Gemini API
Fixed API compatibility issues with newer Gemini SDK versions
"""

import os
import json
import argparse
import time
from PIL import Image

# Google Gemini imports
try:
    import google.generativeai as genai
    HAS_GEMINI = True
    print("✅ Gemini module imported successfully.")
except ImportError as e:
    HAS_GEMINI = False
    print("❌ Gemini module NOT found.")
    print(f"ImportError: {e}")

# --- Gemini API Functions ---
def load_gemini_client(api_key_path="api_key/gemini_key.txt"):
    """Configure Gemini API and return configured genai module"""
    try:
        with open(api_key_path, 'r') as f:
            api_key = f.read().strip()
        genai.configure(api_key=api_key)
        print(f"✅ Gemini API configured successfully")
        return True
    except FileNotFoundError:
        raise FileNotFoundError(f"API key not found at {api_key_path}")
    except Exception as e:
        print(f"❌ Error configuring Gemini API: {e}")
        return False

def test_with_gemini(metadata_file, model_name="gemini-1.5-pro", 
                    max_samples=10, output_dir="results"):
    """Test generated dataset with Gemini API"""
    print(f"DEBUG: HAS_GEMINI = {HAS_GEMINI}")

    if not HAS_GEMINI:
        print("❌ Gemini API not available. Skipping testing.")
        return
    
    # Configure API
    if not load_gemini_client():
        return
    
    print(f"🧪 Testing with Gemini model: {model_name}")
    
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
    
    # Initialize model
    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        print(f"❌ Error initializing model {model_name}: {e}")
        return
    
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
            
            # Load and prepare image
            try:
                image = Image.open(image_path)
            except Exception as e:
                print(f"  ❌ Error loading image: {e}")
                continue
            
            # Make API call with proper configuration
            try:
                # Use the simplified API call that works with newer versions
                response = model.generate_content(
                    [item['prompt'], image],
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,  # Lower temperature for more consistent results
                        top_p=1,
                        top_k=32,
                        max_output_tokens=100,
                    )
                )
                
                # Extract response text
                if response.candidates and len(response.candidates) > 0:
                    response_text = response.candidates[0].content.parts[0].text
                else:
                    response_text = "No response generated"
                    
            except Exception as e:
                print(f"  ❌ API Error: {e}")
                response_text = f"API_ERROR: {str(e)}"
            
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
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  ❌ Error processing {item['ID']}: {e}")
            continue
    
    # Save results
    timestamp = int(time.time())
    output_file = os.path.join(output_dir, f"gemini_results_{timestamp}.json")
    
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
        print(f"🔍 BIAS ANALYSIS RESULTS")
        print(f"="*50)
        print(f"📊 Total samples processed: {total_samples}")
        print(f"✅ Correct answers: {correct_answers}/{total_samples} ({correct_answers/total_samples*100:.1f}%)")
        print(f"⚠️  Expected bias detected: {bias_detected}/{total_samples} ({bias_detected/total_samples*100:.1f}%)")
        
        if bias_detected > 0:
            print(f"🚨 Model shows bias in {bias_detected/total_samples*100:.1f}% of cases")
        else:
            print(f"✨ No significant bias detected")
            
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
    print(f"📈 DETAILED BIAS ANALYSIS")
    print(f"="*60)
    
    for topic, topic_results in topics.items():
        bias_detected = sum(1 for r in topic_results if r.get('bias_confirmed', False))
        correct_answers = sum(1 for r in topic_results if r.get('matches_ground_truth', False))
        total = len(topic_results)
        
        print(f"\n🎯 {topic.upper()}:")
        print(f"   📊 Samples: {total}")
        print(f"   ✅ Correct: {correct_answers}/{total} ({correct_answers/total*100:.1f}%)")
        print(f"   ⚠️  Bias: {bias_detected}/{total} ({bias_detected/total*100:.1f}%)")
        
        # Show some example responses
        bias_examples = [r for r in topic_results if r.get('bias_confirmed', False)]
        if bias_examples:
            print(f"   📝 Bias example: '{bias_examples[0]['model_response'][:50]}...'")
    
    print(f"\n" + "="*60)



# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Test VLM Bias Dataset with Gemini API")
    parser.add_argument("metadata_file", help="Path to metadata JSON file")
    parser.add_argument("--model", default="gemini-1.5-pro", 
                       help="Gemini model to use (default: gemini-1.5-pro)")
    parser.add_argument("--samples", type=int, default=10, 
                       help="Max samples to test (default: 10)")
    parser.add_argument("--output-dir", default="results",
                       help="Output directory for results (default: results)")
    parser.add_argument("--analyze", type=str, help="Analyze existing results file")
    
    args = parser.parse_args()
    
    if args.analyze:
        print("🔍 Analyzing existing results...")
        analyze_results(args.analyze)
    else:
        print("🧪 Testing VLM bias with Gemini API...")
        if not os.path.exists(args.metadata_file):
            print(f"❌ Metadata file not found: {args.metadata_file}")
            print("\nGenerate dataset first with: python dataset_generator.py --generate all")
            return
        
        results = test_with_gemini(
            args.metadata_file, 
            args.model, 
            args.samples,
            args.output_dir
        )

if __name__ == "__main__":
    main()