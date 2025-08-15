# dataset_generator.py - VLM Bias Testing Dataset Generator
"""
Dataset generator for VLM bias testing framework:
- Car logos (Audi rings) with modified counts
- Animals (zebra legs) with altered numbers  
- Adidas stripes with different counts
- Optical illusions (horizontal line length comparison)
"""

import os
import json
import argparse
from PIL import Image, ImageDraw, ImageFont
import textwrap
import re

# --- Configuration ---
AVAILABLE_TOPICS = ['car_logos', 'animals', 'adidas_stripes', 'optical_illusions']
TITLE_TYPES = ['notitle', 'in_image_title']

DEFAULT_TITLES = {
    "car_logos": "Audi",
    "animals": "Zebra", 
    "adidas_stripes": "Adidas",
    "optical_illusions": "Line comparison"
}

# Test questions for bias detection (replicating the research examples)
CAR_LOGO_QUESTIONS = [
    {
        "type": "Q1", 
        "prompt": "How many overlapping circles are there in the logo of this car?",
        "ground_truth": "5",  # Modified from standard 4 Audi rings
        "expected_bias": "4"   # Models expect standard 4 Audi rings
    }
]

ANIMAL_QUESTIONS = [
    {
        "type": "Q1",
        "prompt": "How many legs does this animal have?", 
        "ground_truth": "5",  # Modified from standard 4 zebra legs
        "expected_bias": "4"   # Models expect standard 4 legs
    }
]

ADIDAS_QUESTIONS = [
    {
        "type": "Q1",
        "prompt": "How many visible stripes are there in the logo of the left shoe?",
        "ground_truth": "4",  # Modified from standard 3 Adidas stripes  
        "expected_bias": "3"   # Models expect standard 3 stripes
    }
]

ILLUSION_QUESTIONS = [
    {
        "type": "Q1",
        "prompt": "Are the two horizontal lines equal in length?",
        "ground_truth": "No",
        "expected_bias": "Yes"  # Optical illusion makes them appear equal
    }
]

# --- Utility Functions ---
def sanitize_filename(name):
    """Clean filename for safe storage"""
    name = str(name)
    name = re.sub(r'[\\/*?:"<>|\s]+', '_', name)
    name = re.sub(r'[^\w\-]+', '', name) 
    name = re.sub(r'_+', '_', name)
    return name.strip('_') or "sanitized"

def create_directory_structure(topic):
    """Create output directories"""
    dirs = {}
    for title_type in TITLE_TYPES:
        base_dir = f"vlms-are-biased-{title_type}"
        topic_dir = os.path.join(base_dir, topic)
        img_dir = os.path.join(topic_dir, "images")
        
        os.makedirs(img_dir, exist_ok=True)
        
        dirs[title_type] = {
            "base": topic_dir,
            "images": img_dir
        }
    
    return dirs

def add_title_to_image(image_path, output_path, title_text):
    """Add title bar to image"""
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            width, height = img.size
            font_size = max(20, width // 20)
            
            # Wrap title text
            wrapped_lines = textwrap.wrap(title_text, width=40)
            line_height = int(font_size * 1.2)
            title_height = len(wrapped_lines) * line_height + 40
            
            # Create new image with title space
            new_img = Image.new('RGB', (width, height + title_height), 'white')
            new_img.paste(img, (0, title_height))
            
            # Draw title
            draw = ImageDraw.Draw(new_img)
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            y_pos = 20
            for line in wrapped_lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                x_pos = (width - text_width) // 2
                draw.text((x_pos, y_pos), line, fill='black', font=font)
                y_pos += line_height
            
            new_img.save(output_path, 'PNG')
            return True
    except Exception as e:
        print(f"Error adding title to {image_path}: {e}")
        return False

# --- Dataset Generators ---
def generate_audi_logo_image(output_path, num_rings=5):
    """Generate Audi-style logo with specified number of rings"""
    try:
        img_size = 400
        img = Image.new('RGB', (img_size, img_size), 'white')
        draw = ImageDraw.Draw(img)
        
        # Car silhouette background
        car_width = 300
        car_height = 120
        car_x = (img_size - car_width) // 2
        car_y = img_size // 2 - 50
        
        # Draw simple car shape
        draw.rectangle([car_x, car_y, car_x + car_width, car_y + car_height], 
                      fill='#2C3E50', outline='black', width=2)
        
        # Draw windshield
        draw.polygon([(car_x + 50, car_y), (car_x + 250, car_y), 
                     (car_x + 230, car_y + 40), (car_x + 70, car_y + 40)], 
                     fill='#87CEEB')
        
        # Draw wheels
        wheel_y = car_y + car_height - 10
        draw.ellipse([car_x + 30, wheel_y, car_x + 70, wheel_y + 40], 
                    fill='black', outline='gray', width=3)
        draw.ellipse([car_x + 230, wheel_y, car_x + 270, wheel_y + 40], 
                    fill='black', outline='gray', width=3)
        
        # Draw Audi-style rings (the key bias test element)
        ring_size = 40
        ring_stroke = 8
        center_y = car_y + 60
        
        if num_rings == 4:  # Standard Audi
            ring_positions = [-60, -20, 20, 60]
        elif num_rings == 5:  # Modified version
            ring_positions = [-80, -40, 0, 40, 80]
        else:  # Custom
            spacing = 40
            start_x = -(num_rings - 1) * spacing // 2
            ring_positions = [start_x + i * spacing for i in range(num_rings)]
        
        for i, offset_x in enumerate(ring_positions):
            ring_x = img_size // 2 + offset_x
            
            # Different colors for visibility
            colors = ['#FF0000', '#00FF00', '#0000FF', '#FFD700', '#FF69B4']
            ring_color = colors[i % len(colors)]
            
            draw.ellipse([ring_x - ring_size//2, center_y - ring_size//2, 
                         ring_x + ring_size//2, center_y + ring_size//2],
                        outline=ring_color, width=ring_stroke)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path, 'PNG')
        return True
        
    except Exception as e:
        print(f"Error generating Audi logo image: {e}")
        return False

def generate_zebra_image(output_path, num_legs=5):
    """Generate zebra with specified number of legs"""
    try:
        img_size = 600
        img = Image.new('RGB', (img_size, img_size), '#90EE90')  # Light green background
        draw = ImageDraw.Draw(img)
        
        # Draw zebra body
        body_width = 200
        body_height = 100
        body_x = img_size // 2 - body_width // 2
        body_y = img_size // 2 - 50
        
        # Main body (ellipse)
        draw.ellipse([body_x, body_y, body_x + body_width, body_y + body_height], 
                    fill='white', outline='black', width=3)
        
        # Head
        head_size = 60
        head_x = body_x - head_size + 20
        head_y = body_y - 10
        draw.ellipse([head_x, head_y, head_x + head_size, head_y + head_size], 
                    fill='white', outline='black', width=3)
        
        # Ears
        ear_size = 15
        draw.ellipse([head_x + 10, head_y + 5, head_x + 10 + ear_size, head_y + 5 + ear_size], 
                    fill='white', outline='black', width=2)
        draw.ellipse([head_x + 35, head_y + 5, head_x + 35 + ear_size, head_y + 5 + ear_size], 
                    fill='white', outline='black', width=2)
        
        # Eyes
        draw.ellipse([head_x + 15, head_y + 20, head_x + 20, head_y + 25], fill='black')
        draw.ellipse([head_x + 35, head_y + 20, head_x + 40, head_y + 25], fill='black')
        
        # Tail
        tail_x = body_x + body_width
        tail_y = body_y + body_height // 2
        draw.line([tail_x, tail_y, tail_x + 30, tail_y - 20], fill='black', width=4)
        
        # Legs (the key bias test element)
        leg_width = 15
        leg_height = 80
        leg_y_start = body_y + body_height - 10
        
        # Calculate leg positions
        if num_legs == 4:  # Standard zebra
            leg_positions = [body_x + 30, body_x + 70, body_x + 110, body_x + 150]
        elif num_legs == 5:  # Modified version  
            leg_positions = [body_x + 20, body_x + 60, body_x + 100, body_x + 140, body_x + 180]
        else:  # Custom
            spacing = body_width // (num_legs + 1)
            leg_positions = [body_x + spacing * (i + 1) for i in range(num_legs)]
        
        for leg_x in leg_positions:
            draw.rectangle([leg_x - leg_width//2, leg_y_start, 
                           leg_x + leg_width//2, leg_y_start + leg_height],
                          fill='white', outline='black', width=3)
            
            # Hooves
            draw.ellipse([leg_x - leg_width//2, leg_y_start + leg_height - 5,
                         leg_x + leg_width//2, leg_y_start + leg_height + 5],
                        fill='black')
        
        # Zebra stripes on body
        stripe_spacing = 20
        for i in range(0, body_width, stripe_spacing):
            stripe_x = body_x + i
            draw.rectangle([stripe_x, body_y + 10, stripe_x + 10, body_y + body_height - 10],
                          fill='black')
        
        # Stripes on head
        for i in range(0, head_size, 15):
            stripe_x = head_x + i
            draw.rectangle([stripe_x, head_y + 10, stripe_x + 7, head_y + head_size - 10],
                          fill='black')
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path, 'PNG')
        return True
        
    except Exception as e:
        print(f"Error generating zebra image: {e}")
        return False

def generate_adidas_shoe_image(output_path, num_stripes=4):
    """Generate shoe with Adidas-style stripes"""
    try:
        img_size = 600
        img = Image.new('RGB', (img_size, img_size), '#8B4513')  # Brown basketball court
        draw = ImageDraw.Draw(img)
        
        # Draw court lines
        draw.line([0, img_size//2, img_size, img_size//2], fill='white', width=3)
        draw.line([img_size//2, 0, img_size//2, img_size], fill='white', width=3)
        
        # Basketball
        ball_size = 80
        ball_x = img_size - 150
        ball_y = 100
        draw.ellipse([ball_x, ball_y, ball_x + ball_size, ball_y + ball_size], 
                    fill='#FF8C00', outline='black', width=3)
        
        # Basketball lines
        draw.line([ball_x, ball_y + ball_size//2, ball_x + ball_size, ball_y + ball_size//2], 
                  fill='black', width=2)
        draw.line([ball_x + ball_size//2, ball_y, ball_x + ball_size//2, ball_y + ball_size], 
                  fill='black', width=2)
        
        # Draw shoe (left shoe prominently)
        shoe_width = 180
        shoe_height = 100
        shoe_x = 100
        shoe_y = img_size // 2 + 50
        
        # Main shoe body
        draw.ellipse([shoe_x, shoe_y, shoe_x + shoe_width, shoe_y + shoe_height], 
                    fill='black', outline='gray', width=3)
        
        # Sole
        sole_height = 20
        draw.ellipse([shoe_x - 10, shoe_y + shoe_height - sole_height//2, 
                     shoe_x + shoe_width + 10, shoe_y + shoe_height + sole_height], 
                    fill='white', outline='black', width=2)
        
        # Adidas stripes (the key bias test element)
        stripe_width = 8
        stripe_spacing = 15
        stripe_start_x = shoe_x + 40
        stripe_start_y = shoe_y + 20
        stripe_length = 60
        
        if num_stripes == 3:  # Standard Adidas
            stripe_positions = [0, 1, 2]
        elif num_stripes == 4:  # Modified version
            stripe_positions = [0, 1, 2, 3]
        else:  # Custom
            stripe_positions = list(range(num_stripes))
        
        for i in stripe_positions:
            stripe_x = stripe_start_x + i * stripe_spacing
            # Draw angled stripes
            points = [
                (stripe_x, stripe_start_y),
                (stripe_x + stripe_width, stripe_start_y),
                (stripe_x + stripe_width + 20, stripe_start_y + stripe_length),
                (stripe_x + 20, stripe_start_y + stripe_length)
            ]
            draw.polygon(points, fill='white', outline='black', width=1)
        
        # Shoe laces
        lace_y = shoe_y + 30
        for i in range(4):
            y_pos = lace_y + i * 10
            draw.line([shoe_x + 70, y_pos, shoe_x + 110, y_pos], fill='white', width=2)
        
        # Draw legs in background
        leg_width = 40
        leg_height = 120
        leg_x = shoe_x + shoe_width//2 - leg_width//2
        leg_y = shoe_y - leg_height
        
        draw.rectangle([leg_x, leg_y, leg_x + leg_width, shoe_y], 
                      fill='#DDBEA9', outline='black', width=2)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path, 'PNG')
        return True
        
    except Exception as e:
        print(f"Error generating Adidas shoe image: {e}")
        return False

def generate_line_illusion_image(output_path, equal_length=False):
    """Generate horizontal line comparison (Müller-Lyer style)"""
    try:
        img_size = 400
        img = Image.new('RGB', (img_size, img_size), 'white')
        draw = ImageDraw.Draw(img)
        
        center_y = img_size // 2
        line_thickness = 3
        
        # Top line with inward arrows (appears shorter)
        top_y = center_y - 80
        top_line_length = 150 if equal_length else 140
        top_start_x = (img_size - top_line_length) // 2
        top_end_x = top_start_x + top_line_length
        
        # Draw main line
        draw.line([top_start_x, top_y, top_end_x, top_y], 
                  fill='black', width=line_thickness)
        
        # Inward arrows (make line appear shorter)
        arrow_length = 20
        draw.line([top_start_x, top_y, top_start_x + arrow_length, top_y - arrow_length], 
                  fill='black', width=line_thickness)
        draw.line([top_start_x, top_y, top_start_x + arrow_length, top_y + arrow_length], 
                  fill='black', width=line_thickness)
        
        draw.line([top_end_x, top_y, top_end_x - arrow_length, top_y - arrow_length], 
                  fill='black', width=line_thickness)
        draw.line([top_end_x, top_y, top_end_x - arrow_length, top_y + arrow_length], 
                  fill='black', width=line_thickness)
        
        # Bottom line with outward arrows (appears longer)
        bottom_y = center_y + 80
        bottom_line_length = 150  # Actually same length
        bottom_start_x = (img_size - bottom_line_length) // 2
        bottom_end_x = bottom_start_x + bottom_line_length
        
        # Draw main line
        draw.line([bottom_start_x, bottom_y, bottom_end_x, bottom_y], 
                  fill='black', width=line_thickness)
        
        # Outward arrows (make line appear longer)
        draw.line([bottom_start_x, bottom_y, bottom_start_x - arrow_length, bottom_y - arrow_length], 
                  fill='black', width=line_thickness)
        draw.line([bottom_start_x, bottom_y, bottom_start_x - arrow_length, bottom_y + arrow_length], 
                  fill='black', width=line_thickness)
        
        draw.line([bottom_end_x, bottom_y, bottom_end_x + arrow_length, bottom_y - arrow_length], 
                  fill='black', width=line_thickness)
        draw.line([bottom_end_x, bottom_y, bottom_end_x + arrow_length, bottom_y + arrow_length], 
                  fill='black', width=line_thickness)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path, 'PNG')
        return True
        
    except Exception as e:
        print(f"Error generating line illusion image: {e}")
        return False

# --- Dataset Generation ---
def generate_dataset(topic, num_images=5):
    """Generate test dataset for a topic"""
    print(f"Generating {topic} dataset...")
    
    dirs = create_directory_structure(topic)
    metadata = []
    
    # Get questions for this topic
    question_map = {
        'car_logos': CAR_LOGO_QUESTIONS,
        'animals': ANIMAL_QUESTIONS, 
        'adidas_stripes': ADIDAS_QUESTIONS,
        'optical_illusions': ILLUSION_QUESTIONS
    }
    questions = question_map[topic]
    
    for i in range(num_images):
        # Generate base image (notitle)
        base_filename = f"{topic}_{i+1}_notitle.png"
        titled_filename = f"{topic}_{i+1}_in_image_title.png"
        
        notitle_path = os.path.join(dirs['notitle']['images'], base_filename)
        titled_path = os.path.join(dirs['in_image_title']['images'], titled_filename)
        
        # Generate image based on topic
        success = False
        if topic == "car_logos":
            # Alternate between 4 (standard) and 5 (modified) rings
            num_rings = 5 if i % 2 == 0 else 4
            success = generate_audi_logo_image(notitle_path, num_rings)
        elif topic == "animals": 
            # Alternate between 4 (standard) and 5 (modified) legs
            num_legs = 5 if i % 2 == 0 else 4
            success = generate_zebra_image(notitle_path, num_legs)
        elif topic == "adidas_stripes":
            # Alternate between 3 (standard) and 4 (modified) stripes
            num_stripes = 4 if i % 2 == 0 else 3
            success = generate_adidas_shoe_image(notitle_path, num_stripes)
        else:  # optical_illusions
            # Lines that are actually different vs actually same
            equal_length = i % 2 == 1
            success = generate_line_illusion_image(notitle_path, equal_length)
        
        if not success:
            continue
            
        # Create titled version
        title_text = DEFAULT_TITLES[topic]
        add_title_to_image(notitle_path, titled_path, title_text)
        
        # Generate metadata for each question type
        for q in questions:
            # Notitle version
            metadata.append({
                "ID": f"{topic}_{i+1}_notitle_{q['type']}",
                "image_path": f"images/{base_filename}",
                "topic": topic,
                "prompt": q['prompt'],
                "ground_truth": q['ground_truth'], 
                "expected_bias": q['expected_bias'],
                "with_title": False,
                "type_of_question": q['type']
            })
            
            # Titled version
            metadata.append({
                "ID": f"{topic}_{i+1}_in_image_title_{q['type']}",
                "image_path": f"images/{titled_filename}",
                "topic": topic,
                "prompt": q['prompt'],
                "ground_truth": q['ground_truth'],
                "expected_bias": q['expected_bias'], 
                "with_title": True,
                "type_of_question": q['type']
            })
    
    # Save metadata
    for title_type in TITLE_TYPES:
        topic_metadata = [m for m in metadata if 
                         (title_type == 'notitle' and not m['with_title']) or
                         (title_type == 'in_image_title' and m['with_title'])]
        
        if topic_metadata:
            meta_file = os.path.join(dirs[title_type]['base'], 
                                   f"{topic}_{title_type}_metadata.json")
            with open(meta_file, 'w') as f:
                json.dump(topic_metadata, f, indent=2)
            print(f"Saved {len(topic_metadata)} metadata entries to {meta_file}")
    
    return len(metadata)

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="VLM Bias Testing Dataset Generator")
    parser.add_argument("--generate", choices=AVAILABLE_TOPICS + ['all'], 
                       help="Generate dataset")
    parser.add_argument("--num-images", type=int, default=5,
                       help="Number of images to generate per topic")
    
    args = parser.parse_args()
    
    if args.generate:
        print("=== VLM Bias Dataset Generator ===")
        topics = AVAILABLE_TOPICS if args.generate == 'all' else [args.generate]
        
        total_generated = 0
        for topic in topics:
            count = generate_dataset(topic, args.num_images)
            total_generated += count
            
        print(f"\nGenerated {total_generated} total samples across {len(topics)} topics")
        print("\nDatasets created:")
        for topic in topics:
            print(f"  - {topic}: Tests bias in {DEFAULT_TITLES[topic]} recognition")
        print("\nNext: Use test_gemini.py to evaluate with Gemini API")
        
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python dataset_generator.py --generate all")
        print("  python dataset_generator.py --generate car_logos")

if __name__ == "__main__":
    main()