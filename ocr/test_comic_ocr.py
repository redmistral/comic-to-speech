#!/usr/bin/env python3
"""
Test and compare different comic OCR methods
"""
import os
import sys
import time
from pathlib import Path
import subprocess

def test_ocr_method(script_name, image_path, method_name):
    """
    Test a specific OCR method and measure performance
    """
    print(f"\n{'='*60}")
    print(f"Testing: {method_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_name, image_path],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ Success! Completed in {elapsed:.2f} seconds")
            
            # Try to read the output file
            output_files = [
                output_dir / 'comic_text_extracted.txt',
                output_dir / 'comic_bubbles_text.txt', 
                output_dir / 'extracted_text_sorted.txt'
            ]
            
            for out_file in output_files:
                if out_file.exists():
                    with open(out_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    print(f"\nExtracted text preview (first 500 chars):")
                    print("-" * 40)
                    print(content[:500])
                    print("-" * 40)
                    
                    # Rename to preserve results
                    new_name = output_dir / f"{method_name.lower().replace(' ', '_')}_{out_file.name}"
                    out_file.rename(new_name)
                    print(f"Output saved to: {new_name}")
                    
                    return True, elapsed, len(content)
        else:
            print(f"✗ Failed with return code: {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr[:500]}")
            return False, elapsed, 0
            
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout after 60 seconds")
        return False, 60, 0
    except Exception as e:
        print(f"✗ Error: {e}")
        return False, 0, 0


def main():
    print("="*60)
    print("COMIC OCR METHODS COMPARISON TEST")
    print("="*60)
    
    # Create output directory
    output_dir = Path("ocr_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Check which scripts are available
    scripts = [
        ('comic_ocr_advanced.py', 'Panel-Aware OCR'),
        ('bubble_extractor.py', 'Bubble Detection'),
        ('ocr_sorted.py', 'Original Sorted OCR')
    ]
    
    available_scripts = []
    for script, name in scripts:
        if os.path.exists(script):
            available_scripts.append((script, name))
            print(f"✓ Found: {script}")
        else:
            print(f"✗ Missing: {script}")
    
    if not available_scripts:
        print("\nError: No OCR scripts found!")
        print("Please ensure the scripts are in the current directory")
        return 1
    
    # Get test image
    print(f"\nLooking for comic images...")
    
    # Check uploaded images first
    upload_dir = Path('/mnt/user-data/uploads')
    test_images = []
    
    if upload_dir.exists():
        for ext in ['.jpg', '.jpeg', '.png', '.PNG', '.webp']:
            test_images.extend(list(upload_dir.glob(f'*{ext}')))
    
    # Also check current directory
    for ext in ['.jpg', '.jpeg', '.png', '.PNG', '.webp']:
        test_images.extend(list(Path('.').glob(f'*{ext}')))
    
    # Filter out debug/output images
    test_images = [img for img in test_images 
                  if 'debug' not in str(img).lower() 
                  and 'output' not in str(img).lower()]
    
    if not test_images:
        print("No comic images found!")
        return 1
    
    print(f"\nFound {len(test_images)} test images:")
    for i, img in enumerate(test_images[:10]):
        print(f"  {i+1}. {img.name}")
    
    choice = input("\nSelect image to test (1-{}, or 'all' for all): ".format(len(test_images))).strip()
    
    if choice.lower() == 'all':
        selected_images = test_images
    else:
        try:
            idx = int(choice) - 1
            selected_images = [test_images[idx]]
        except:
            selected_images = [test_images[0]]
    
    # Run tests
    results = []
    
    for image_path in selected_images:
        print(f"\n{'#'*60}")
        print(f"TESTING IMAGE: {image_path.name}")
        print(f"{'#'*60}")
        
        for script, method_name in available_scripts:
            success, elapsed, text_length = test_ocr_method(
                script, str(image_path), method_name
            )
            
            results.append({
                'image': image_path.name,
                'method': method_name,
                'success': success,
                'time': elapsed,
                'text_chars': text_length
            })
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n{'Image':<30} {'Method':<20} {'Status':<10} {'Time':<10} {'Chars':<10}")
    print("-" * 80)
    
    for r in results:
        status = "✓ Success" if r['success'] else "✗ Failed"
        print(f"{r['image'][:29]:<30} {r['method']:<20} {status:<10} {r['time']:.2f}s{'':<6} {r['text_chars']:<10}")
    
    # Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    
    successful = [r for r in results if r['success']]
    if successful:
        # Find best by text extracted
        best_extraction = max(successful, key=lambda x: x['text_chars'])
        print(f"\n✓ Most text extracted: {best_extraction['method']}")
        print(f"  ({best_extraction['text_chars']} characters)")
        
        # Find fastest
        fastest = min(successful, key=lambda x: x['time'])
        print(f"\n✓ Fastest method: {fastest['method']}")
        print(f"  ({fastest['time']:.2f} seconds)")
        
        print("\nNext steps:")
        print("1. Review the output files to see which method works best")
        print("2. Check the extracted text for accuracy")
        print("3. Use the method that gives the best results for your comics")
    else:
        print("\n⚠ No methods succeeded. Troubleshooting tips:")
        print("1. Check that all dependencies are installed (run setup_comic_ocr.sh)")
        print("2. Ensure image files are readable")
        print("3. Check Google Cloud credentials if using Vision API")
        print("4. Try with a different comic image")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    exit(main())