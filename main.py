#!/usr/bin/env python3
"""
Text-to-Diagram Converter using Nano Banana Pro (Gemini 3 Pro Image)

Converts "How to" documentation into visual flowchart diagrams.
Supports auto-extraction of multiple workflows from a single document.
"""

import os
import re
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
import google.genai as genai
from google.genai import types
from google.genai.errors import ClientError

load_dotenv()

# Rate limiting settings
REQUEST_DELAY = 60  # seconds between requests (free tier is strict)
MAX_RETRIES = 3

# Model options:
# - "gemini-2.0-flash-preview-image-generation" (free tier available)
# - "imagen-3.0-generate-002" (paid tier)
DEFAULT_MODEL = "gemini-2.0-flash-preview-image-generation"
MODEL = DEFAULT_MODEL


def create_diagram_prompt(how_to_text: str) -> str:
    """Create a prompt optimized for diagram generation."""
    return f"""Create a clear, professional flowchart diagram for this process documentation.

Requirements:
- Use boxes/rectangles for steps
- Use diamonds for decision points
- Use arrows to show flow direction
- Include brief labels on each node
- Use a clean, minimal style with good contrast
- Vertical flow (top to bottom)
- Number the steps if sequential

Documentation to visualize:

{how_to_text}

Generate a flowchart diagram that makes this process easy to understand at a glance."""


def get_image_extension(data: bytes) -> str:
    """Detect image format from magic bytes."""
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return '.png'
    elif data[:2] == b'\xff\xd8':
        return '.jpg'
    elif data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return '.webp'
    elif data[:6] in (b'GIF87a', b'GIF89a'):
        return '.gif'
    return '.png'  # default fallback


def generate_diagram(text: str, output_path: str = "diagram", retries: int = MAX_RETRIES) -> str:
    """Generate a diagram from text using Nano Banana Pro with retry logic."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")

    client = genai.Client(api_key=api_key)

    prompt = create_diagram_prompt(text)

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)]
        )
    ]

    config = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        temperature=1.0
    )

    for attempt in range(retries):
        try:
            print(f"Generating diagram with {MODEL}...")

            # Stream and collect image data
            image_data = None
            for chunk in client.models.generate_content_stream(
                model=MODEL,
                contents=contents,
                config=config
            ):
                if chunk.candidates:
                    for part in chunk.candidates[0].content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            image_data = part.inline_data.data

            if image_data:
                # Detect actual format and use correct extension
                ext = get_image_extension(image_data)
                base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
                final_path = base_path + ext

                with open(final_path, 'wb') as f:
                    f.write(image_data)
                print(f"Diagram saved to: {final_path}")
                return final_path
            else:
                print("No image generated.")
                return None

        except ClientError as e:
            if e.code == 429:  # Rate limit
                # Extract retry delay from error if available
                wait_time = 60
                if 'retryDelay' in str(e):
                    import re
                    match = re.search(r'retry in (\d+)', str(e), re.IGNORECASE)
                    if match:
                        wait_time = int(match.group(1)) + 5

                if attempt < retries - 1:
                    print(f"Rate limited. Waiting {wait_time}s before retry ({attempt + 1}/{retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed after {retries} attempts due to rate limiting.")
                    return None
            else:
                print(f"API Error: {e}")
                return None

    return None


def extract_workflows(text: str) -> list[dict]:
    """
    Extract all 'How to' sections from documentation.
    Returns list of dicts with 'title' and 'content' keys.
    """
    # Pattern matches "### **How to..." or "### How to..." headers
    # and captures content until the next ### header or end of text
    pattern = r'###\s*\*?\*?\s*(How to[^*\n]+)\*?\*?\s*\n(.*?)(?=\n###|\n---|\Z)'

    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

    workflows = []
    for title, content in matches:
        title = title.strip().strip('*')
        content = content.strip()
        if content:  # Only include if there's actual content
            workflows.append({
                'title': title,
                'content': f"{title}\n\n{content}"
            })

    return workflows


def slugify(text: str) -> str:
    """Convert title to filename-safe slug."""
    # Remove "How to " prefix for shorter filenames
    text = re.sub(r'^how\s+to\s+', '', text, flags=re.IGNORECASE)
    # Convert to lowercase and replace spaces/special chars with underscores
    text = re.sub(r'[^a-z0-9]+', '_', text.lower())
    # Remove leading/trailing underscores
    return text.strip('_')


def generate_all_workflows(text: str, output_dir: str = "diagrams") -> list[str]:
    """
    Extract all workflows from text and generate diagrams for each.
    Returns list of generated file paths.
    """
    workflows = extract_workflows(text)

    if not workflows:
        print("No 'How to' sections found in the document.")
        return []

    print(f"Found {len(workflows)} workflow(s):")
    for i, w in enumerate(workflows, 1):
        print(f"  {i}. {w['title']}")

    total_time = len(workflows) * REQUEST_DELAY // 60
    print(f"\nEstimated time: ~{total_time} minutes (rate limit: {REQUEST_DELAY}s between requests)")
    print()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    generated = []
    failed = []
    for i, workflow in enumerate(workflows, 1):
        print(f"\n[{i}/{len(workflows)}] Generating: {workflow['title']}")
        filename = f"{i:02d}_{slugify(workflow['title'])}"
        filepath = output_path / filename

        result = generate_diagram(workflow['content'], str(filepath))
        if result:
            generated.append(result)
        else:
            failed.append(workflow['title'])

        # Delay between requests to avoid rate limits (except after last one)
        if i < len(workflows):
            print(f"Waiting {REQUEST_DELAY}s before next request...")
            time.sleep(REQUEST_DELAY)

    if failed:
        print(f"\nFailed to generate {len(failed)} diagram(s):")
        for title in failed:
            print(f"  - {title}")

    return generated


def main():
    global MODEL

    # Parse arguments
    auto_extract = '--auto' in sys.argv or '-a' in sys.argv

    # Check for model override
    for arg in sys.argv:
        if arg.startswith('--model='):
            MODEL = arg.split('=', 1)[1]
            print(f"Using model: {MODEL}")

    args = [a for a in sys.argv[1:] if not a.startswith('-')]

    if not args:
        input_file = Path(__file__).parent / "sample_input.txt"
    else:
        input_file = Path(args[0])

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    text = input_file.read_text()

    if auto_extract:
        # Auto-extract mode: find all "How to" sections and generate diagrams
        output_dir = args[1] if len(args) > 1 else f"{input_file.stem}_diagrams"
        generated = generate_all_workflows(text, output_dir)
        print(f"\n{'='*50}")
        print(f"Generated {len(generated)} diagram(s) in '{output_dir}/'")
    else:
        # Single diagram mode (original behavior)
        output_file = input_file.stem + "_diagram"
        generate_diagram(text, output_file)


if __name__ == "__main__":
    main()
