#!/usr/bin/env python3
"""
Text-to-Diagram Converter using Nano Banana Pro (Gemini 3 Pro Image)

Converts "How to" documentation into visual flowchart diagrams.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import google.genai as genai
from google.genai import types

load_dotenv()

# Model options:
# - "gemini-3-pro-image-preview" (Nano Banana Pro - requires paid tier)
# - "gemini-2.5-flash-image" (free tier available)
MODEL = "gemini-3-pro-image-preview"


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


def generate_diagram(text: str, output_path: str = "diagram") -> str:
    """Generate a diagram from text using Nano Banana Pro."""
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

    print(f"Generating diagram with Nano Banana Pro ({MODEL})...")

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
        # Remove any existing extension from output_path
        base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
        final_path = base_path + ext

        with open(final_path, 'wb') as f:
            f.write(image_data)
        print(f"Diagram saved to: {final_path}")
        return final_path
    else:
        print("No image generated.")
        return None


def main():
    if len(sys.argv) < 2:
        # Default: use sample input
        input_file = Path(__file__).parent / "sample_input.txt"
    else:
        input_file = Path(sys.argv[1])

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    text = input_file.read_text()
    output_file = input_file.stem + "_diagram"  # extension added by generate_diagram

    generate_diagram(text, output_file)


if __name__ == "__main__":
    main()
